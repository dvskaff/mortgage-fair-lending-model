"""Turn raw HMDA records into one clean row per application: the modeling sample, features and audit groups.

Only information a lender has when it makes the decision is used as a feature. Three kinds of columns are left out:
* protected characteristics (race, ethnicity, sex, age): they are kept only to audit the model afterwards;
* geography and census-tract demographics (they can act as proxies for race);
* anything recorded after the decision (interest rate, fees, denial reasons, who bought the loan).
Whether there is a co-applicant is also left out, because it can stand in for marital status, which the Equal Credit
Opportunity Act protects (the combined income of both applicants is still used).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

# HMDA reports debt-to-income in bands below 36% and above 49% (exact values in between).
DTI_SQL = """CASE trim(debt_to_income_ratio)
    WHEN '<20%' THEN 15 WHEN '20%-<30%' THEN 25 WHEN '30%-<36%' THEN 33
    WHEN '50%-60%' THEN 55 WHEN '>60%' THEN 65
    ELSE try_cast(debt_to_income_ratio AS DOUBLE) END"""

# Audit groups. Hispanic or Latino applicants of any race form one group (the usual fair-lending convention);
# the reference group is non-Hispanic White.
RACE_SQL = """CASE
    WHEN derived_ethnicity = 'Hispanic or Latino' THEN 'Hispanic or Latino'
    WHEN derived_race = 'White' AND derived_ethnicity = 'Not Hispanic or Latino' THEN 'Non-Hispanic White'
    WHEN derived_race = 'Black or African American' THEN 'Black'
    WHEN derived_race = 'Asian' THEN 'Asian'
    WHEN derived_race = 'Race Not Available' OR derived_race IS NULL THEN 'Not provided'
    ELSE 'Other, multiple or joint' END"""
SEX_SQL = """CASE derived_sex WHEN 'Male' THEN 'Male' WHEN 'Female' THEN 'Female' WHEN 'Joint' THEN 'Joint (two applicants)'
    ELSE 'Not provided' END"""
AGE_SQL = """CASE applicant_age
    WHEN '<25' THEN 'Under 35' WHEN '25-34' THEN 'Under 35' WHEN '35-44' THEN '35-54' WHEN '45-54' THEN '35-54'
    WHEN '55-64' THEN '55-64' WHEN '65-74' THEN '65 or older' WHEN '>74' THEN '65 or older'
    ELSE 'Not provided' END"""

REFERENCE = {"race_ethnicity": "Non-Hispanic White", "sex": "Male", "age_group": "35-54"}
NOT_PROVIDED = "Not provided"
AUDIT_COLUMNS = list(REFERENCE)

AUS = {
    "1": "Fannie Mae DU",
    "2": "Freddie Mac LP",
    "3": "FHA TOTAL",
    "4": "USDA GUS",
    "5": "Other",
    "6": "None used",
    "7": "Lender proprietary system",
    "1111": "Exempt",
}
LOAN_TYPE = {"1": "Conventional", "2": "FHA", "3": "VA", "4": "USDA"}


def _code_case(column: str, mapping: dict[str, str], other: str = "Not reported") -> str:
    whens = " ".join(f"WHEN '{k}' THEN '{v}'" for k, v in mapping.items())
    return f"CASE trim({column}) {whens} ELSE '{other}' END"


def build(db_path: Path) -> pd.DataFrame:
    """Create main.applications and main.sample_flow in DuckDB. Returns the sample-flow table."""
    con = duckdb.connect(str(db_path))
    try:
        con.execute(f"""
CREATE OR REPLACE TABLE main.typed AS
SELECT
    try_cast(activity_year AS INTEGER)                         AS year,
    lei,
    try_cast(action_taken AS INTEGER)                          AS action_taken,
    (try_cast(action_taken AS INTEGER) = 3)::INTEGER           AS denied,
    try_cast(occupancy_type AS INTEGER)                        AS occupancy_type,
    coalesce(trim(loan_purpose), '1')                          AS loan_purpose,
    coalesce(trim(lien_status), '1')                           AS lien_status,
    coalesce(trim(reverse_mortgage), '2')                      AS reverse_mortgage,
    coalesce(trim(open_end_line_of_credit), '2')               AS open_end,
    coalesce(trim(business_or_commercial_purpose), '2')        AS business_purpose,
    coalesce(trim(total_units), '1')                           AS total_units_raw,
    try_cast(loan_amount AS DOUBLE)                            AS loan_amount,
    try_cast(property_value AS DOUBLE)                         AS property_value,
    try_cast(combined_ltv AS DOUBLE)                           AS cltv_raw,
    try_cast(income AS DOUBLE)                                 AS income_k,
    {DTI_SQL}                                                  AS dti,
    try_cast(loan_term AS DOUBLE)                              AS loan_term,
    try_cast(ffiec_msa_md_median_family_income AS DOUBLE)      AS area_median_income,
    {_code_case("loan_type", LOAN_TYPE)}                       AS loan_type,
    CASE trim(preapproval) WHEN '1' THEN 'Requested' WHEN '2' THEN 'Not requested' ELSE 'Not reported' END
                                                               AS preapproval,
    coalesce(nullif(trim(conforming_loan_limit), ''), 'NA')    AS conforming_loan_limit,
    CASE WHEN derived_dwelling_category ILIKE '%manufactured%' THEN 'Manufactured home'
         WHEN derived_dwelling_category ILIKE 'multifamily%' THEN 'Multifamily (5+ units)'
         ELSE 'Site-built, 1-4 units' END                      AS dwelling,
    {_code_case("submission_of_application", {"1": "Direct", "2": "Not direct (e.g. broker)", "1111": "Exempt"})}
                                                               AS submission,
    {_code_case("initially_payable_to_institution", {"1": "Yes", "2": "No", "1111": "Exempt"})}
                                                               AS payable_to_lender,
    {_code_case("aus_1", AUS)}                                 AS aus,
    {RACE_SQL}                                                 AS race_ethnicity,
    {SEX_SQL}                                                  AS sex,
    {AGE_SQL}                                                  AS age_group,
    derived_race, derived_ethnicity, derived_sex, applicant_age
FROM raw.lar""").fetchall()

        steps = [
            ("Approved or denied applications (as downloaded)", "TRUE"),
            ("Home purchase, first lien", "loan_purpose = '1' AND lien_status = '1'"),
            ("Principal residence", "occupancy_type = 1"),
            (
                "Not a reverse mortgage, open-end line or business loan",
                "reverse_mortgage <> '1' AND open_end <> '1' AND business_purpose <> '1'",
            ),
            ("1-4 unit property", "total_units_raw IN ('1', '2', '3', '4')"),
            (
                "Valid loan amount and decision",
                "loan_amount > 0 AND action_taken IN (1, 2, 3) AND year IS NOT NULL",
            ),
        ]
        flow, cond = [], []
        for label, c in steps:
            cond.append(f"({c})")
            n, d = con.execute(
                f"SELECT count(*), sum(denied) FROM main.typed WHERE {' AND '.join(cond)}"
            ).fetchall()[0]
            flow.append({"step": label, "applications": n, "denied": int(d or 0)})

        con.execute(f"""
CREATE OR REPLACE TABLE main.applications AS
SELECT
    row_number() OVER (ORDER BY year, lei, loan_amount)        AS app_id,
    year, lei, denied, action_taken,
    loan_amount / 1000                                         AS loan_amount_k,
    property_value / 1000                                      AS property_value_k,
    CASE WHEN cltv_raw > 0 AND cltv_raw <= 250 THEN cltv_raw END AS cltv,
    dti,
    CASE WHEN income_k > 0 THEN income_k END                   AS income_k,
    CASE WHEN income_k > 0 THEN least(loan_amount / (income_k * 1000), 25) END AS loan_to_income,
    CASE WHEN income_k > 0 AND area_median_income > 0
         THEN least(income_k * 1000 / area_median_income, 25) END AS income_to_area_median,
    CASE WHEN loan_term > 0 AND loan_term <= 600 THEN loan_term END AS loan_term,
    try_cast(total_units_raw AS INTEGER)                       AS total_units,
    loan_type, preapproval, conforming_loan_limit, dwelling, submission, payable_to_lender, aus,
    race_ethnicity, sex, age_group
FROM main.typed
WHERE {" AND ".join(cond)}""").fetchall()
        con.execute("DROP TABLE main.typed").fetchall()
        flow_df = pd.DataFrame(flow)
        con.register("flow_df", flow_df)
        con.execute("CREATE OR REPLACE TABLE main.sample_flow AS SELECT * FROM flow_df").fetchall()
        return flow_df
    finally:
        con.close()


def load_applications(db_path: Path) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute("SELECT * FROM main.applications ORDER BY app_id").df()
    finally:
        con.close()
