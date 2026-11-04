"""SYNTHETIC test data in the HMDA Data Browser CSV layout, used only by the tests and the offline CI run.

The relationships are planted so the tests can check that the pipeline recovers them (e.g. denial rises with
debt-to-income). None of these numbers describe real applicants or lenders; any page built from them carries a
"SYNTHETIC TEST DATA" banner.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

YEARS = (2022, 2023, 2024)

RACES = [  # (derived_race, derived_ethnicity, share, income multiplier, planted residual effect on log-odds)
    ("White", "Not Hispanic or Latino", 0.55, 1.00, 0.0),
    ("Black or African American", "Not Hispanic or Latino", 0.22, 0.80, 0.35),
    ("White", "Hispanic or Latino", 0.08, 0.85, 0.15),
    ("Asian", "Not Hispanic or Latino", 0.06, 1.15, 0.0),
    ("Race Not Available", "Ethnicity Not Available", 0.07, 1.00, 0.2),
    ("2 or more minority races", "Not Hispanic or Latino", 0.02, 0.95, 0.0),
]


def _dti_text(v: float) -> str:
    if np.isnan(v):
        return "NA"
    if v < 20:
        return "<20%"
    if v < 30:
        return "20%-<30%"
    if v < 36:
        return "30%-<36%"
    if v < 50:
        return str(int(v))
    return "50%-60%" if v <= 60 else ">60%"


LENDERS = [f"SYNTHLENDER{i:09d}" for i in range(40)]


def make_year(year: int, n: int, rng: np.random.Generator, lender_effect: dict[str, float]) -> pd.DataFrame:
    lenders = LENDERS
    lei = rng.choice(lenders, n, p=np.r_[np.full(10, 0.06), np.full(30, 0.4 / 30)])

    r = rng.choice(len(RACES), n, p=[x[2] for x in RACES])
    race = np.array([RACES[i][0] for i in r])
    eth = np.array([RACES[i][1] for i in r])
    inc_mult = np.array([RACES[i][3] for i in r])
    resid = np.array([RACES[i][4] for i in r])

    sex = rng.choice(["Male", "Female", "Joint", "Sex Not Available"], n, p=[0.38, 0.27, 0.30, 0.05])
    age = rng.choice(
        ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74", "8888"],
        n,
        p=[0.07, 0.33, 0.24, 0.14, 0.1, 0.07, 0.03, 0.02],
    )
    income = np.round(np.exp(rng.normal(np.log(85), 0.5, n)) * inc_mult)
    ltv = np.clip(rng.normal(88, 9, n), 30, 105)
    loan = np.maximum(5000 + 10000 * np.round(income * 1000 * rng.uniform(2, 5, n) / 10000), 15000)
    prop = 5000 + 10000 * np.round(loan / (ltv / 100) / 10000)
    dti = np.clip(rng.normal(37, 9, n), 5, 75)
    loan_type = rng.choice(["1", "2", "3", "4"], n, p=[0.62, 0.24, 0.1, 0.04])
    aus = rng.choice(["1", "2", "3", "6", "7"], n, p=[0.45, 0.2, 0.2, 0.1, 0.05])

    logit = (
        -2.6
        + 0.07 * (dti - 36)
        + 0.05 * (ltv - 88)
        - 0.6 * np.log(income / 85)
        + np.vectorize(lender_effect.get)(lei)
        + resid
        + 0.6 * (aus == "6")
        + 0.3 * (year - YEARS[0]) / 2
    )
    denied = rng.random(n) < 1 / (1 + np.exp(-logit))
    action = np.where(denied, "3", np.where(rng.random(n) < 0.03, "2", "1"))

    dti_s = np.array([_dti_text(v) for v in dti], dtype=object)
    dti_s[rng.random(n) < 0.04] = "NA"
    ltv_s = np.char.mod("%.3f", ltv).astype(object)
    ltv_s[rng.random(n) < 0.03] = "NA"
    income_s = income.astype(int).astype(str).astype(object)
    income_s[rng.random(n) < 0.02] = "NA"

    occupancy = rng.choice(["1", "2", "3"], n, p=[0.93, 0.05, 0.02])
    units = rng.choice(["1", "2", "5-24"], n, p=[0.97, 0.02, 0.01])
    df = pd.DataFrame(
        {
            "activity_year": year,
            "lei": lei,
            "derived_msa-md": "12060",
            "state_code": "GA",
            "county_code": "13121",
            "census_tract": "13121000100",
            "conforming_loan_limit": np.where(loan > 766550, "NC", "C"),
            "derived_dwelling_category": np.where(
                rng.random(n) < 0.04,
                "Single Family (1-4 Units):Manufactured",
                "Single Family (1-4 Units):Site-Built",
            ),
            "derived_ethnicity": eth,
            "derived_race": race,
            "derived_sex": sex,
            "action_taken": action,
            "purchaser_type": np.where(denied, "0", "1"),
            "preapproval": rng.choice(["1", "2"], n, p=[0.1, 0.9]),
            "loan_type": loan_type,
            "loan_purpose": "1",
            "lien_status": "1",
            "reverse_mortgage": np.where(rng.random(n) < 0.005, "1", "2"),
            "open-end_line_of_credit": "2",
            "business_or_commercial_purpose": "2",
            "loan_amount": loan.astype(int),
            "loan_to_value_ratio": ltv_s,
            "interest_rate": np.where(denied, "NA", "6.5"),
            "loan_term": "360",
            "property_value": prop.astype(int),
            "occupancy_type": occupancy,
            "total_units": units,
            "income": income_s,
            "debt_to_income_ratio": dti_s,
            "applicant_sex": "1",
            "co-applicant_sex": np.where(sex == "Joint", "2", "5"),
            "applicant_age": age,
            "applicant_age_above_62": "No",
            "submission_of_application": rng.choice(["1", "2"], n, p=[0.85, 0.15]),
            "initially_payable_to_institution": "1",
            "aus-1": aus,
            "denial_reason-1": np.where(denied, "1", "10"),
            "tract_minority_population_percent": "40.0",
            "ffiec_msa_md_median_family_income": "96400",
        }
    )
    return df


def make_files(raw_dir: Path, n_per_year: int = 5000, seed: int = 7) -> dict[int, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    lender_effect = dict(zip(LENDERS, rng.normal(0, 0.6, len(LENDERS)), strict=True))  # stable across years
    files = {}
    for year in YEARS:
        path = raw_dir / f"hmda_SYNTH_{year}.csv"
        make_year(year, n_per_year, rng, lender_effect).to_csv(path, index=False)
        files[year] = path
    return files
