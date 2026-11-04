from dataclasses import replace
from datetime import UTC, datetime

import duckdb
import numpy as np
import pytest

from mortgage import download, fixture, prepare
from mortgage.config import Config

HEADER = fixture.make_year(
    2024, 5, np.random.default_rng(0), dict.fromkeys(fixture.LENDERS, 0.0)
).columns.tolist()


def test_map_columns_handles_hyphens_and_aliases():
    header = [h.replace("loan_to_value_ratio", "combined_loan_to_value_ratio") for h in HEADER]
    m = download.map_columns(header)
    assert m["combined_ltv"] == "combined_loan_to_value_ratio"
    assert m["aus_1"] == "aus-1"
    assert m["open_end_line_of_credit"] == "open-end_line_of_credit"
    assert download.map_columns(HEADER)["combined_ltv"] == "loan_to_value_ratio"


def test_map_columns_rejects_missing_required():
    with pytest.raises(download.SchemaError, match="debt_to_income_ratio"):
        download.map_columns([h for h in HEADER if h != "debt_to_income_ratio"])


class FakeResponse:
    def __init__(self, status: int, body: bytes):
        self.status_code, self.body = status, body
        self.text = body.decode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def iter_content(self, chunk_size):
        yield self.body


class FakeHTTP:
    """Rejects requests with 3 data filters; 2025 not published (400), 2024 header only, 2023-2021 have data."""

    def __init__(self):
        self.calls = []

    def get(self, url, params, stream, timeout):
        year = int(params["years"])
        self.calls.append((year, len(params) - 2))
        assert params["states"] == "GA" and params["actions_taken"] == "1,2,3"
        if len(params) - 2 > 2:
            return FakeResponse(400, b'{"errorType":"too-many-filters"}')
        if year == 2025:
            return FakeResponse(400, b"invalid year")
        if year == 2024:
            return FakeResponse(200, b"activity_year,lei\n")
        return FakeResponse(200, b"activity_year,lei\n%d,X\n" % year)


def test_fetch_latest_relaxes_filters_and_skips_unpublished_years(tmp_path, caplog):
    cfg = replace(Config(), data_dir=tmp_path)
    http = FakeHTTP()
    with caplog.at_level("WARNING"):
        files = download.fetch_latest(cfg, http, today=datetime(2026, 3, 1, tzinfo=UTC))
    assert list(files) == [2021, 2022, 2023]
    assert [c for c in http.calls if c[0] == 2023] == [(2023, 3), (2023, 2)]  # dropped lien_statuses, then OK
    assert [c for c in http.calls if c[0] == 2025] == [(2025, 3), (2025, 2), (2025, 1)]
    assert "too-many-filters" in caplog.text and "invalid year" in caplog.text
    assert not (tmp_path / "raw" / "hmda_GA_2024.csv").exists()


def test_json_error_with_status_200_is_not_data(tmp_path):
    class JsonHTTP:
        def get(self, url, params, stream, timeout):
            return FakeResponse(200, b'{"errorType":"x","message":"bad"}')

    cfg = replace(Config(), data_dir=tmp_path)
    assert download.fetch_year(cfg, 2024, tmp_path / "x.csv", JsonHTTP()) is None


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    d = tmp_path_factory.mktemp("prep")
    cfg = replace(Config(), data_dir=d)
    files = fixture.make_files(d / "raw", n_per_year=1500)
    raw = download.load(cfg, files)
    flow = prepare.build(cfg.db_path)
    return cfg, raw, flow, prepare.load_applications(cfg.db_path)


def test_load_and_filters(prepared):
    cfg, raw, flow, df = prepared
    assert raw == 4500
    assert flow["applications"].iloc[0] == raw
    assert flow["applications"].is_monotonic_decreasing
    assert flow["applications"].iloc[-2] == len(df)
    assert flow["applications"].iloc[-1] == df["complete_file"].sum() < len(df)
    complete = df[df["complete_file"]]
    assert complete[["dti", "cltv", "property_value_k", "income_k"]].notna().all().all()
    assert set(df["year"]) == set(fixture.YEARS)
    con = duckdb.connect(str(cfg.db_path), read_only=True)
    # excluded rows really are gone
    bad = con.execute("""SELECT count(*) FROM raw.lar WHERE occupancy_type <> '1' OR total_units = '5-24'
                         OR reverse_mortgage = '1'""").fetchall()[0][0]
    con.close()
    assert bad > 0 and len(df) <= raw - bad


def test_no_protected_or_post_decision_columns_in_features(prepared):
    from mortgage.model import FEATURES

    banned = (
        "race",
        "ethnic",
        "sex",
        "age",
        "interest",
        "denial_reason",
        "purchaser",
        "tract",
        "co_applicant",
        "payable",
    )
    assert not [f for f in FEATURES if any(b in f for b in banned)]


def test_dti_bands_and_codes(prepared):
    *_, df = prepared
    assert set(df["dti"].dropna().unique()) <= {15, 25, 33, 55, 65, *range(36, 50)}
    assert df["dti"].isna().any()  # "NA" becomes missing, not zero (and the file is not "complete")
    assert not df.loc[df["dti"].isna(), "complete_file"].any()
    assert df["income_k"].min() > 0
    assert set(df["denied"]) == {0, 1}
    assert set(df["race_ethnicity"]) <= {
        "Non-Hispanic White",
        "Black",
        "Hispanic or Latino",
        "Asian",
        "Not provided",
        "Other, multiple or joint",
    }
    # White applicants of Hispanic ethnicity are counted as Hispanic, not White
    assert "Hispanic or Latino" in set(df["race_ethnicity"])


def test_dti_sql_mapping():
    rows = duckdb.sql(f"""SELECT {prepare.DTI_SQL} FROM (VALUES ('<20%'), ('20%-<30%'), ('30%-<36%'), ('42'),
                          ('50%-60%'), ('>60%'), ('NA'), ('Exempt')) t(debt_to_income_ratio)""").fetchall()
    assert [r[0] for r in rows] == [15, 25, 33, 42, 55, 65, None, None]
