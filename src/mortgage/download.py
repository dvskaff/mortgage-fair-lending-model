"""Download HMDA loan-level records from the CFPB Data Browser API and load them into DuckDB.

HMDA (Home Mortgage Disclosure Act) data covers nearly every U.S. mortgage application. The API returns one CSV per
request; each year is fetched separately and the newest years that are published are used automatically.
Column names have changed slightly between the documentation and the files (e.g. hyphens vs. underscores), so
names are normalized and every column the project needs is checked before anything is modeled.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import Config

log = logging.getLogger(__name__)

# Normalized name used here -> accepted names in the file (after lower-casing and replacing "-" with "_").
REQUIRED = {
    "activity_year": ("activity_year",),
    "lei": ("lei",),
    "action_taken": ("action_taken",),
    "loan_type": ("loan_type",),
    "loan_amount": ("loan_amount",),
    "combined_ltv": ("combined_loan_to_value_ratio", "loan_to_value_ratio"),
    "property_value": ("property_value",),
    "income": ("income",),
    "debt_to_income_ratio": ("debt_to_income_ratio",),
    "loan_term": ("loan_term",),
    "occupancy_type": ("occupancy_type",),
    "derived_race": ("derived_race",),
    "derived_ethnicity": ("derived_ethnicity",),
    "derived_sex": ("derived_sex",),
    "applicant_age": ("applicant_age", "ageapplicant"),
}
OPTIONAL = {
    "preapproval": ("preapproval",),
    "conforming_loan_limit": ("conforming_loan_limit",),
    "derived_dwelling_category": ("derived_dwelling_category",),
    "total_units": ("total_units",),
    "submission_of_application": ("submission_of_application",),
    "initially_payable_to_institution": ("initially_payable_to_institution",),
    "aus_1": ("aus_1",),
    "reverse_mortgage": ("reverse_mortgage",),
    "open_end_line_of_credit": ("open_end_line_of_credit",),
    "business_or_commercial_purpose": ("business_or_commercial_purpose",),
    "ffiec_msa_md_median_family_income": ("ffiec_msa_md_median_family_income",),
    "loan_purpose": ("loan_purpose",),
    "lien_status": ("lien_status",),
}


class SchemaError(RuntimeError):
    pass


def normalize(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def map_columns(header: list[str]) -> dict[str, str | None]:
    """Our column -> the file's column (None for a missing optional column)."""
    norm = {normalize(h): h for h in header}
    mapping, missing = {}, []
    for spec, required in ((REQUIRED, True), (OPTIONAL, False)):
        for col, candidates in spec.items():
            found = next((norm[c] for c in candidates if c in norm), None)
            if found is None and required:
                missing.append(col)
            mapping[col] = found
    if missing:
        raise SchemaError(f"HMDA file is missing {missing}. Columns present: {sorted(norm)}")
    return mapping


def session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=6, backoff_factor=3, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=["GET"]
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = "mortgage-fair-lending-model (github.com/dvskaff/mortgage-fair-lending-model)"
    return s


class Unavailable(Exception):
    """The API returned no data for a request (rejected or empty); carries what it said."""


def _download(cfg: Config, params: dict, dest: Path, http: requests.Session) -> None:
    with http.get(cfg.api_url, params=params, stream=True, timeout=900) as r:
        if 400 <= r.status_code < 500:
            raise Unavailable(f"HTTP {r.status_code}: {r.text[:300].strip()}")
        r.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    with dest.open("rb") as fh:
        head = [line for _, line in zip(range(2), fh, strict=False)]
    if len(head) < 2 or not head[0].lower().startswith(b"activity_year"):
        snippet = b"".join(head)[:300].decode("utf-8", "replace").strip()
        dest.unlink()
        raise Unavailable(f"no data rows (response began: {snippet!r})")


def fetch_year(cfg: Config, year: int, dest: Path, http: requests.Session) -> Path | None:
    """Download one year's CSV, or None if the API has no data for it (e.g. not published yet).

    The API's rules on combining filters are not fully documented, so if a request is rejected the filters are
    relaxed one at a time (the dropped ones are re-applied locally in prepare.py). Every rejection is logged."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    filters = list(cfg.filters.items())
    while True:
        params = {"states": cfg.state, "years": str(year), **dict(filters)}
        try:
            _download(cfg, params, dest, http)
        except Unavailable as exc:
            log.warning("HMDA %s with filters %s: %s", year, dict(filters), exc)
            if len(filters) > 1:
                filters = filters[:-1]
                continue
            return None
        log.info(
            "HMDA %s %s: %s MB (filters %s)",
            cfg.state,
            year,
            f"{dest.stat().st_size / 1e6:,.0f}",
            dict(filters),
        )
        return dest


def fetch_latest(
    cfg: Config, http: requests.Session | None = None, today: datetime | None = None
) -> dict[int, Path]:
    """The newest `n_years` published years (HMDA for year Y is released around spring of Y+1)."""
    http = http or session()
    year = (today or datetime.now(UTC)).year - 1
    files: dict[int, Path] = {}
    while len(files) < cfg.n_years and year >= cfg.first_year:
        try:
            path = fetch_year(cfg, year, cfg.data_dir / "raw" / f"hmda_{cfg.state}_{year}.csv", http)
        # One bad year should not stop the run (too few years still will).
        except requests.RequestException as exc:
            log.warning("HMDA %s could not be downloaded: %s", year, exc)
            path = None
        if path:
            files[year] = path
        else:
            log.info("No HMDA data for %s; trying %s", year, year - 1)
        year -= 1
    if len(files) < 2:
        raise RuntimeError(
            "Fewer than two years of HMDA data could be downloaded; see the warnings above for what the API said."
        )
    return dict(sorted(files.items()))


def load(cfg: Config, files: dict[int, Path]) -> int:
    """Create raw.lar with normalized column names (all text) from the yearly CSVs."""
    con = duckdb.connect(str(cfg.db_path))
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS raw").fetchall()
        selects = []
        for path in files.values():
            header = con.execute(
                "SELECT * FROM read_csv(?, all_varchar=true) LIMIT 0", [str(path)]
            ).description
            mapping = map_columns([c[0] for c in header])
            cols = ", ".join(
                f'"{src}" AS {col}' if src else f"NULL::VARCHAR AS {col}" for col, src in mapping.items()
            )
            selects.append(f"SELECT {cols} FROM read_csv('{path}', all_varchar=true, header=true)")
        con.execute("CREATE OR REPLACE TABLE raw.lar AS " + " UNION ALL ".join(selects)).fetchall()
        con.execute(
            """CREATE OR REPLACE TABLE raw.load_info AS SELECT ?::VARCHAR AS state,
                       ?::VARCHAR AS years, ?::TIMESTAMP AS loaded_at""",
            [cfg.state, ",".join(map(str, files)), datetime.now(UTC).replace(tzinfo=None)],
        ).fetchall()
        return con.execute("SELECT count(*) FROM raw.lar").fetchall()[0][0]
    finally:
        con.close()
