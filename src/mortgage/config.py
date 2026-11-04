"""Project settings. Paths can be overridden with environment variables (used by tests and CI)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _path(env: str, default: Path) -> Path:
    return Path(os.environ[env]).resolve() if os.environ.get(env) else default


@dataclass
class Config:
    # CFPB's public HMDA Data Browser API (no key or account needed).
    api_url: str = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"
    state: str = "GA"  # Georgia: a large, diverse mortgage market
    state_name: str = "Georgia"
    n_years: int = 3  # the newest available years: oldest two train the models, the newest is held out
    first_year: int = 2018  # the current HMDA format starts in 2018
    # Home-purchase, first-lien loans that were approved (1, 2) or denied (3).
    filters: dict[str, str] = field(
        default_factory=lambda: {"actions_taken": "1,2,3", "loan_purposes": "1", "lien_statuses": "1"}
    )

    seed: int = 42
    data_dir: Path = field(default_factory=lambda: _path("MORTGAGE_DATA_DIR", ROOT / "data"))
    site_dir: Path = field(default_factory=lambda: _path("MORTGAGE_SITE_DIR", ROOT / "site"))
    readme_path: Path = field(default_factory=lambda: ROOT / "README.md")

    repo_url: str = "https://github.com/dvskaff/mortgage-fair-lending-model"
    site_url: str = "https://dvskaff.github.io/mortgage-fair-lending-model/"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "hmda.duckdb"
