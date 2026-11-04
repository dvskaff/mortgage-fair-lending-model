"""End-to-end run: download HMDA -> DuckDB -> features -> models -> fairness audit -> site, model card and README.

python -m mortgage.pipeline              # real data from the CFPB HMDA Data Browser API
python -m mortgage.pipeline --fixture    # offline run on synthetic test data (CI and tests only)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path

from . import download, fairness, fixture, model, prepare
from .config import Config
from .readme import update_readme
from .report import build_site

log = logging.getLogger("mortgage")


def run(cfg: Config, *, use_fixture: bool = False) -> dict:
    t0 = time.time()
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract: the newest published years (or synthetic CSVs for tests).
    files = fixture.make_files(cfg.data_dir / "raw") if use_fixture else download.fetch_latest(cfg)
    log.info("Years: %s", list(files))

    # 2. Load and prepare in DuckDB.
    rows = download.load(cfg, files)
    flow = prepare.build(cfg.db_path)
    df = prepare.load_applications(cfg.db_path)
    log.info("Loaded %s raw rows; %s applications in the modeling sample", f"{rows:,}", f"{len(df):,}")

    # 3. Models and audit.
    res = model.run(df, seed=cfg.seed)
    audit = fairness.audit(res.test, min_n=100 if use_fixture else 300)

    # 4. Site, model card, README.
    summary = build_site(cfg, df, flow, res, audit, synthetic=use_fixture)
    if not use_fixture:
        update_readme(cfg.readme_path, summary)
        (cfg.readme_path.parent / "MODEL_CARD.md").write_text(summary["model_card"], encoding="utf-8")
    log.info("Done in %.0fs. Site: %s", time.time() - t0, cfg.site_dir / "index.html")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--fixture", action="store_true", help="offline run on synthetic test data")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config()
    if args.fixture:
        tmp = Path(os.environ.get("MORTGAGE_FIXTURE_DIR", "/tmp/mortgage-fixture")).resolve()
        shutil.rmtree(tmp, ignore_errors=True)
        cfg = replace(cfg, data_dir=tmp, site_dir=tmp / "site")
    s = run(cfg, use_fixture=args.fixture)
    main_model = next(m for m in s["metrics"] if m["model"].startswith("LightGBM, monotonic"))
    print(
        json.dumps(
            {
                "years": s["years"],
                "n_applications": s["n_applications"],
                "roc_auc": round(main_model["roc_auc"], 4),
                "headline": s["headline"],
            },
            default=float,
        )
    )


if __name__ == "__main__":
    main()
