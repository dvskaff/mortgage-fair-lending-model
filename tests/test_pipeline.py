import json
import shutil
from dataclasses import replace
from pathlib import Path

from mortgage import pipeline
from mortgage.config import Config
from mortgage.readme import update_readme

ROOT = Path(__file__).resolve().parents[1]


def test_end_to_end_on_synthetic_data(tmp_path):
    cfg = replace(
        Config(), data_dir=tmp_path / "data", site_dir=tmp_path / "site", readme_path=tmp_path / "README.md"
    )
    shutil.copy(ROOT / "README.md", cfg.readme_path)
    summary = pipeline.run(cfg, use_fixture=True)

    page = (cfg.site_dir / "index.html").read_text()
    assert "SYNTHETIC TEST DATA" in page
    assert "Fair-lending audit" in page and "Model card" in page
    results = json.loads((cfg.site_dir / "results.json").read_text())
    assert results["synthetic"] is True
    assert results["years"]["test"] == 2024
    assert "SYNTHETIC" in (cfg.site_dir / "model_card.md").read_text()

    # the fixture run never touches the README; the real run rewrites only the results block
    assert cfg.readme_path.read_text() == (ROOT / "README.md").read_text()
    update_readme(cfg.readme_path, summary)
    text = cfg.readme_path.read_text()
    assert "| LightGBM, monotonic + calibrated |" in text
    assert text.count("<!-- RESULTS:START -->") == 1
    before, after = (ROOT / "README.md").read_text().split("<!-- RESULTS:END -->")
    assert text.endswith(after)
