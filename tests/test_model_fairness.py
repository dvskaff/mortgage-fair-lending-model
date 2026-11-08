from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from mortgage import download, fairness, fixture, model, prepare
from mortgage.config import Config


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("model")
    cfg = replace(Config(), data_dir=d)
    download.load(cfg, fixture.make_files(d / "raw", n_per_year=4000))
    prepare.build(cfg.db_path)
    return prepare.load_applications(cfg.db_path)


@pytest.fixture(scope="module")
def result(data):
    return model.run(data, seed=1)


def test_split_is_out_of_time(data, result):
    assert result.years["train"] == [2022] and result.years["valid"] == 2023 and result.years["test"] == 2024
    assert set(result.test["year"]) == {2024}


def test_models_beat_chance_and_baseline(result):
    m = result.metrics.set_index("model")
    assert m.loc["LightGBM, monotonic + calibrated", "roc_auc"] > 0.65
    assert m.loc["WoE logistic scorecard", "roc_auc"] > 0.65
    assert (
        m.loc["LightGBM, monotonic + calibrated", "roc_auc"]
        > m.loc["Baseline: lender's past denial rate only", "roc_auc"]
    )
    assert m["gini"].sub(2 * m["roc_auc"] - 1).abs().max() < 1e-9


def test_planted_drivers_found(result):
    top = result.shap["importance"]["feature"].head(4).tolist()
    assert "dti" in top and "cltv" in top
    dep = result.shap["dependence"]["dti"]
    assert np.corrcoef(dep["x"], dep["shap"])[0, 1] > 0.5  # higher DTI -> higher denial log-odds


def test_monotonic_constraints_hold(data):
    train, (valid,) = model.add_lender_features(data[data.year == 2022], [data[data.year == 2023]], seed=0)
    booster, cats = model.fit_lgbm(train, valid, seed=0)
    base, _ = model.to_lgb_frame(valid.head(300), cats)
    for feature, sign in model.MONOTONE.items():
        preds = []
        for v in np.linspace(np.nanmin(base[feature]), np.nanmax(base[feature]), 25):
            x = base.copy()
            x[feature] = v
            preds.append(booster.predict(x, raw_score=True))
        assert (np.diff(np.vstack(preds), axis=0) * sign >= -1e-9).all(), feature


def test_lender_encoding_is_out_of_fold():
    # a lender with a single application: its own outcome must not leak into its feature
    train = pd.DataFrame({"lei": ["A"] * 50 + ["B"], "denied": [0] * 50 + [1]})
    enc, _ = model.add_lender_features(train, [], seed=0)
    prior = train["denied"].mean()
    assert enc.loc[50, "lender_denial_rate"] == pytest.approx(prior)


def test_binning_is_monotonic():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.uniform(0, 100, 20000))
    y = pd.Series((rng.random(20000) < 0.05 + 0.002 * x).astype(int))
    b = model.fit_binning(x, y, "x")
    rates = pd.Series(y.values).groupby(b.labels(x).values).mean().reindex(b.ordered_bins()).dropna()
    assert rates.is_monotonic_increasing
    assert 2 <= len(b.edges) + 1 <= 8


def test_scorecard_points_and_signs(result):
    pts = result.scorecard_points
    assert (pts["coefficient"] <= 0).all()
    dti = pts[pts.feature == "dti"]
    finite = dti[dti["bin"] != "Missing"]
    assert finite["points"].iloc[0] > finite["points"].iloc[-1]  # low DTI earns more points


def test_calibration_and_threshold(result):
    assert 0 < result.threshold < 1
    cal = result.calibration["LightGBM, monotonic + calibrated"]
    assert len(cal) == 10 and cal["observed"].is_monotonic_increasing is not None
    assert abs(result.test["pred_denied"].mean() - result.test["denied"].mean()) < 0.1


def test_wilson_interval():
    lo, hi = fairness.wilson(np.array([10]), np.array([100]))
    assert lo[0] == pytest.approx(0.0552, abs=1e-3) and hi[0] == pytest.approx(0.1744, abs=1e-3)


def test_audit_recovers_planted_gap(result):
    a = fairness.audit(result.test, min_n=100)
    race = a["race_ethnicity"]
    t = race["groups"].set_index("group")
    assert t.loc["Non-Hispanic White", "air_observed"] == pytest.approx(1)
    assert t.loc["Black", "denial_ratio"] > 1.2  # the fixture plants a residual gap for this group
    o = race["odds"].set_index("group")
    assert o.loc["Black", "odds_ratio_unadjusted"] > 1
    assert o.loc["Black", "lo_adjusted"] > 1
    s = race["summary"]
    assert 0 <= s["demographic_parity_difference"] <= 1 and 0 <= s["equalized_odds_difference"] <= 1
    assert set(a) == {"race_ethnicity", "sex", "age_group"}


def test_single_feature_leakage_check(result):
    sf = result.single_feature
    assert set(sf["feature"]) == set(model.FEATURES)
    assert sf["auc"].between(0.5, 1).all() and sf["auc"].is_monotonic_decreasing
    assert sf["auc"].iloc[0] < 0.8  # no single field gives the answer away in the fixture


def test_scorer_export_reproduces_the_scorecard(result, tmp_path):
    from mortgage import scorer

    card = scorer.export(result, result.years)
    t = result.test.head(300)
    mine = [scorer.score(card, r) for r in t.to_dict(orient="records")]
    assert np.allclose(mine, t["score"], atol=0.05)  # the page's arithmetic matches the fitted scorecard
    assert {f["feature"] for f in card["features"]} == set(result.scorecard.binnings)
    used = {f["feature"] for f in card["features"]}
    assert not used & {"race_ethnicity", "sex", "age_group"}  # protected characteristics are never inputs
    assert card["percentiles"] == sorted(card["percentiles"]) and len(card["bands"]) == 20
    assert sum(b["n"] for b in card["bands"]) == len(result.test)
    approved, denied = (
        card["presets"][k] for k in ("Typical approved application", "Typical denied application")
    )
    assert approved["dti"] <= denied["dti"]
    scorer.write_page(tmp_path, card, "https://example.org", "Georgia", synthetic=True)
    page = (tmp_path / "scorer.html").read_text()
    assert "SYNTHETIC TEST DATA" in page and "data/scorecard.json" in page and "not a credit decision" in page
