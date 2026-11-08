"""Denial models: a weight-of-evidence logistic scorecard and a monotonic, calibrated LightGBM, trained on older years
and evaluated on the newest year (an out-of-time test, the way a lender would use a model)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import KFold

log = logging.getLogger(__name__)

NUMERIC = [
    "loan_amount_k",
    "property_value_k",
    "cltv",
    "dti",
    "income_k",
    "loan_to_income",
    "income_to_area_median",
    "loan_term",
    "total_units",
    "lender_denial_rate",
    "lender_log_volume",
]
CATEGORICAL = [
    "loan_type",
    "preapproval",
    "conforming_loan_limit",
    "dwelling",
    "submission",
    "aus",
]
FEATURES = NUMERIC + CATEGORICAL
# Directions a credit officer would sign off on: more debt or leverage never lowers the risk; more income never raises it.
MONOTONE = {"dti": 1, "cltv": 1, "loan_to_income": 1, "income_k": -1, "income_to_area_median": -1}

LABELS = {
    "loan_amount_k": "Loan amount ($K)",
    "property_value_k": "Property value ($K)",
    "cltv": "Combined loan-to-value (%)",
    "dti": "Debt-to-income (%)",
    "income_k": "Income ($K)",
    "loan_to_income": "Loan ÷ income",
    "income_to_area_median": "Income ÷ area median",
    "loan_term": "Loan term (months)",
    "total_units": "Units",
    "lender_denial_rate": "Lender's past denial rate",
    "lender_log_volume": "Lender size (log applications)",
    "loan_type": "Loan type",
    "preapproval": "Preapproval",
    "conforming_loan_limit": "Conforming limit",
    "dwelling": "Dwelling type",
    "submission": "Application channel",
    "payable_to_lender": "Payable to lender",
    "aus": "Underwriting system",
}


# --------------------------------------------------------------------------- splits and lender encoding
def split_years(df: pd.DataFrame) -> tuple[list[int], int, int]:
    years = sorted(df["year"].unique())
    if len(years) < 3:
        raise ValueError(f"Need three years (train, validation, test); got {years}")
    return [int(y) for y in years[:-2]], int(years[-2]), int(years[-1])


def add_lender_features(train: pd.DataFrame, others: list[pd.DataFrame], seed: int, m: float = 50.0):
    """Each lender's smoothed historical denial rate and size, learned from the training years only.
    Training rows get out-of-fold values so a row's own outcome never feeds its feature (target leakage)."""
    prior = train["denied"].mean()

    def encode(fit: pd.DataFrame, apply: pd.DataFrame) -> pd.DataFrame:
        g = fit.groupby("lei")["denied"].agg(["sum", "count"])
        rate = (g["sum"] + m * prior) / (g["count"] + m)
        out = pd.DataFrame(index=apply.index)
        out["lender_denial_rate"] = apply["lei"].map(rate).fillna(prior).astype(float)
        out["lender_log_volume"] = np.log1p(apply["lei"].map(g["count"]).fillna(0).astype(float))
        return out

    train = train.copy()
    enc = pd.DataFrame(index=train.index, columns=["lender_denial_rate", "lender_log_volume"], dtype=float)
    for fit_idx, oof_idx in KFold(5, shuffle=True, random_state=seed).split(train):
        enc.iloc[oof_idx] = encode(train.iloc[fit_idx], train.iloc[oof_idx]).to_numpy()
    enc["lender_log_volume"] = np.log1p(train["lei"].map(train["lei"].value_counts()).astype(float))
    train[enc.columns] = enc
    out = []
    for o in others:
        o = o.copy()
        o[["lender_denial_rate", "lender_log_volume"]] = encode(train, o)
        out.append(o)
    return train, out


# --------------------------------------------------------------------------- metrics
def ks_stat(y: np.ndarray, p: np.ndarray) -> float:
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    auc = roc_auc_score(y, p)
    return {
        "roc_auc": auc,
        "gini": 2 * auc - 1,
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
        "ks": ks_stat(y, p),
        "observed_rate": float(y.mean()),
        "mean_predicted": float(p.mean()),
        "n": int(len(y)),
    }


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    q = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False)
    t = (
        pd.DataFrame({"bin": q, "y": y, "p": p})
        .groupby("bin")
        .agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    )
    return t.reset_index()


# --------------------------------------------------------------------------- WoE scorecard
@dataclass
class Binning:
    feature: str
    kind: str  # "numeric" | "categorical"
    edges: list[float] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    woe: dict[str, float] = field(default_factory=dict)

    def labels(self, x: pd.Series) -> pd.Series:
        if self.kind == "numeric":
            idx = np.searchsorted(self.edges, x.to_numpy(dtype=float), side="right")
            lab = pd.Series([self._interval(i) for i in idx], index=x.index)
            return lab.where(x.notna(), "Missing")
        return x.astype(str).where(x.astype(str).isin(self.categories), "Other")

    def _interval(self, i: int) -> str:
        lo = "−∞" if i == 0 else f"{self.edges[i - 1]:,.3g}"
        hi = "∞" if i == len(self.edges) else f"{self.edges[i]:,.3g}"
        return f"[{lo}, {hi})"

    def ordered_bins(self) -> list[str]:
        if self.kind == "numeric":
            order = [self._interval(i) for i in range(len(self.edges) + 1)]
            return [b for b in [*order, "Missing"] if b in self.woe]
        return sorted(self.woe, key=lambda k: -self.woe[k])

    def transform(self, x: pd.Series) -> np.ndarray:
        return self.labels(x).map(self.woe).fillna(0.0).to_numpy(dtype=float)


def _woe(labels: pd.Series, y: pd.Series) -> dict[str, float]:
    t = pd.crosstab(labels, y).reindex(columns=[0, 1], fill_value=0)
    good, bad = t[0] + 0.5, t[1] + 0.5
    return np.log((good / good.sum()) / (bad / bad.sum())).to_dict()


def fit_binning(x: pd.Series, y: pd.Series, name: str, max_bins: int = 8, min_share: float = 0.05) -> Binning:
    if not pd.api.types.is_numeric_dtype(x):
        share = x.astype(str).value_counts(normalize=True)
        b = Binning(name, "categorical", categories=list(share[share >= 0.01].index))
        b.woe = _woe(b.labels(x), y)
        return b
    ok = x.notna()
    edges = sorted(set(np.nanquantile(x[ok], np.linspace(0, 1, max_bins + 1)[1:-1]).round(6)))
    rho = spearmanr(x[ok], y[ok]).statistic if x[ok].nunique() > 1 else np.nan
    direction = float(np.sign(rho)) if np.isfinite(rho) and rho != 0 else 1.0
    while edges:  # merge until every bin is big enough and the denial rate moves in one direction
        idx = np.searchsorted(edges, x[ok].to_numpy(dtype=float), side="right")
        stats = pd.DataFrame({"b": idx, "y": y[ok].to_numpy()}).groupby("b")["y"].agg(["mean", "size"])
        stats = stats.reindex(range(len(edges) + 1)).fillna({"mean": 0, "size": 0})
        small = stats["size"] < min_share * ok.sum()
        bad = (np.diff(stats["mean"].to_numpy()) * direction) < 0
        if small.any():
            i = int(np.argmax(small.to_numpy()))
            edges.pop(min(i, len(edges) - 1))
        elif bad.any():
            edges.pop(int(np.argmax(bad)))
        else:
            break
    b = Binning(name, "numeric", edges=[float(e) for e in edges])
    b.woe = _woe(b.labels(x), y)
    return b


@dataclass
class Scorecard:
    binnings: dict[str, Binning]
    model: LogisticRegression
    iv_all: pd.DataFrame | None = None
    pdo: float = 20.0
    base_score: float = 600.0
    base_odds: float = 20.0  # approval:denial odds at the base score

    def woe_matrix(self, df: pd.DataFrame) -> np.ndarray:
        return np.column_stack([b.transform(df[f]) for f, b in self.binnings.items()])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.woe_matrix(df))[:, 1]

    def points_table(self) -> pd.DataFrame:
        """Points per bin: score = sum of points; higher = more likely to be approved."""
        factor = self.pdo / np.log(2)
        offset = self.base_score - factor * np.log(self.base_odds)
        n = len(self.binnings)
        b0 = self.model.intercept_[0]
        rows = []
        for coef, (f, b) in zip(self.model.coef_[0], self.binnings.items(), strict=True):
            for label in b.ordered_bins():
                w = b.woe[label]
                pts = -(coef * w + b0 / n) * factor + offset / n
                rows.append(
                    {
                        "feature": f,
                        "label": LABELS.get(f, f),
                        "bin": label,
                        "woe": w,
                        "coefficient": coef,
                        "points": pts,
                    }
                )
        return pd.DataFrame(rows)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        factor = self.pdo / np.log(2)
        offset = self.base_score - factor * np.log(self.base_odds)
        logit = self.model.decision_function(self.woe_matrix(df))
        return offset - factor * logit


def fit_scorecard(train: pd.DataFrame, features: list[str], min_iv: float = 0.02) -> Scorecard:
    """Bin every feature, keep those with information value >= min_iv, then drop features whose coefficient has the
    wrong sign (a sign flip means the feature only corrects for a correlated one, which makes the points unreadable)."""
    y = train["denied"]
    binnings = {f: fit_binning(train[f], y, f) for f in features}
    sc = Scorecard(binnings, LogisticRegression(C=1.0, max_iter=2000))
    sc.iv_all = information_value(train, sc)
    keep = [f for f in sc.iv_all.query("iv >= @min_iv")["feature"]]
    while True:
        sc.binnings = {f: binnings[f] for f in keep}
        sc.model = LogisticRegression(C=1.0, max_iter=2000).fit(sc.woe_matrix(train), y)
        coefs = pd.Series(sc.model.coef_[0], index=keep)
        if (coefs <= 0).all() or len(keep) == 1:
            break
        keep.remove(coefs.idxmax())  # WoE is high for low-risk bins, so every coefficient should be negative
    return sc


def information_value(train: pd.DataFrame, sc: Scorecard) -> pd.DataFrame:
    rows = []
    y = train["denied"]
    for f, b in sc.binnings.items():
        t = pd.crosstab(b.labels(train[f]), y).reindex(columns=[0, 1], fill_value=0) + 0.5
        g, d = t[0] / t[0].sum(), t[1] / t[1].sum()
        rows.append(
            {
                "feature": f,
                "label": LABELS.get(f, f),
                "iv": float(((g - d) * np.log(g / d)).sum()),
                "bins": len(t),
            }
        )
    return pd.DataFrame(rows).sort_values("iv", ascending=False)


# --------------------------------------------------------------------------- LightGBM
def to_lgb_frame(df: pd.DataFrame, categories: dict[str, list[str]] | None = None):
    x = df[FEATURES].copy()
    cats = categories or {c: sorted(x[c].astype(str).unique()) for c in CATEGORICAL}
    for c in CATEGORICAL:
        x[c] = pd.Categorical(x[c].astype(str), categories=cats[c])
    for c in NUMERIC:
        x[c] = x[c].astype(float)
    return x, cats


def fit_lgbm(train: pd.DataFrame, valid: pd.DataFrame, seed: int, monotone: bool = True):
    xt, cats = to_lgb_frame(train)
    xv, _ = to_lgb_frame(valid, cats)
    params = dict(
        objective="binary",
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=100,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=1,
        lambda_l2=5.0,
        cat_smooth=20,
        seed=seed,
        verbose=-1,
        deterministic=True,
        force_row_wise=True,
    )
    if monotone:
        params["monotone_constraints"] = [MONOTONE.get(f, 0) for f in FEATURES]
        params["monotone_constraints_method"] = "intermediate"  # "advanced" broke monotonicity in tests
    dtrain = lgb.Dataset(xt, train["denied"], categorical_feature=CATEGORICAL)
    dvalid = lgb.Dataset(xv, valid["denied"], reference=dtrain)
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return booster, cats


@dataclass
class Result:
    years: dict
    metrics: pd.DataFrame
    curves: dict
    calibration: dict
    test: pd.DataFrame  # test-year rows with predictions (and audit groups)
    threshold: float
    scorecard_points: pd.DataFrame
    iv: pd.DataFrame
    shap: dict
    booster_info: dict
    single_feature: pd.DataFrame | None = None
    scorecard: Scorecard | None = (
        None  # the fitted scorecard itself (bins, coefficients), for the online scorer
    )


def single_feature_auc(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Leakage check: how well does each feature alone rank the held-out year? A single application-time field
    that nearly separates denials from approvals is more likely a reporting artifact than a credit signal."""
    rows = []
    for f in FEATURES:
        if f in CATEGORICAL:
            rates = train.groupby(train[f].astype(str))["denied"].mean()
            s = test[f].astype(str).map(rates).fillna(train["denied"].mean())
        else:
            s = test[f].astype(float).fillna(test[f].astype(float).median())
        auc = roc_auc_score(test["denied"], s) if s.nunique() > 1 else 0.5
        rows.append({"feature": f, "label": LABELS.get(f, f), "auc": max(auc, 1 - auc)})
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def explain(booster, x: pd.DataFrame, seed: int, n: int = 4000) -> dict:
    """TreeSHAP values (log-odds) for a sample of test applications."""
    import warnings

    import shap

    sample = x.sample(min(n, len(x)), random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = shap.TreeExplainer(booster).shap_values(sample)
    values = values[1] if isinstance(values, list) else values
    imp = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": np.abs(values).mean(axis=0)})
    imp["label"] = imp["feature"].map(LABELS)
    dep = {}
    for f in ("dti", "cltv", "loan_to_income", "lender_denial_rate"):
        j = FEATURES.index(f)
        keep = sample[f].notna().to_numpy()
        dep[f] = {
            "x": sample[f].to_numpy()[keep].round(3).tolist(),
            "shap": values[keep, j].round(4).tolist(),
        }
    return {
        "importance": imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True),
        "dependence": dep,
        "n": len(sample),
    }


def run(df: pd.DataFrame, seed: int = 42) -> Result:
    train_years, valid_year, test_year = split_years(df)
    train = df[df["year"].isin(train_years)]
    valid = df[df["year"] == valid_year]
    test = df[df["year"] == test_year]
    train, (valid, test) = add_lender_features(train, [valid, test], seed)
    log.info(
        "Train %s: %s rows · validation %s: %s · test %s: %s",
        train_years,
        f"{len(train):,}",
        valid_year,
        f"{len(valid):,}",
        test_year,
        f"{len(test):,}",
    )

    sc = fit_scorecard(train, FEATURES)
    booster, cats = fit_lgbm(train, valid, seed)
    free, _ = fit_lgbm(train, valid, seed, monotone=False)
    xv, _ = to_lgb_frame(valid, cats)
    xs, _ = to_lgb_frame(test, cats)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
    iso.fit(booster.predict(xv, num_iteration=booster.best_iteration), valid["denied"])

    yt = test["denied"].to_numpy()
    raw = booster.predict(xs, num_iteration=booster.best_iteration)
    preds = {
        "LightGBM, monotonic + calibrated": iso.predict(raw),
        "LightGBM, unconstrained": free.predict(xs, num_iteration=free.best_iteration),
        "WoE logistic scorecard": sc.predict(test),
        "Baseline: lender's past denial rate only": test["lender_denial_rate"].to_numpy(),
    }
    rows = [{"model": k, **metrics(yt, p)} for k, p in preds.items()]
    val_metrics = metrics(
        valid["denied"].to_numpy(), iso.predict(booster.predict(xv, num_iteration=booster.best_iteration))
    )

    main = preds["LightGBM, monotonic + calibrated"]
    # Decision rule: flag the riskiest applications so the flagged share on the validation year equals the share
    # actually denied that year (keeps the model's overall denial rate in line with lenders' decisions).
    pv = iso.predict(booster.predict(xv, num_iteration=booster.best_iteration))
    threshold = float(np.quantile(pv, 1 - valid["denied"].mean()))

    out = test.copy()
    out["p_denial"] = main
    out["p_scorecard"] = preds["WoE logistic scorecard"]
    out["score"] = sc.score(test)
    raw_logit = booster.predict(xs, num_iteration=booster.best_iteration, raw_score=True)
    out["model_logit"] = raw_logit
    out["pred_denied"] = (main >= threshold).astype(int)

    curves = {}
    for k in (
        "LightGBM, monotonic + calibrated",
        "WoE logistic scorecard",
        "Baseline: lender's past denial rate only",
    ):
        fpr, tpr, _ = roc_curve(yt, preds[k])
        keep = np.unique(np.linspace(0, len(fpr) - 1, min(len(fpr), 300)).astype(int))
        curves[k] = {"fpr": fpr[keep].round(4).tolist(), "tpr": tpr[keep].round(4).tolist()}

    return Result(
        years={
            "train": train_years,
            "valid": valid_year,
            "test": test_year,
            "n_train": len(train),
            "n_valid": len(valid),
            "n_test": len(test),
        },
        metrics=pd.DataFrame(rows),
        curves=curves,
        calibration={
            k: calibration_table(yt, preds[k])
            for k in ("LightGBM, monotonic + calibrated", "WoE logistic scorecard")
        },
        test=out,
        threshold=threshold,
        scorecard_points=sc.points_table(),
        iv=sc.iv_all.assign(in_scorecard=lambda d: d["feature"].isin(list(sc.binnings))),
        shap=explain(booster, xs, seed),
        single_feature=single_feature_auc(train, test),
        scorecard=sc,
        booster_info={
            "best_iteration": int(booster.best_iteration),
            "unconstrained_best_iteration": int(free.best_iteration),
            "validation": val_metrics,
        },
    )
