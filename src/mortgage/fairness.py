"""Fair-lending audit of the observed decisions and of the model, on the held-out year.

For each protected characteristic (race/ethnicity, sex, age) it reports
* observed denial rates with 95% Wilson intervals and the adverse impact ratio of approval rates (the "four-fifths"
  screen used in disparate-impact analysis);
* the model's flag rates and error rates per group with Fairlearn (demographic parity and equalized odds);
* calibration per group: does the model over- or under-predict denials for any group?
* unadjusted vs. adjusted odds ratios of denial (logistic regression with lender-clustered standard errors), where the
  adjustment is the model's risk score built from the credit factors HMDA records.
HMDA has no credit scores or credit histories, so a gap that remains after adjustment is a flag for review, not proof
of discrimination.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from fairlearn.metrics import (
    MetricFrame,
    count,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
)
from sklearn.metrics import roc_auc_score

from .prepare import NOT_PROVIDED, REFERENCE

log = logging.getLogger(__name__)
TITLES = {"race_ethnicity": "Race and ethnicity", "sex": "Sex", "age_group": "Age"}
FOUR_FIFTHS = 0.8


def wilson(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return centre - half, centre + half


def group_table(d: pd.DataFrame, attr: str, min_n: int) -> pd.DataFrame:
    ref = REFERENCE[attr]
    g = (
        d.groupby(attr)
        .agg(
            n=("denied", "size"),
            denied=("denied", "sum"),
            mean_p=("p_denial", "mean"),
            flagged=("pred_denied", "mean"),
        )
        .reset_index()
        .rename(columns={attr: "group"})
    )
    g = g[g["n"] >= min_n].copy()
    g["denial_rate"] = g["denied"] / g["n"]
    g["lo"], g["hi"] = wilson(g["denied"], g["n"])
    g["approval_rate"] = 1 - g["denial_rate"]
    r = g.set_index("group")
    if ref not in r.index:
        raise ValueError(f"Reference group {ref!r} has fewer than {min_n} applications")
    g["denial_ratio"] = g["denial_rate"] / r.loc[ref, "denial_rate"]
    g["air_observed"] = g["approval_rate"] / r.loc[ref, "approval_rate"]
    g["air_model"] = (1 - g["flagged"]) / (1 - r.loc[ref, "flagged"])
    g["calibration_ratio"] = g["mean_p"] / g["denial_rate"].where(g["denial_rate"] > 0)

    sub = d[d[attr].isin(g["group"])]
    mf = MetricFrame(
        metrics={"fpr": false_positive_rate, "fnr": false_negative_rate},
        y_true=sub["denied"],
        y_pred=sub["pred_denied"],
        sensitive_features=sub[attr],
    )
    g = g.merge(mf.by_group.reset_index().rename(columns={attr: "group"}), on="group", how="left")
    aucs = {
        k: roc_auc_score(v["denied"], v["p_denial"]) if v["denied"].nunique() == 2 else np.nan
        for k, v in sub.groupby(attr)
    }
    g["auc"] = g["group"].map(aucs)
    g["is_reference"] = g["group"] == ref
    order = (
        [ref]
        + sorted(
            [x for x in g["group"] if x not in (ref, NOT_PROVIDED)],
            key=lambda x: -g.set_index("group").loc[x, "n"],
        )
        + [x for x in g["group"] if x == NOT_PROVIDED]
    )
    return g.set_index("group").loc[order].reset_index()


def fairlearn_summary(d: pd.DataFrame, attr: str, groups: list[str]) -> dict:
    """Gaps between the most and least favoured groups (applicants who did not provide the information excluded)."""
    sub = d[d[attr].isin([g for g in groups if g != NOT_PROVIDED])]
    y, yp, s = sub["denied"], sub["pred_denied"], sub[attr]
    mf = MetricFrame(
        metrics={"n": count, "flag_rate": selection_rate}, y_true=y, y_pred=yp, sensitive_features=s
    )
    return {
        "groups": len(mf.by_group),
        "demographic_parity_difference": float(demographic_parity_difference(y, yp, sensitive_features=s)),
        "demographic_parity_ratio": float(demographic_parity_ratio(y, yp, sensitive_features=s)),
        "equalized_odds_difference": float(equalized_odds_difference(y, yp, sensitive_features=s)),
        "observed_denial_rate_gap": float(
            sub.groupby(attr)["denied"].mean().max() - sub.groupby(attr)["denied"].mean().min()
        ),
    }


def odds_ratios(d: pd.DataFrame, attr: str, groups: list[str]) -> pd.DataFrame:
    """Odds ratio of denial vs. the reference group, before and after adjusting for the model's credit-risk score."""
    ref = REFERENCE[attr]
    sub = d[d[attr].isin(groups)].reset_index(drop=True)
    dummies = pd.get_dummies(sub[attr]).drop(columns=ref).astype(float)
    clusters = pd.factorize(sub["lei"])[0]
    out = []
    for label, x in (("unadjusted", dummies), ("adjusted", dummies.assign(model_logit=sub["model_logit"]))):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.Logit(sub["denied"].astype(float), sm.add_constant(x)).fit(
                disp=0, maxiter=200, cov_type="cluster", cov_kwds={"groups": clusters}
            )
        ci = fit.conf_int()
        for gname in dummies.columns:
            out.append(
                {
                    "group": gname,
                    "model": label,
                    "odds_ratio": float(np.exp(fit.params[gname])),
                    "lo": float(np.exp(ci.loc[gname, 0])),
                    "hi": float(np.exp(ci.loc[gname, 1])),
                    "p_value": float(fit.pvalues[gname]),
                }
            )
    t = pd.DataFrame(out).pivot(index="group", columns="model")
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    t = t.reset_index()
    raw, adj = np.log(t["odds_ratio_unadjusted"]), np.log(t["odds_ratio_adjusted"])
    t["share_of_gap_explained"] = np.where(raw > 0.05, 1 - adj / raw, np.nan)
    n = sub[attr].value_counts()
    t["n"] = t["group"].map(n)
    return t.sort_values("n", ascending=False).reset_index(drop=True)


def audit(test: pd.DataFrame, min_n: int = 300) -> dict:
    """Run the audit for every protected characteristic on the held-out year."""
    out = {}
    for attr in REFERENCE:
        table = group_table(test, attr, min_n)
        groups = table["group"].tolist()
        try:
            odds = odds_ratios(test, attr, groups)
        except Exception as exc:  # e.g. too few denials in a small group
            log.warning("Odds-ratio model for %s failed: %s", attr, exc)
            odds = pd.DataFrame()
        out[attr] = {
            "title": TITLES[attr],
            "reference": REFERENCE[attr],
            "groups": table,
            "summary": fairlearn_summary(test, attr, groups),
            "odds": odds,
        }
    return out
