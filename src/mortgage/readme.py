"""Rewrites the results block in README.md after each run."""

from __future__ import annotations

import re
from pathlib import Path

MAIN = "LightGBM, monotonic + calibrated"
CARD = "WoE logistic scorecard"


def results_block(s: dict) -> str:
    y = s["years"]
    m = {r["model"]: r for r in s["metrics"]}
    lines = [
        f"_HMDA {s['state_name']} home-purchase applications, {min(y['train'])}–{y['test']} · "
        f"{s['n_applications']:,} applications · trained on {', '.join(map(str, y['train']))}, "
        f"validated on {y['valid']}, tested on {y['test']}_",
        "",
        "| Model (held-out year) | ROC AUC | PR AUC | KS | Brier |",
        "|---|---|---|---|---|",
        *[
            f"| {k} | {r['roc_auc']:.3f} | {r['pr_auc']:.3f} | {r['ks']:.3f} | {r['brier']:.4f} |"
            for k, r in m.items()
        ],
        "",
        f"**Denial rate in {y['test']}:** {s['denial_rate_test']:.1%} · **top features (SHAP):** "
        f"{', '.join(s['top_features'])}",
    ]
    h = s.get("headline")
    if h:
        lines += [
            "",
            f"**Largest race/ethnicity gap:** {h['group']} applicants were denied {h['denial_rate']:.1%} of "
            f"the time vs. {h['reference_rate']:.1%} for {h['reference']} applicants ({h['denial_ratio']:.2f}×).",
        ]
        if "odds_ratio_adjusted" in h:
            lines.append(
                f"Odds ratio of denial {h['odds_ratio_unadjusted']:.2f} unadjusted → "
                f"{h['odds_ratio_adjusted']:.2f} (95% CI {h['lo_adjusted']:.2f}–{h['hi_adjusted']:.2f}) after "
                "adjusting for the credit factors in the model's score. HMDA has no credit scores, so this is "
                "a flag for review, not proof of discrimination."
            )
    f = s["fairness"]
    lines += [
        "",
        "| Fairness (held-out year) | Actual denial-rate gap | Model flag-rate gap | Equalized-odds diff. |",
        "|---|---|---|---|",
    ]
    names = {"race_ethnicity": "Race and ethnicity", "sex": "Sex", "age_group": "Age"}
    for k, v in f.items():
        sm = v["summary"]
        lines.append(
            f"| {names.get(k, k)} | {sm['observed_denial_rate_gap'] * 100:.1f} pts | "
            f"{sm['demographic_parity_difference'] * 100:.1f} pts | "
            f"{sm['equalized_odds_difference'] * 100:.1f} pts |"
        )
    lines += ["", f"_Last run: {s['built_at']}_"]
    return "\n".join(lines)


def update_readme(readme: Path, summary: dict) -> None:
    text = readme.read_text()
    pattern = r"(<!-- RESULTS:START -->\n).*?(\n<!-- RESULTS:END -->)"
    block = results_block(summary)
    readme.write_text(re.sub(pattern, lambda m: m.group(1) + block + m.group(2), text, flags=re.S))
