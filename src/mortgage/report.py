"""The dashboard (site/index.html), the model card (MODEL_CARD.md) and results.json."""

from __future__ import annotations

import html
import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs_version

from .config import Config
from .fairness import FOUR_FIFTHS
from .model import LABELS, MONOTONE, Result
from .prepare import NOT_PROVIDED
from .report_base import (
    AQUA,
    BLUE,
    CSS,
    GRID,
    LAYOUT_JS,
    MUTED,
    ORANGE,
    VIOLET,
    _card,
    _div,
    _int,
    _pct,
    _style,
    _table,
)

MAIN = "LightGBM, monotonic + calibrated"
EXTRA_CSS = """
.callout { border-left:4px solid #b7791f; background:#fbf6ea; padding:10px 14px; border-radius:6px; margin:10px 0 14px;
           color:#52514e; max-width:900px; }
.flag { color:#c4352b; font-weight:600; }
.mc h3 { margin-top:18px; }
.mc ul { color:#52514e; }
.cards { grid-template-columns:repeat(auto-fit,minmax(165px,1fr)); }
td.l:first-child { min-width:150px; }
.tight th, .tight td { padding:7px 6px; }
.ci { display:block; font-size:11.5px; color:#898781; }
main a { color:#2a78d6; }
"""


def _f(v, d=3) -> str:
    return "–" if v is None or pd.isna(v) else f"{v:.{d}f}"


def _ratio(v) -> str:
    return "–" if pd.isna(v) else f"{v:.2f}"


def _air(v) -> str:
    if pd.isna(v):
        return "–"
    return f"<span class='flag'>{v:.2f}</span>" if v < FOUR_FIFTHS else f"{v:.2f}"


# --------------------------------------------------------------------------- figures
def fig_years(df: pd.DataFrame, years: dict) -> go.Figure:
    t = df.groupby("year").agg(n=("denied", "size"), rate=("denied", "mean")).reset_index()
    role = {**{y: "train" for y in years["train"]}, years["valid"]: "validation", years["test"]: "test"}
    colors = [{"train": BLUE, "validation": AQUA, "test": ORANGE}[role[y]] for y in t["year"]]
    fig = go.Figure(
        go.Bar(
            x=t["year"].astype(str) + " (" + t["year"].map(role) + ")",
            y=t["rate"],
            marker_color=colors,
            text=[f"{r:.1%}" for r in t["rate"]],
            textposition="outside",
            customdata=t["n"],
            hovertemplate="%{x}<br>Denial rate %{y:.1%}<br>%{customdata:,} applications<extra></extra>",
        )
    )
    fig.update_traces(marker_line_width=0, width=0.55)
    _style(fig, "Denial rate by year", height=320, hovermode="closest", showlegend=False)
    fig.update_yaxes(tickformat=".0%", range=[0, max(t["rate"]) * 1.25])
    return fig


def fig_roc(res: Result) -> go.Figure:
    fig = go.Figure()
    auc = res.metrics.set_index("model")["roc_auc"]
    short = {
        MAIN: "LightGBM",
        "WoE logistic scorecard": "Scorecard",
        "Baseline: lender's past denial rate only": "Lender baseline",
    }
    for (name, c), color in zip(res.curves.items(), [BLUE, AQUA, MUTED], strict=False):
        fig.add_trace(
            go.Scatter(
                x=c["fpr"],
                y=c["tpr"],
                mode="lines",
                line=dict(color=color, width=2),
                name=f"{short.get(name, name)} (AUC {auc[name]:.3f})",
                hovertemplate="False-positive rate %{x:.2f}<br>True-positive rate %{y:.2f}",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color=GRID, dash="dot", width=1),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    _style(
        fig,
        f"ROC curves, held-out year {res.years['test']}",
        height=380,
        hovermode="closest",
        x_title="Approved applications flagged (false-positive rate)",
        y_title="Denied applications caught (true-positive rate)",
    )
    fig.update_layout(legend=dict(orientation="v", yanchor="bottom", y=0.02, xanchor="right", x=0.98))
    return fig


def fig_calibration(res: Result) -> go.Figure:
    fig = go.Figure()
    top = 0.0
    for (name, t), color in zip(res.calibration.items(), [BLUE, AQUA], strict=False):
        top = max(top, t["predicted"].max(), t["observed"].max())
        fig.add_trace(
            go.Scatter(
                x=t["predicted"],
                y=t["observed"],
                mode="lines+markers",
                name=name.split(",")[0] if "Light" not in name else "LightGBM (calibrated)",
                line=dict(color=color, width=2),
                marker=dict(size=8),
                customdata=t["n"],
                hovertemplate="Predicted %{x:.1%}<br>Observed %{y:.1%}<br>%{customdata:,} applications",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, top * 1.05],
            y=[0, top * 1.05],
            mode="lines",
            showlegend=False,
            hoverinfo="skip",
            line=dict(color=MUTED, dash="dot", width=1),
        )
    )
    _style(
        fig,
        "Calibration: predicted vs. observed denial rate (deciles)",
        height=380,
        hovermode="closest",
        x_title="Predicted denial probability",
        y_title="Observed denial rate",
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="left", x=0.02))
    return fig


def fig_shap(res: Result) -> go.Figure:
    imp = res.shap["importance"].head(12).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=imp["mean_abs_shap"],
            y=imp["label"],
            orientation="h",
            marker_color=BLUE,
            hovertemplate="%{y}<br>mean |SHAP| %{x:.3f} log-odds<extra></extra>",
        )
    )
    fig.update_traces(marker_line_width=0)
    _style(
        fig,
        "What moves the model: mean |SHAP value|",
        height=420,
        hovermode="closest",
        showlegend=False,
        x_title="Average impact on the log-odds of denial",
        margin=dict(l=210, r=20, t=74, b=48),
    )
    return fig


def fig_dependence(res: Result, feature: str, color: str) -> go.Figure:
    d = res.shap["dependence"][feature]
    fig = go.Figure(
        go.Scattergl(
            x=d["x"],
            y=d["shap"],
            mode="markers",
            marker=dict(size=5, color=color, opacity=0.35),
            hovertemplate=f"{LABELS[feature]} %{{x}}<br>SHAP %{{y:+.2f}}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    _style(
        fig,
        f"{LABELS[feature]}: effect on the denial log-odds",
        height=320,
        hovermode="closest",
        showlegend=False,
        x_title=LABELS[feature],
        y_title="SHAP value (log-odds)",
    )
    if feature in ("dti", "cltv"):
        fig.update_xaxes(range=[0, 100] if feature == "dti" else [20, 110])
    return fig


def fig_groups(table: pd.DataFrame, title: str) -> go.Figure:
    t = table.iloc[::-1]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t["denial_rate"],
            y=t["group"],
            mode="markers",
            name="Observed denial rate (95% CI)",
            marker=dict(color=BLUE, size=11),
            error_x=dict(
                type="data",
                symmetric=False,
                array=t["hi"] - t["denial_rate"],
                arrayminus=t["denial_rate"] - t["lo"],
                color=BLUE,
                thickness=1.5,
                width=0,
            ),
            customdata=t["n"],
            hovertemplate="%{y}<br>Observed %{x:.1%}<br>%{customdata:,} applications<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t["mean_p"],
            y=t["group"],
            mode="markers",
            name="Model's predicted denial rate",
            marker=dict(color=ORANGE, size=10, symbol="diamond"),
            hovertemplate="%{y}<br>Predicted %{x:.1%}<extra></extra>",
        )
    )
    _style(fig, title, height=110 + 48 * len(t), hovermode="closest", margin=dict(l=190, r=20, t=74, b=48))
    fig.update_xaxes(tickformat=".0%", rangemode="tozero", showgrid=True, gridcolor=GRID)
    return fig


def fig_odds(odds: pd.DataFrame, title: str) -> go.Figure:
    t = odds[odds["group"] != NOT_PROVIDED].iloc[::-1]
    fig = go.Figure()
    for col, name, color, shift in (
        ("unadjusted", "Unadjusted", ORANGE, 0.15),
        ("adjusted", "Adjusted for credit factors (model score)", BLUE, -0.15),
    ):
        y = np.arange(len(t)) + shift
        fig.add_trace(
            go.Scatter(
                x=t[f"odds_ratio_{col}"],
                y=y,
                mode="markers",
                name=name,
                marker=dict(color=color, size=10),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=t[f"hi_{col}"] - t[f"odds_ratio_{col}"],
                    arrayminus=t[f"odds_ratio_{col}"] - t[f"lo_{col}"],
                    color=color,
                    thickness=1.5,
                    width=0,
                ),
                customdata=np.c_[t["group"], t[f"lo_{col}"], t[f"hi_{col}"]],
                hovertemplate="%{customdata[0]}<br>Odds ratio %{x:.2f} (95% CI %{customdata[1]:.2f}–%{customdata[2]:.2f})"
                "<extra>" + name + "</extra>",
            )
        )
    fig.add_vline(x=1, line=dict(color=MUTED, width=1, dash="dot"))
    _style(
        fig,
        title,
        height=150 + 62 * len(t),
        hovermode="closest",
        margin=dict(l=190, r=20, t=100, b=48),
        x_title="Odds ratio of denial vs. reference group (log scale)",
    )
    fig.update_xaxes(type="log", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(tickvals=list(range(len(t))), ticktext=t["group"].tolist())
    return fig


# --------------------------------------------------------------------------- tables & text
def metrics_table(m: pd.DataFrame) -> str:
    rows = [
        [
            html.escape(r.model),
            _f(r.roc_auc),
            _f(r.gini),
            _f(r.pr_auc),
            _f(r.ks),
            _f(r.brier, 4),
            _f(r.log_loss),
            _pct(r.mean_predicted),
        ]
        for r in m.itertuples()
    ]
    return _table(["Model", "ROC AUC", "Gini", "PR AUC", "KS", "Brier", "Log loss", "Mean predicted"], rows)


def group_rows(t: pd.DataFrame) -> str:
    rows = []
    for r in t.itertuples():
        name = html.escape(r.group) + (" <span class='muted'>(reference)</span>" if r.is_reference else "")
        rows.append(
            [
                name,
                _int(r.n),
                f"{r.denial_rate:.1%}<span class='ci'>{r.lo:.1%}–{r.hi:.1%}</span>",
                _ratio(r.denial_ratio),
                _air(r.air_observed),
                _air(r.air_model),
                f"{r.fpr:.1%}",
                f"{r.fnr:.1%}",
                _ratio(r.calibration_ratio),
            ]
        )
    return (
        "<div class='tight'>"
        + _table(
            [
                "Group",
                "Applications",
                "Denial rate (95% CI)",
                "Ratio to ref.",
                "AIR, actual",
                "AIR, model",
                "Wrongly flagged",
                "Missed denials",
                "Pred. ÷ actual",
            ],
            rows,
        )
        + "</div>"
    )


def odds_rows(o: pd.DataFrame) -> str:
    if o.empty:
        return "<p class='muted'>The odds-ratio model could not be estimated for this run.</p>"
    rows = []
    for r in o.itertuples():
        rows.append(
            [
                html.escape(r.group),
                _int(r.n),
                f"{r.odds_ratio_unadjusted:.2f} <span class='muted'>({r.lo_unadjusted:.2f}–{r.hi_unadjusted:.2f})</span>",
                f"{r.odds_ratio_adjusted:.2f} <span class='muted'>({r.lo_adjusted:.2f}–{r.hi_adjusted:.2f})</span>",
                "<0.001" if r.p_value_adjusted < 0.001 else f"{r.p_value_adjusted:.3f}",
            ]
        )
    return _table(
        [
            "Group",
            "Applications",
            "Unadjusted odds ratio (95% CI)",
            "Adjusted odds ratio (95% CI)",
            "p (adjusted)",
        ],
        rows,
    )


def summary_rows(audit: dict) -> str:
    rows = []
    for a in audit.values():
        s = a["summary"]
        rows.append(
            [
                a["title"],
                str(s["groups"]),
                f"{s['observed_denial_rate_gap'] * 100:.1f} pts",
                f"{s['demographic_parity_difference'] * 100:.1f} pts",
                f"{s['demographic_parity_ratio']:.2f}",
                f"{s['equalized_odds_difference'] * 100:.1f} pts",
            ]
        )
    return _table(
        [
            "Characteristic",
            "Groups",
            "Actual denial-rate gap",
            "Model flag-rate gap (DP difference)",
            "DP ratio",
            "Equalized-odds difference",
        ],
        rows,
    )


def headline_group(audit: dict) -> dict | None:
    """The race/ethnicity group with the highest observed denial ratio vs. the reference group."""
    a = audit["race_ethnicity"]
    t = a["groups"][(~a["groups"]["is_reference"]) & (a["groups"]["group"] != NOT_PROVIDED)]
    if t.empty:
        return None
    g = t.sort_values("denial_ratio", ascending=False).iloc[0]
    out = {
        "group": g["group"],
        "reference": a["reference"],
        "denial_rate": float(g["denial_rate"]),
        "reference_rate": float(a["groups"].set_index("group").loc[a["reference"], "denial_rate"]),
        "denial_ratio": float(g["denial_ratio"]),
        "air_observed": float(g["air_observed"]),
        "n": int(g["n"]),
    }
    o = a["odds"]
    if not o.empty and g["group"] in set(o["group"]):
        r = o.set_index("group").loc[g["group"]]
        out.update(
            {
                k: float(r[k])
                for k in (
                    "odds_ratio_unadjusted",
                    "lo_unadjusted",
                    "hi_unadjusted",
                    "odds_ratio_adjusted",
                    "lo_adjusted",
                    "hi_adjusted",
                    "p_value_adjusted",
                )
            }
        )
    return out


def odds_sentence(h: dict | None) -> str:
    if not h or "odds_ratio_adjusted" not in h:
        return ""
    raw, adj = h["odds_ratio_unadjusted"], h["odds_ratio_adjusted"]
    sig = h["lo_adjusted"] > 1 or h["hi_adjusted"] < 1
    s = (
        f"For {h['group']} applicants the odds of denial were {raw:.2f}× those of {h['reference']} applicants before "
        f"adjustment and {adj:.2f}× (95% CI {h['lo_adjusted']:.2f}–{h['hi_adjusted']:.2f}) after adjusting for "
        f"the credit factors in the model's score. "
    )
    if raw > 1 and adj < raw:
        s += (
            f"The credit factors HMDA records account for about {(1 - np.log(adj) / np.log(raw)):.0%} of the gap "
            "on the log-odds scale; "
        )
        s += (
            "the rest is statistically significant and would be a reason for a closer review with data HMDA "
            "does not contain (credit scores and histories)."
            if sig
            else "the rest is not statistically distinguishable from zero."
        )
    return s


def md_to_html(md: str) -> str:
    """Just enough Markdown for the model card: headings, bullets, tables, quotes, bold and italics."""
    import re

    def inline(t: str) -> str:
        t = html.escape(t)
        t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
        return re.sub(r"(?<![\w])_(.+?)_(?![\w])", r"<i>\1</i>", t)

    out, items, table = [], [], []

    def flush() -> None:
        if items:
            out.append("<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>")
            items.clear()
        if table:
            head, body = table[0], [r for r in table[2:]]
            out.append(_table(head, body))
            table.clear()

    for line in md.splitlines():
        if line.startswith("- "):
            items.append(inline(line[2:]))
            continue
        if line.startswith("|"):
            table.append([inline(c.strip()) for c in line.strip("|").split("|")])
            continue
        flush()
        if line.startswith("## "):
            out.append(f"<h3>{inline(line[3:])}</h3>")
        elif line.startswith("# "):
            continue  # the section already has a title
        elif line.startswith("> "):
            out.append(f"<div class='callout'>{inline(line[2:])}</div>")
        elif line.strip():
            out.append(f"<p class='muted'>{inline(line)}</p>")
    flush()
    return "".join(out)


def model_card(cfg: Config, res: Result, audit: dict, n_total: int, built_at: str, synthetic: bool) -> str:
    m = res.metrics.set_index("model")
    y = res.years
    imp = ", ".join(res.shap["importance"]["label"].head(5))
    up = ", ".join(LABELS[k].lower() for k, v in MONOTONE.items() if v > 0)
    down = ", ".join(LABELS[k].lower() for k, v in MONOTONE.items() if v < 0)
    mono = f"predicted risk can never fall as {up} rise, and never rise as {down} rise"
    h = headline_group(audit)
    lines = [
        "# Model card: mortgage denial model (HMDA, " + cfg.state_name + ")",
        "",
        "> **SYNTHETIC TEST DATA: these numbers are not real.**"
        if synthetic
        else f"_Generated automatically on {built_at} by the pipeline in this repository._",
        "",
        "## Intended use",
        "- **What it is:** a model of how lenders decided home-purchase mortgage applications, trained on public HMDA "
        "records, plus a fair-lending audit of those decisions and of the model.",
        "- **Intended users:** analysts studying lending outcomes; a portfolio demonstration of model development, "
        "validation and fairness testing.",
        "- **Not for:** approving or denying real applications, pricing, or judging any individual or lender. HMDA "
        "lacks credit scores and credit history, which drive real underwriting decisions.",
        "",
        "## Data",
        f"- Source: CFPB HMDA Data Browser API (public loan-level records), state of {cfg.state_name}.",
        f"- Sample: {n_total:,} home-purchase, first-lien, principal-residence applications for 1-4 unit homes that "
        "were approved or denied (withdrawn and incomplete applications excluded), restricted to complete files: "
        "income, debt-to-income, combined loan-to-value and property value all reported.",
        f"- Out-of-time split: train {', '.join(map(str, y['train']))} ({y['n_train']:,}) · validation {y['valid']} "
        f"({y['n_valid']:,}, early stopping, calibration, decision threshold) · test {y['test']} ({y['n_test']:,}).",
        "- Target: denied (action taken = 3) vs. approved (originated, or approved but not accepted).",
        "",
        "## Features",
        "- Used: loan amount, property value, combined loan-to-value, debt-to-income, income, loan-to-income, income "
        "relative to the area median, loan term, units, loan type, preapproval, conforming-limit status, dwelling "
        "type, application channel, automated underwriting system, and "
        "the lender's smoothed historical denial rate and size (from training years only, out-of-fold).",
        "- Excluded on purpose: race, ethnicity, sex and age (used only for the audit); co-applicant presence (can "
        "stand in for marital status); census-tract demographics and geography (can act as proxies for race); and "
        "anything recorded after the decision (interest rate, rate spread, fees, denial reasons, purchaser). "
        "'Initially payable to the institution' is also excluded: HMDA reports it as not applicable for most denied "
        "applications, so it leaks the outcome.",
        f"- Monotonic constraints (LightGBM): {mono}.",
        "",
        "## Performance (held-out year " + str(y["test"]) + ")",
        "| Model | ROC AUC | PR AUC | KS | Brier |",
        "|---|---|---|---|---|",
        *[
            f"| {k} | {r.roc_auc:.3f} | {r.pr_auc:.3f} | {r.ks:.3f} | {r.brier:.4f} |"
            for k, r in m.iterrows()
        ],
        "",
        f"- Observed denial rate in {y['test']}: {m.loc[MAIN, 'observed_rate']:.1%}; mean calibrated prediction: "
        f"{m.loc[MAIN, 'mean_predicted']:.1%}.",
        f"- Most influential features (mean |SHAP|): {imp}.",
        f"- Leakage check: the strongest single feature on its own reaches ROC AUC "
        f"{res.single_feature['auc'].iloc[0]:.3f} ({res.single_feature['label'].iloc[0]}).",
        f"- Decision threshold for the audit: flag applications with predicted probability ≥ {res.threshold:.3f} "
        f"(the share flagged in {y['valid']} equals that year's actual denial rate).",
        "",
        "## Fairness evaluation (held-out year)",
        "| Characteristic | Actual denial-rate gap | Model flag-rate gap | Equalized-odds difference |",
        "|---|---|---|---|",
        *[
            f"| {a['title']} | {a['summary']['observed_denial_rate_gap'] * 100:.1f} pts | "
            f"{a['summary']['demographic_parity_difference'] * 100:.1f} pts | "
            f"{a['summary']['equalized_odds_difference'] * 100:.1f} pts |"
            for a in audit.values()
        ],
        "",
    ]
    if h:
        lines.append(
            f"- Largest race/ethnicity gap: {h['group']} applicants were denied {h['denial_rate']:.1%} of the "
            f"time vs. {h['reference_rate']:.1%} for {h['reference']} applicants ({h['denial_ratio']:.2f}×; "
            f"approval adverse impact ratio {h['air_observed']:.2f})."
        )
        if odds_sentence(h):
            lines.append("- " + odds_sentence(h))
    lines += [
        "",
        "## Limitations and ethical considerations",
        "- HMDA has no credit scores, credit histories, assets or employment data, so differences that remain after "
        "adjustment cannot be attributed to discrimination; they mark where a closer review would start.",
        "- The model learns past lending decisions, including any bias in them. It must not be used to make decisions.",
        "- HMDA reports debt-to-income, loan-to-value, property value and income as not applicable when the lender "
        "made no credit decision on them or did not rely on them (often files denied early). Missing values would leak "
        "the outcome, so only complete files are modeled; the excluded files are mostly denials, so the audit "
        "describes decisions on complete files (the dashboard shows rates on all decided files too).",
        "- Lender identity is used as a feature; lenders that serve different communities may differ in denial rates "
        "for reasons that are themselves worth auditing.",
        "- Race and ethnicity are as reported in HMDA: usually self-reported, but for some in-person applications where "
        "the applicant declined, recorded by the lender from visual observation or surname. Applicants with no "
        "information form a separate 'Not provided' group.",
        "- Adjusting the model with protected characteristics to equalize outcomes would itself be disparate "
        "treatment under ECOA, so no such mitigation is applied here.",
    ]
    return "\n".join(lines) + "\n"


def build_site(
    cfg: Config,
    df: pd.DataFrame,
    flow: pd.DataFrame,
    res: Result,
    audit: dict,
    synthetic: bool = False,
    df_all: pd.DataFrame | None = None,
) -> dict:
    df_all = df if df_all is None else df_all
    site = cfg.site_dir
    site.mkdir(parents=True, exist_ok=True)
    built_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    y = res.years
    m = res.metrics.set_index("model")
    test_rate = m.loc[MAIN, "observed_rate"]
    h = headline_group(audit)
    card_md = model_card(cfg, res, audit, len(df), built_at, synthetic)
    (site / "model_card.md").write_text(card_md, encoding="utf-8")

    banner = (
        (
            "<div class='synthetic'>SYNTHETIC TEST DATA: generated for automated tests; these numbers are not "
            "real</div>"
        )
        if synthetic
        else ""
    )
    cards = "".join(
        [
            _card(
                "Applications modeled",
                _int(len(df)),
                f"{cfg.state_name}, {min(df['year'])}–{max(df['year'])}",
            ),
            _card(
                f"Denial rate, {y['test']}",
                f"{test_rate:.1%}",
                f"{_int(y['n_test'])} applications in the held-out year",
            ),
            _card("ROC AUC, held-out year", _f(m.loc[MAIN, "roc_auc"]), "monotonic LightGBM, calibrated"),
            _card(
                "Scorecard ROC AUC", _f(m.loc["WoE logistic scorecard", "roc_auc"]), "WoE logistic regression"
            ),
            _card(
                f"{h['group']} vs. {h['reference']}" if h else "Largest gap",
                f"{h['denial_ratio']:.2f}×" if h else "–",
                f"denial rate {h['denial_rate']:.1%} vs. {h['reference_rate']:.1%}" if h else "",
            ),
            _card(
                "Adjusted odds ratio",
                f"{h['odds_ratio_adjusted']:.2f}" if h and "odds_ratio_adjusted" in h else "–",
                f"95% CI {h['lo_adjusted']:.2f}–{h['hi_adjusted']:.2f}, after credit factors"
                if h and "odds_ratio_adjusted" in h
                else "",
            ),
        ]
    )
    flow_rows = [
        [
            html.escape(r.step),
            _int(r.applications),
            f"{r.denied / r.applications:.1%}" if r.applications else "–",
        ]
        for r in flow.itertuples()
    ]
    inc = df_all[~df_all["complete_file"]]
    miss_note = (
        f"{_int(len(inc))} decided applications ({len(inc) / len(df_all):.1%}) are not complete files; "
        f"{inc['denied'].mean() if len(inc) else 0:.1%} of them were denied. HMDA reports those fields as not "
        "applicable when the lender made no credit decision on them or did not rely on them, so their absence "
        "would leak the outcome into the model."
    )
    t_all = df_all[df_all["year"] == y["test"]]
    ctx = (
        t_all.groupby("race_ethnicity")
        .agg(
            n_all=("denied", "size"),
            rate_all=("denied", "mean"),
            n_complete=("complete_file", "sum"),
        )
        .join(
            t_all[t_all["complete_file"]].groupby("race_ethnicity")["denied"].mean().rename("rate_complete")
        )
        .reindex(audit["race_ethnicity"]["groups"]["group"])
        .dropna(subset=["n_all"])
    )
    ctx_rows = [
        [html.escape(g), _int(r.n_all), f"{r.rate_all:.1%}", _int(r.n_complete), f"{r.rate_complete:.1%}"]
        for g, r in ctx.iterrows()
    ]
    context_html = _table(["Group", "All decided", "Denied", "Complete files", "Denied"], ctx_rows)

    race = audit["race_ethnicity"]
    other_attrs = [a for k, a in audit.items() if k != "race_ethnicity"]
    other_html = "".join(
        f"<h3>{a['title']} (reference: {html.escape(a['reference'])})</h3>{group_rows(a['groups'])}"
        f"<details><summary>Odds ratios</summary>{odds_rows(a['odds'])}</details>"
        for a in other_attrs
    )

    iv = res.iv
    iv_rows = [
        [html.escape(r.label), _f(r.iv), str(r.bins), "✓" if r.in_scorecard else ""] for r in iv.itertuples()
    ]
    pts = res.scorecard_points
    pts_rows = [
        [html.escape(r.label), html.escape(str(r.bin)), _f(r.woe, 2), f"{r.points:.0f}"]
        for r in pts.itertuples()
    ]
    card_url = f"{cfg.repo_url}/blob/main/MODEL_CARD.md"
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mortgage Denial Model &amp; Fair-Lending Audit</title>
<meta name="description" content="Predicting mortgage denials from public HMDA data with a calibrated LightGBM model and a credit scorecard, then auditing decisions and the model for disparities by race, ethnicity, sex and age.">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🏠</text></svg>">
<script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js" charset="utf-8"></script>
<style>{CSS}{EXTRA_CSS}</style></head>
<body>{banner}
<header><div class="wrap">
  <h1>Mortgage denial modeling &amp; fair-lending audit</h1>
  <p>Which home-purchase mortgage applications get denied, and are outcomes different across race, ethnicity, sex and
  age once credit factors are taken into account? Built from {_int(len(df))} public HMDA applications in
  {cfg.state_name} ({min(df["year"])}–{max(df["year"])}), with the newest year held out for testing.</p>
  <a class="btn" href="scorer.html">Try the scorecard on an application</a><a class="btn" href="#model-card">Model card</a>
  <a class="btn" href="{cfg.repo_url}">Code on GitHub</a>
</div></header>
<main>
<div class="cards">{cards}</div>

<section><h2>The data</h2>
<p>Most U.S. mortgage lenders must report every application they receive under the Home Mortgage Disclosure Act.
This project uses {cfg.state_name}'s home-purchase applications that ended in a decision. Models train on
{", ".join(map(str, y["train"]))}, tune on {y["valid"]} and are tested on {y["test"]}, a year they never saw.</p>
<div class="grid2"><div><h3>Sample</h3>{_table(["Step", "Applications", "Share denied"], flow_rows)}
<p class="muted">{miss_note}</p></div><div>{_div(fig_years(df, y))}
<h3>Denial rates in {y["test"]}: all decided applications vs. complete files</h3>{context_html}</div></div></section>

<section><h2>How well can denials be predicted?</h2>
<p>Four models on the held-out year: a monotonic LightGBM with isotonic calibration (the main model), the same model
without monotonic constraints (what the constraints cost in accuracy), a weight-of-evidence logistic scorecard (the
classic, fully explainable credit model), and a baseline that only knows the lender's past denial rate.</p>
{metrics_table(res.metrics)}
<details><summary>Leakage check: each feature on its own</summary>
<p class="muted">ROC AUC of each feature used alone on the held-out year (categories scored by their training-year
denial rate). A single application-time field that nearly separates denials from approvals would point to a
reporting artifact rather than a credit signal.</p>
{_table(["Feature", "ROC AUC alone"], [[html.escape(r.label), _f(r.auc)] for r in res.single_feature.itertuples()])}
</details>
<div class="grid2">{_div(fig_roc(res))}{_div(fig_calibration(res))}</div>
<p class="muted">Calibration was fitted on {y["valid"]}; if the market's denial rate shifts between years, predictions
shift with it (mean prediction {_pct(m.loc[MAIN, "mean_predicted"])} vs. actual {_pct(test_rate)} in {y["test"]}).</p>
</section>

<section><h2>What drives the predictions?</h2>
<p>SHAP values split each prediction into each feature's contribution (on the log-odds scale) for
{_int(res.shap["n"])} sampled test applications. Debt-to-income, loan-to-value and income are constrained so a worse
value can never lower the predicted risk.</p>
<div class="grid2">{_div(fig_shap(res))}<div>{_div(fig_dependence(res, "dti", BLUE))}
{_div(fig_dependence(res, "cltv", VIOLET))}</div></div></section>

<section><h2>The scorecard</h2>
<p>The scorecard bins each feature (merging bins until the denial rate moves in one direction), converts the bins
to weight of evidence, keeps features with information value ≥ 0.02 and correct coefficient signs, and scales the
logistic regression to points: 600 points = 20:1 odds of approval, +20 points doubles the odds.
<b><a href="scorer.html">Try it on an application</a></b>: enter loan, income and lender details and see the score,
the points for each factor and how applications with similar scores were decided in {y["test"]}.</p>
<div class="grid2"><div><h3>Information value</h3>{_table(["Feature", "IV", "Bins", "In scorecard"], iv_rows)}</div>
<div><h3>Points</h3><details><summary>Show all {len(pts_rows)} bins</summary>
{_table(["Feature", "Bin", "WoE", "Points"], pts_rows, left_cols=2)}</details></div></div></section>

<section id="fair-lending"><h2>Fair-lending audit: race and ethnicity</h2>
<p>Observed denial rates in {y["test"]} and the model's predicted rate per group. Approval <b>adverse impact
ratios</b> (AIR) below {FOUR_FIFTHS} (the "four-fifths" screen) are shown in red. The model flags the riskiest
applications at the threshold that matches lenders' overall denial rate in {y["valid"]}; error rates show who it
would wrongly flag (approved applicants) and whose denials it would miss.</p>
{_div(fig_groups(race["groups"], f"Denial rate by race and ethnicity, {y['test']}"))}
{group_rows(race["groups"])}
<h3>Do the gaps remain after accounting for credit factors?</h3>
<p>Logistic regressions of denial on group membership, before and after adding the model's risk score (all the
credit information HMDA records), with standard errors clustered by lender. {odds_sentence(h)}</p>
{_div(fig_odds(race["odds"], "Odds ratio of denial vs. " + race["reference"])) if not race["odds"].empty else ""}
{odds_rows(race["odds"])}
<div class="callout"><b>How to read this:</b> HMDA has no credit scores or credit histories, which are central to
underwriting. A gap that remains after adjustment is therefore a signal for review, not proof of discrimination. And
because the model learns from past decisions, it can reproduce any bias in them.</div>
</section>

<section><h2>Fair-lending audit: sex and age</h2>
<p>The same checks for sex and age. Fairlearn summaries across all characteristics are at the bottom.</p>
{other_html}
<h3>Summary (Fairlearn)</h3>
{summary_rows(audit)}
<p class="muted">Gaps are between the groups with the highest and lowest rates, excluding applicants who did not
provide the information. Equalized-odds difference = the larger of the gaps in the model's false-positive and
false-negative rates.</p>
</section>

<section class="mc" id="model-card"><h2>Model card</h2>
<p>Also kept in the repository as <a href="{card_url}">MODEL_CARD.md</a>, regenerated on every run.</p>
{md_to_html(card_md)}</section>

<section><h2>How it works</h2><ul class="method">
<li><b>Data:</b> the CFPB HMDA Data Browser API (no key), one CSV per year; the newest {cfg.n_years} published years
are found automatically. Column names are normalized and checked before anything else runs.</li>
<li><b>Preparation (DuckDB SQL):</b> typed and filtered application table, HMDA's debt-to-income bands converted to
numbers, audit groups derived from HMDA's race, ethnicity, sex and age fields.</li>
<li><b>Models:</b> LightGBM with monotonic constraints and early stopping, isotonic calibration on the validation
year; a WoE logistic scorecard with monotonic binning and IV-based selection; lender history encoded out-of-fold.</li>
<li><b>Explanations:</b> TreeSHAP (shap). <b>Fairness:</b> Fairlearn MetricFrame, Wilson intervals, adverse impact
ratios, statsmodels logistic regression with lender-clustered errors.</li>
<li><b>Automation:</b> GitHub Actions runs the tests, rebuilds everything monthly and publishes this page.</li>
</ul></section>
</main>
<footer>Data: Consumer Financial Protection Bureau, HMDA Data Browser (public data) · Built {built_at} ·
<a href="{cfg.repo_url}">Source code</a></footer>
<script>{LAYOUT_JS}</script>
</body></html>"""
    (site / "index.html").write_text(page, encoding="utf-8")
    (site / ".nojekyll").touch()

    summary = {
        "state": cfg.state,
        "state_name": cfg.state_name,
        "years": y,
        "n_applications": int(len(df)),
        "denial_rate_test": float(test_rate),
        "threshold": res.threshold,
        "metrics": res.metrics.to_dict(orient="records"),
        "top_features": res.shap["importance"]["label"].head(5).tolist(),
        "headline": h,
        "fairness": {
            k: {
                "summary": a["summary"],
                "groups": a["groups"].drop(columns=["is_reference"]).to_dict(orient="records"),
                "odds": a["odds"].to_dict(orient="records"),
            }
            for k, a in audit.items()
        },
        "booster": res.booster_info,
        "synthetic": synthetic,
        "built_at": built_at,
    }
    (site / "results.json").write_text(json.dumps(summary, indent=2, default=float))
    return {**summary, "model_card": card_md}
