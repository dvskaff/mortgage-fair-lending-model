"""Try the scorecard: a page (site/scorer.html) where anyone can enter a mortgage application and see how the
weight-of-evidence scorecard scores it, factor by factor.

The fitted scorecard (bin edges, points per bin, scaling) is exported to data/scorecard.json; the page reproduces
the score in JavaScript and compares it with how applications with similar scores were actually decided in the
held-out year. Race, ethnicity, sex and age are not inputs: the scorecard never sees them.
"""

from __future__ import annotations

import html
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .model import LABELS, Result, Scorecard

log = logging.getLogger(__name__)

# Labels in the page's own units (the dashboard's labels name the model's units, e.g. "$K" or "log applications")
PAGE_LABELS = {"loan_amount_k": "Loan amount", "property_value_k": "Property value", "income_k": "Income",
               "lender_log_volume": "Lender size"}  # fmt: skip
CONFORMING = {"C": "Conforming", "NC": "Nonconforming", "U": "Undetermined", "NA": "Not applicable"}
# Raw inputs on the page and the scorecard features each one feeds (an input is shown only if one is used).
INPUTS = {
    "loan_amount": ["loan_amount_k", "cltv", "loan_to_income"],
    "property_value": ["property_value_k", "cltv"],
    "income": ["income_k", "loan_to_income", "income_to_area_median"],
    "area_median": ["income_to_area_median"],
    "dti": ["dti"],
    "loan_term": ["loan_term"],
    "total_units": ["total_units"],
    "lender_rate": ["lender_denial_rate"],
    "lender_apps": ["lender_log_volume"],
}


def _bins(sc: Scorecard, f: str) -> list[dict]:
    pts = sc.points_table().query("feature == @f").set_index("bin")["points"]
    b = sc.binnings[f]
    out = []
    if b.kind == "numeric":
        for i in range(len(b.edges) + 1):
            label = b._interval(i)
            if label in pts:
                out.append({"label": label, "points": round(float(pts[label]), 2),
                            "lo": None if i == 0 else b.edges[i - 1], "hi": None if i == len(b.edges) else b.edges[i]})  # fmt: skip
        if "Missing" in pts:
            out.append({"label": "Missing", "points": round(float(pts["Missing"]), 2), "missing": True})
    else:
        out = [{"label": k, "points": round(float(pts[k]), 2)} for k in b.ordered_bins()]
    return out


def _fallback_points(sc: Scorecard) -> float:
    """Points for a bin the scorecard never saw (weight of evidence 0): the neutral share of the score."""
    factor = sc.pdo / np.log(2)
    offset = sc.base_score - factor * np.log(sc.base_odds)
    n = len(sc.binnings)
    return float(-(sc.model.intercept_[0] / n) * factor + offset / n)


def export(res: Result, years: dict) -> dict:
    sc = res.scorecard
    test = res.test
    factor = sc.pdo / np.log(2)
    offset = sc.base_score - factor * np.log(sc.base_odds)
    pts = sc.points_table()
    # each test application's points per factor, to describe the "typical" applicant
    typical = {}
    for f, b in sc.binnings.items():
        p = pts.query("feature == @f").set_index("bin")["points"]
        typical[f] = float(b.labels(test[f]).map(p).fillna(_fallback_points(sc)).mean())
    feats = [{"feature": f, "label": PAGE_LABELS.get(f, LABELS.get(f, f)), "kind": b.kind, "edges": b.edges, "bins": _bins(sc, f),
              "max_points": round(float(pts.query("feature == @f")["points"].max()), 2),
              "typical_points": round(typical[f], 2)} for f, b in sc.binnings.items()]  # fmt: skip

    area = (
        (test["income_k"] * 1000 / test["income_to_area_median"]).replace([np.inf, -np.inf], np.nan).dropna()
    )
    approved, denied = test[test["denied"] == 0], test[test["denied"] == 1]

    def profile(d: pd.DataFrame) -> dict:
        med = lambda c: float(d[c].median())  # noqa: E731
        mode = lambda c: str(d[c].astype(str).mode().iloc[0])  # noqa: E731
        return {
            "loan_amount": round(med("loan_amount_k") * 1000, -3), "property_value": round(med("property_value_k") * 1000, -3),
            "income": round(med("income_k") * 1000, -3), "area_median": round(float(area.median()), -2),
            "dti": round(med("dti")), "loan_term": round(med("loan_term")), "total_units": 1,
            "lender_rate": round(med("lender_denial_rate") * 100, 1),
            "lender_apps": int(round(float(np.expm1(d["lender_log_volume"].median())))),
            **{c: mode(c) for c in ("loan_type", "preapproval", "conforming_loan_limit", "dwelling", "submission", "aus")},
        }  # fmt: skip

    scores = test["score"].to_numpy()
    q = pd.qcut(pd.Series(scores).rank(method="first"), 20, labels=False)
    bands = (
        pd.DataFrame({"band": q, "score": scores, "denied": test["denied"].to_numpy(),
                      "p": test["p_scorecard"].to_numpy()})
        .groupby("band").agg(lo=("score", "min"), hi=("score", "max"), n=("denied", "size"),
                             denial_rate=("denied", "mean"), predicted=("p", "mean"))
        .round(4).to_dict(orient="records")
    )  # fmt: skip
    options = {c: [{"value": v, "label": CONFORMING.get(v, v) if c == "conforming_loan_limit" else v}
                   for v in test[c].astype(str).value_counts().index]
               for c in ("loan_type", "preapproval", "conforming_loan_limit", "dwelling", "submission", "aus")}  # fmt: skip
    used = set(sc.binnings)
    return {
        "years": years, "test_year": years["test"], "test_applications": int(len(test)),
        "test_denial_rate": float(test["denied"].mean()),
        "scaling": {"pdo": sc.pdo, "base_score": sc.base_score, "base_odds": sc.base_odds,
                    "factor": factor, "offset": offset},
        "fallback_points": round(_fallback_points(sc), 2), "features": feats,
        "inputs": {k: bool(used & set(v)) for k, v in INPUTS.items()},
        "options": options, "presets": {"Typical approved application": profile(approved),
                                        "Typical denied application": profile(denied)},
        "percentiles": np.percentile(scores, np.arange(101)).round(2).tolist(), "bands": bands,
    }  # fmt: skip


def score(card: dict, values: dict) -> float:
    """The same calculation the page does in JavaScript (used by the tests to check the page's numbers)."""
    total = 0.0
    for f in card["features"]:
        v = values.get(f["feature"])
        if f["kind"] == "numeric":
            if v is None or (isinstance(v, float) and np.isnan(v)):
                b = next((x for x in f["bins"] if x["label"] == "Missing"), None)
            else:
                i = sum(1 for e in f["edges"] if e <= v)
                b = next(
                    (
                        x
                        for x in f["bins"]
                        if not x.get("missing") and x.get("lo") == (None if i == 0 else f["edges"][i - 1])
                    ),
                    None,
                )
        else:
            b = next((x for x in f["bins"] if x["label"] == str(v)), None)
            b = b or next((x for x in f["bins"] if x["label"] == "Other"), None)
        total += b["points"] if b else card["fallback_points"]
    return total


# --------------------------------------------------------------------------- page
PAGE_CSS = """
.scorer { display:grid; grid-template-columns:minmax(0,360px) minmax(0,1fr); gap:20px; align-items:start; }
@media (max-width: 860px) { .scorer { grid-template-columns:minmax(0,1fr); } }
.form label { display:block; font-size:13px; color:#52514e; margin:10px 0 3px; }
.form input, .form select { width:100%; padding:7px 9px; border:1px solid #c3c2b7; border-radius:8px; font:inherit;
  font-size:14px; background:#fff; }
.form .hint { font-size:11.5px; color:#898781; margin-top:2px; }
.form h3 { margin:18px 0 0; font-size:13px; text-transform:uppercase; letter-spacing:.05em; color:#898781; }
.presets { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:6px; }
.presets button { border:1px solid #c3c2b7; background:#fff; border-radius:999px; padding:3px 11px; font-size:13px;
  color:#104281; cursor:pointer; }
.big { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin-bottom:14px; }
.big .card .value { font-size:30px; }
.factor { display:grid; grid-template-columns:minmax(0,190px) minmax(0,1fr) 64px; gap:10px; align-items:center;
  padding:6px 0; border-bottom:1px solid #e1e0d9; font-size:13.5px; }
.factor .name b { display:block; font-weight:600; } .factor .name span { color:#898781; font-size:12px; }
.track { position:relative; height:16px; background:#f4f3ef; border-radius:4px; }
.track .mid { position:absolute; left:50%; top:-3px; bottom:-3px; width:1px; background:#898781; }
.track .bar { position:absolute; top:2px; bottom:2px; border-radius:4px; }
.factor .pts { text-align:right; font-variant-numeric:tabular-nums; }
@media (max-width: 560px) {  /* phones: name and points on one line, the bar full width below */
  .factor { grid-template-columns:minmax(0,1fr) 64px; grid-template-areas:"name pts" "track track"; row-gap:4px; }
  .factor .name { grid-area:name; } .factor .track { grid-area:track; } .factor .pts { grid-area:pts; }
}
.legend { font-size:12px; color:#52514e; display:flex; gap:14px; flex-wrap:wrap; margin:6px 0 2px; }
.legend i { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:-1px; }
ul.reasons { margin:6px 0 12px; padding-left:20px; color:#52514e; }
.callout { border-left:4px solid #b7791f; background:#fbf6ea; padding:10px 14px; border-radius:6px; margin:14px 0;
  color:#52514e; font-size:13.5px; }
main a { color:#2a78d6; }
"""

PAGE_JS = """
const card = await (await fetch("data/scorecard.json")).json();
const $ = (id) => document.getElementById(id);
const used = new Set(card.features.map((f) => f.feature));
const fmtInt = (v) => Math.round(v).toLocaleString();
const pct = (v, d = 1) => (100 * v).toFixed(d) + "%";
// show only the inputs the scorecard uses
for (const [k, on] of Object.entries(card.inputs)) if (!on) $("row_" + k).style.display = "none";
for (const [c, opts] of Object.entries(card.options)) {
  const sel = $(c); if (!used.has(c)) { $("row_" + c).style.display = "none"; continue; }
  for (const o of opts) { const e = document.createElement("option"); e.value = o.value; e.textContent = o.label; sel.appendChild(e); }
}
const fields = ["loan_amount", "property_value", "income", "area_median", "dti", "loan_term", "total_units", "lender_rate",
  "lender_apps", ...Object.keys(card.options)];
function setValues(p) { for (const k of fields) if (p[k] !== undefined) $(k).value = p[k]; update(); }
for (const [name, p] of Object.entries(card.presets)) {
  const b = document.createElement("button"); b.type = "button"; b.textContent = name; b.onclick = () => setValues(p);
  $("presets").appendChild(b);
}
function features() {
  const n = (k) => { const v = parseFloat($(k).value); return Number.isFinite(v) ? v : null; };
  const loan = n("loan_amount"), value = n("property_value"), income = n("income"), area = n("area_median");
  const f = {
    loan_amount_k: loan === null ? null : loan / 1000, property_value_k: value === null ? null : value / 1000,
    cltv: loan && value ? (100 * loan) / value : null, dti: n("dti"), income_k: income === null ? null : income / 1000,
    loan_to_income: loan && income ? Math.min(loan / income, 25) : null,
    income_to_area_median: income && area ? Math.min(income / area, 25) : null,
    loan_term: n("loan_term"), total_units: n("total_units"),
    lender_denial_rate: n("lender_rate") === null ? null : n("lender_rate") / 100,
    lender_log_volume: n("lender_apps") === null ? null : Math.log1p(n("lender_apps")),
  };
  for (const c of Object.keys(card.options)) f[c] = $(c).value;
  return f;
}
function binFor(f, v) {
  if (f.kind === "numeric") {
    if (v === null) return f.bins.find((b) => b.label === "Missing");
    const i = f.edges.filter((e) => e <= v).length;   // same rule as numpy.searchsorted(side="right")
    const lo = i === 0 ? null : f.edges[i - 1];
    return f.bins.find((b) => !b.missing && b.lo === lo);
  }
  return f.bins.find((b) => b.label === v) || f.bins.find((b) => b.label === "Other");
}
const DISPLAY = {
  loan_amount_k: (v) => "$" + fmtInt(v * 1000), property_value_k: (v) => "$" + fmtInt(v * 1000), income_k: (v) => "$" + fmtInt(v * 1000),
  cltv: (v) => v.toFixed(1) + "%", dti: (v) => v + "%", loan_to_income: (v) => v.toFixed(2) + "×",
  income_to_area_median: (v) => v.toFixed(2) + "×", lender_denial_rate: (v) => pct(v), lender_log_volume: (v) => fmtInt(Math.expm1(v)) + " applications",
  loan_term: (v) => v + " months", total_units: (v) => String(v),
};
// bins in the same units as the inputs ("$275K to under $385K" rather than "[275, 385)")
const money = (k) => "$" + (k >= 1000 ? (k / 1000).toFixed(2).replace(/\.?0+$/, "") + "M" : Math.round(k) + "K");
const UNIT = {
  loan_amount_k: money, property_value_k: money, income_k: money,
  cltv: (v) => +v.toPrecision(3) + "%", dti: (v) => +v.toPrecision(3) + "%", lender_denial_rate: (v) => +(100 * v).toPrecision(3) + "%",
  loan_to_income: (v) => +v.toPrecision(3) + "×", income_to_area_median: (v) => +v.toPrecision(3) + "×",
  lender_log_volume: (v) => fmtInt(Math.expm1(v)) + " applications", loan_term: (v) => v + " months", total_units: (v) => String(v),
};
function binText(f, b) {
  if (!b) return "a value not seen in training";
  if (f.kind !== "numeric") return null;
  if (b.missing) return "missing";
  const u = UNIT[f.feature] || ((v) => String(+v.toPrecision(3)));
  if (b.lo === null) return "bin: under " + u(b.hi);
  if (b.hi === null) return "bin: " + u(b.lo) + " or more";
  return "bin: " + u(b.lo) + " to under " + u(b.hi);
}
function update() {
  const x = features();
  const s = card.scaling;
  let total = 0; const rows = [];
  for (const f of card.features) {
    const v = x[f.feature]; const b = binFor(f, v);
    const p = b ? b.points : card.fallback_points;
    total += p;
    rows.push({f, v, bin: binText(f, b), p, vsTypical: p - f.typical_points, gap: f.max_points - p});
  }
  const pDenial = 1 / (1 + Math.exp(-(s.offset - total) / s.factor));
  const below = card.percentiles.filter((q) => q < total).length;   // share of test-year applications scoring lower
  const band = card.bands.find((b) => total <= b.hi) || card.bands[card.bands.length - 1];
  $("score").textContent = Math.round(total);
  $("pd").textContent = pct(pDenial);
  $("pctl").textContent = Math.min(below, 100) + "%";
  $("band").textContent = pct(band.denial_rate);
  $("bandnote").textContent = `of ${band.n.toLocaleString()} applications scoring ${Math.round(band.lo)}–${Math.round(band.hi)} were denied`;
  const span = Math.max(...rows.map((r) => Math.abs(r.vsTypical)), 1);
  $("factors").innerHTML = rows.sort((a, b) => a.vsTypical - b.vsTypical).map((r) => {
    const w = (50 * Math.abs(r.vsTypical)) / span;
    const left = r.vsTypical < 0 ? 50 - w : 50;
    const shown = r.v === null || r.v === undefined ? "missing" : (DISPLAY[r.f.feature] ? DISPLAY[r.f.feature](r.v) : r.v);
    return `<div class="factor"><div class="name"><b>${r.f.label}</b><span>${shown}${r.bin ? " · " + r.bin : ""}</span></div>
      <div class="track" title="${r.vsTypical >= 0 ? "+" : ""}${r.vsTypical.toFixed(1)} points vs. the typical ${card.test_year} applicant"><div class="mid"></div>
      <div class="bar" style="left:${left}%;width:${w}%;background:${r.vsTypical < 0 ? "#eb6834" : "#2a78d6"}"></div></div>
      <div class="pts">${r.p.toFixed(0)} pts</div></div>`;
  }).join("");
  const reasons = rows.filter((r) => r.gap > 0.5).sort((a, b) => b.gap - a.gap).slice(0, 3);
  $("reasons").innerHTML = reasons.length ? reasons.map((r) => `<li><b>${r.f.label}</b>: ${r.p.toFixed(0)} of a possible
    ${r.f.max_points.toFixed(0)} points</li>`).join("") : "<li>Every factor is already in its best bin.</li>";
}
document.querySelectorAll(".form input, .form select").forEach((e) => e.addEventListener("input", update));
setValues(card.presets["Typical approved application"]);
"""


def write_page(site: Path, card: dict, repo_url: str, state_name: str, synthetic: bool) -> None:
    from .report_base import CSS

    (site / "data").mkdir(parents=True, exist_ok=True)
    (site / "data" / "scorecard.json").write_text(json.dumps(card, default=float))
    y = card["years"]
    train = ", ".join(map(str, y["train"]))
    banner = "<div class='synthetic'>SYNTHETIC TEST DATA: built by the test suite; these numbers are not real</div>" if synthetic else ""  # fmt: skip

    def num(key, label, hint="", step="any"):
        return (f"<div id='row_{key}'><label for='{key}'>{label}</label>"
                f"<input id='{key}' type='number' min='0' step='{step}' inputmode='decimal'>"
                f"{f'<div class=hint>{hint}</div>' if hint else ''}</div>")  # fmt: skip

    def sel(key, label):
        return f"<div id='row_{key}'><label for='{key}'>{label}</label><select id='{key}'></select></div>"

    form = "".join([
        "<h3>Loan and property</h3>", num("loan_amount", "Loan amount ($)", step="1000"),
        num("property_value", "Property value ($)", "Loan-to-value is loan ÷ value (no second lien assumed)", "1000"),
        sel("loan_type", "Loan type"), num("loan_term", "Loan term (months)", step="12"),
        num("total_units", "Units", step="1"), sel("dwelling", "Dwelling type"),
        sel("conforming_loan_limit", "Conforming loan limit"),
        "<h3>Applicant</h3>", num("income", "Annual income ($)", step="1000"),
        num("area_median", "Area median family income ($)", f"For the metro area; {state_name}'s typical value is pre-filled", "100"),
        num("dti", "Debt-to-income (%)", step="1"), sel("preapproval", "Preapproval"),
        "<h3>Lender and process</h3>",
        num("lender_rate", "Lender's past denial rate (%)", f"Learned from the training year(s), {train}", "0.1"),
        num("lender_apps", "Lender's applications in the training year(s)", step="1"),
        sel("submission", "Application channel"), sel("aus", "Underwriting system"),
    ])  # fmt: skip
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mortgage Scorecard Scorer</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🏠</text></svg>">
<style>{CSS}{PAGE_CSS}</style></head><body>{banner}
<header><div class="wrap"><h1>Try the scorecard</h1>
<p>Enter a home-purchase mortgage application and see how the weight-of-evidence scorecard scores it, factor by
factor. The scorecard was trained on {html.escape(state_name)} HMDA decisions from {train} and is compared here with
the {card["test_applications"]:,} applications of {card["test_year"]}, which it never saw.</p>
<a class="btn" href="./">Back to the dashboard</a><a class="btn" href="./#model-card">Model card</a>
<a class="btn" href="{repo_url}">Code on GitHub</a></div></header>
<main><section><div class="scorer">
<form class="form" onsubmit="return false"><div class="presets" id="presets"></div>{form}</form>
<div>
<div class="big">
<div class="card"><div class="label">Score</div><div class="value" id="score">–</div><div class="note">higher = more likely approved</div></div>
<div class="card"><div class="label">Predicted chance of denial</div><div class="value" id="pd">–</div><div class="note">from the scorecard's logistic model</div></div>
<div class="card"><div class="label">Scores higher than</div><div class="value" id="pctl">–</div><div class="note">of {card["test_year"]} applications</div></div>
<div class="card"><div class="label">Actually denied, similar scores</div><div class="value" id="band">–</div><div class="note" id="bandnote"></div></div>
</div>
<h3>Points by factor, compared with the typical {card["test_year"]} applicant</h3>
<div class="legend"><span><i style="background:#2a78d6"></i>more points than typical (helps)</span>
<span><i style="background:#eb6834"></i>fewer points than typical (hurts)</span></div>
<div id="factors"></div>
<h3>Factors costing the most points</h3><ul class="reasons" id="reasons"></ul>
<p class="callout">This shows how a credit scorecard turns an application into a score; it is not a credit decision.
It learned from lenders' past decisions, which can carry their biases, and HMDA has no credit scores, credit history,
assets or employment, which real underwriting relies on. Race, ethnicity, sex and age are not inputs; the
<a href="./#fair-lending">fair-lending audit</a> on the dashboard measures how outcomes differ across them.</p>
</div></div></section></main>
<footer>Data: Consumer Financial Protection Bureau, HMDA Data Browser (public data) ·
<a href="{repo_url}">Source code</a></footer>
<script type="module">{PAGE_JS}</script></body></html>"""
    (site / "scorer.html").write_text(page, encoding="utf-8")
    log.info("Scorer page written (%d scorecard factors)", len(card["features"]))
