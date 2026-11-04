# Mortgage Denial Modeling & Fair-Lending Audit (HMDA)

[![CI](https://github.com/dvskaff/mortgage-fair-lending-model/actions/workflows/ci.yml/badge.svg)](https://github.com/dvskaff/mortgage-fair-lending-model/actions/workflows/ci.yml)
[![Build model](https://github.com/dvskaff/mortgage-fair-lending-model/actions/workflows/pipeline.yml/badge.svg)](https://github.com/dvskaff/mortgage-fair-lending-model/actions/workflows/pipeline.yml)
![Python](https://img.shields.io/badge/python-3.13-blue) ![LightGBM](https://img.shields.io/badge/LightGBM-4.7-green)
![Fairlearn](https://img.shields.io/badge/Fairlearn-0.15-purple)

**Live dashboard: https://dvskaff.github.io/mortgage-fair-lending-model/** · **[Model card](MODEL_CARD.md)**

Every year, most U.S. mortgage lenders must report each application they receive under the Home Mortgage Disclosure Act
(HMDA). This project uses Georgia's public HMDA records to answer two questions:

1. **How well can mortgage denials be predicted from the information a lender has at application time?**
   A monotonic, calibrated LightGBM model and a classic weight-of-evidence credit scorecard, trained on older years
   and tested on the newest year (an out-of-time test), explained with SHAP.
2. **Do outcomes differ by race, ethnicity, sex or age, and do the differences remain once credit factors are
   accounted for?** A fair-lending audit of both the lenders' actual decisions and the model: adverse impact ratios,
   Fairlearn error-rate and parity metrics, calibration by group, and adjusted odds ratios with lender-clustered
   standard errors.

## Results

Rewritten automatically after every run.

<!-- RESULTS:START -->
_HMDA Georgia home-purchase applications, 2023–2025 · 348,927 applications · trained on 2023, validated on 2024, tested on 2025_

| Model (held-out year) | ROC AUC | PR AUC | KS | Brier |
|---|---|---|---|---|
| LightGBM, monotonic + calibrated | 0.847 | 0.596 | 0.527 | 0.0583 |
| LightGBM, unconstrained | 0.851 | 0.610 | 0.533 | 0.0580 |
| WoE logistic scorecard | 0.800 | 0.400 | 0.441 | 0.0751 |
| Baseline: lender's past denial rate only | 0.679 | 0.225 | 0.269 | 0.0855 |

**Denial rate in 2025:** 10.1% · **top features (SHAP):** Lender's past denial rate, Debt-to-income (%), Preapproval, Combined loan-to-value (%), Loan type

**Largest race/ethnicity gap:** Black applicants were denied 16.8% of the time vs. 6.5% for Non-Hispanic White applicants (2.57×).
Odds ratio of denial 2.89 unadjusted → 1.76 (95% CI 1.57–1.97) after adjusting for the credit factors in the model's score. HMDA has no credit scores, so this is a flag for review, not proof of discrimination.

| Fairness (held-out year) | Actual denial-rate gap | Model flag-rate gap | Equalized-odds diff. |
|---|---|---|---|
| Race and ethnicity | 10.2 pts | 10.2 pts | 6.8 pts |
| Sex | 4.8 pts | 5.9 pts | 3.9 pts |
| Age | 2.8 pts | 2.1 pts | 3.6 pts |

_Last run: 2026-11-04 14:16 UTC_
<!-- RESULTS:END -->

## How it works

```
CFPB HMDA Data Browser API (public, no key): one CSV per year, newest 3 published years found automatically
        │  src/mortgage/download.py  streamed download, column-name normalization and schema check → DuckDB raw.lar
        ▼
DuckDB SQL (src/mortgage/prepare.py)
        typed + filtered application table: home purchase, first lien, principal residence, 1-4 units,
        approved or denied · DTI bands → numbers · audit groups (race/ethnicity, sex, age) kept apart from features
        ▼
Models (src/mortgage/model.py)            train: oldest year(s) · validation: middle year · test: newest year
        WoE logistic scorecard   monotonic binning, information-value selection, sign checks, points (PDO 20)
        LightGBM                 monotonic constraints, early stopping, isotonic calibration on the validation year
        lender history           smoothed denial rate + size, out-of-fold on training years (no target leakage)
        SHAP                     TreeSHAP importance and dependence
        ▼
Fair-lending audit (src/mortgage/fairness.py), on the held-out year
        denial rates + Wilson CIs · approval adverse impact ratios (four-fifths screen)
        Fairlearn MetricFrame: flag rates, false-positive / false-negative rates, demographic parity, equalized odds
        calibration by group · unadjusted vs. adjusted odds ratios (statsmodels, lender-clustered SEs)
        ▼
Dashboard (GitHub Pages) · MODEL_CARD.md · this README's results block  ← GitHub Actions, monthly
```

### Design choices

* **Only application-time information is used as a feature.** Race, ethnicity, sex and age are excluded (they are
  only used for the audit), as are census-tract demographics and geography (possible proxies for race), whether there
  is a co-applicant (can stand in for marital status), and anything recorded after the decision (interest rate, fees,
  denial reasons, purchaser).
* **No outcome leakage from HMDA's reporting rules.** HMDA reports "initially payable to the institution" as not
  applicable for most denied applications, and debt-to-income, loan-to-value, property value and income as not
  applicable when the lender made no credit decision on them or did not rely on them. The first field is excluded and
  the models are trained and audited on complete files (all four reported); the dashboard shows denial rates with and
  without that restriction.
* **Out-of-time evaluation.** Models never see the test year, the way a model is used in practice.
* **Monotonic constraints.** Predicted risk can never fall as debt-to-income, loan-to-value or loan-to-income rise, or
  rise as income rises. The dashboard shows what the constraints cost by comparing against an unconstrained model.
* **No "fairness fix".** Adjusting a lending model with protected characteristics would itself be disparate
  treatment under the Equal Credit Opportunity Act, so the project measures and documents instead.

### Limitations

HMDA has no credit scores, credit histories, assets or employment data. A difference that remains after adjusting for
the credit factors HMDA does record is therefore a reason for a closer look, not proof of discrimination. The model
learns from past lending decisions and can reproduce any bias in them; it is not meant for making credit decisions.

## Repository layout

```
src/mortgage/      config · download · prepare · model · fairness · report (+ report_base) · readme · pipeline
                   fixture.py: SYNTHETIC HMDA-format data for tests (pages built from it carry a red banner)
tests/             downloader (fake HTTP), preparation, models, fairness metrics, end-to-end run
.github/workflows/ ci.yml (lint + tests on every push) · pipeline.yml (monthly real-data run → GitHub Pages)
```

## Run it

Everything runs in GitHub Actions: **Actions → Build model → Run workflow**. Locally:

```bash
pip install -r requirements-dev.txt
pytest -q
PYTHONPATH=src python -m mortgage.pipeline            # real data (downloads a few hundred MB)
PYTHONPATH=src python -m mortgage.pipeline --fixture  # synthetic data, offline
```

Data: Consumer Financial Protection Bureau, [HMDA Data Browser](https://ffiec.cfpb.gov/data-browser/) (public data).
Code: MIT license.
