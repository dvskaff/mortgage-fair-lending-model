# Model card: mortgage denial model (HMDA, Georgia)

_Generated automatically on 2026-11-08 20:13 UTC by the pipeline in this repository._

## Intended use
- **What it is:** a model of how lenders decided home-purchase mortgage applications, trained on public HMDA records, plus a fair-lending audit of those decisions and of the model.
- **Intended users:** analysts studying lending outcomes; a portfolio demonstration of model development, validation and fairness testing.
- **Not for:** approving or denying real applications, pricing, or judging any individual or lender. HMDA lacks credit scores and credit history, which drive real underwriting decisions.

## Data
- Source: CFPB HMDA Data Browser API (public loan-level records), state of Georgia.
- Sample: 348,927 home-purchase, first-lien, principal-residence applications for 1-4 unit homes that were approved or denied (withdrawn and incomplete applications excluded), restricted to complete files: income, debt-to-income, combined loan-to-value and property value all reported.
- Out-of-time split: train 2023 (119,203) · validation 2024 (115,824, early stopping, calibration, decision threshold) · test 2025 (113,900).
- Target: denied (action taken = 3) vs. approved (originated, or approved but not accepted).

## Features
- Used: loan amount, property value, combined loan-to-value, debt-to-income, income, loan-to-income, income relative to the area median, loan term, units, loan type, preapproval, conforming-limit status, dwelling type, application channel, automated underwriting system, and the lender's smoothed historical denial rate and size (from training years only, out-of-fold).
- Excluded on purpose: race, ethnicity, sex and age (used only for the audit); co-applicant presence (can stand in for marital status); census-tract demographics and geography (can act as proxies for race); and anything recorded after the decision (interest rate, rate spread, fees, denial reasons, purchaser). 'Initially payable to the institution' is also excluded: HMDA reports it as not applicable for most denied applications, so it leaks the outcome.
- Monotonic constraints (LightGBM): predicted risk can never fall as debt-to-income (%), combined loan-to-value (%), loan ÷ income rise, and never rise as income ($k), income ÷ area median rise.

## Performance (held-out year 2025)
| Model | ROC AUC | PR AUC | KS | Brier |
|---|---|---|---|---|
| LightGBM, monotonic + calibrated | 0.848 | 0.598 | 0.528 | 0.0583 |
| LightGBM, unconstrained | 0.850 | 0.610 | 0.530 | 0.0580 |
| WoE logistic scorecard | 0.800 | 0.401 | 0.443 | 0.0750 |
| Baseline: lender's past denial rate only | 0.679 | 0.225 | 0.269 | 0.0855 |

- Observed denial rate in 2025: 10.1%; mean calibrated prediction: 10.2%.
- Most influential features (mean |SHAP|): Lender's past denial rate, Debt-to-income (%), Preapproval, Combined loan-to-value (%), Loan type.
- Leakage check: the strongest single feature on its own reaches ROC AUC 0.688 (Debt-to-income (%)).
- Decision threshold for the audit: flag applications with predicted probability ≥ 0.202 (the share flagged in 2024 equals that year's actual denial rate).

## Fairness evaluation (held-out year)
| Characteristic | Actual denial-rate gap | Model flag-rate gap | Equalized-odds difference |
|---|---|---|---|
| Race and ethnicity | 10.2 pts | 9.9 pts | 6.6 pts |
| Sex | 4.8 pts | 5.7 pts | 4.1 pts |
| Age | 2.8 pts | 2.0 pts | 3.8 pts |

- Largest race/ethnicity gap: Black applicants were denied 16.8% of the time vs. 6.5% for Non-Hispanic White applicants (2.57×; approval adverse impact ratio 0.89).
- For Black applicants the odds of denial were 2.89× those of Non-Hispanic White applicants before adjustment and 1.76× (95% CI 1.57–1.97) after adjusting for the credit factors in the model's score. The credit factors HMDA records account for about 47% of the gap on the log-odds scale; the rest is statistically significant and would be a reason for a closer review with data HMDA does not contain (credit scores and histories).

## Limitations and ethical considerations
- HMDA has no credit scores, credit histories, assets or employment data, so differences that remain after adjustment cannot be attributed to discrimination; they mark where a closer review would start.
- The model learns past lending decisions, including any bias in them. It must not be used to make decisions.
- HMDA reports debt-to-income, loan-to-value, property value and income as not applicable when the lender made no credit decision on them or did not rely on them (often files denied early). Missing values would leak the outcome, so only complete files are modeled; the excluded files are mostly denials, so the audit describes decisions on complete files (the dashboard shows rates on all decided files too).
- Lender identity is used as a feature; lenders that serve different communities may differ in denial rates for reasons that are themselves worth auditing.
- Race and ethnicity are as reported in HMDA: usually self-reported, but for some in-person applications where the applicant declined, recorded by the lender from visual observation or surname. Applicants with no information form a separate 'Not provided' group.
- Adjusting the model with protected characteristics to equalize outcomes would itself be disparate treatment under ECOA, so no such mitigation is applied here.
