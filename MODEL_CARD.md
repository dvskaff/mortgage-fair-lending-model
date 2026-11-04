# Model card: mortgage denial model (HMDA, Georgia)

_Generated automatically on 2026-11-04 13:45 UTC by the pipeline in this repository._

## Intended use
- **What it is:** a model of how lenders decided home-purchase mortgage applications, trained on public HMDA records, plus a fair-lending audit of those decisions and of the model.
- **Intended users:** analysts studying lending outcomes; a portfolio demonstration of model development, validation and fairness testing.
- **Not for:** approving or denying real applications, pricing, or judging any individual or lender. HMDA lacks credit scores and credit history, which drive real underwriting decisions.

## Data
- Source: CFPB HMDA Data Browser API (public loan-level records), state of Georgia.
- Sample: 403,351 home-purchase, first-lien, principal-residence applications for 1-4 unit homes that were approved or denied (withdrawn and incomplete files excluded).
- Out-of-time split: train 2023 (137,135) · validation 2024 (134,285, early stopping, calibration, decision threshold) · test 2025 (131,931).
- Target: denied (action taken = 3) vs. approved (originated, or approved but not accepted).

## Features
- Used: loan amount, property value, combined loan-to-value, debt-to-income, income, loan-to-income, income relative to the area median, loan term, units, loan type, preapproval, conforming-limit status, dwelling type, application channel, whether the loan was payable to the lender, automated underwriting system, and the lender's smoothed historical denial rate and size (from training years only, out-of-fold).
- Excluded on purpose: race, ethnicity, sex and age (used only for the audit); co-applicant presence (can stand in for marital status); census-tract demographics and geography (can act as proxies for race); and anything recorded after the decision (interest rate, rate spread, fees, denial reasons, purchaser).
- Monotonic constraints (LightGBM): predicted risk can never fall as debt-to-income (%), combined loan-to-value (%), loan ÷ income rise, and never rise as income ($k), income ÷ area median rise.

## Performance (held-out year 2025)
| Model | ROC AUC | PR AUC | KS | Brier |
|---|---|---|---|---|
| LightGBM, monotonic + calibrated | 0.921 | 0.806 | 0.698 | 0.0562 |
| LightGBM, unconstrained | 0.923 | 0.817 | 0.701 | 0.0555 |
| WoE logistic scorecard | 0.873 | 0.676 | 0.594 | 0.0776 |
| Baseline: lender's past denial rate only | 0.768 | 0.460 | 0.392 | 0.1013 |

- Observed denial rate in 2025: 15.0%; mean calibrated prediction: 15.1%.
- Most influential features (mean |SHAP|): Debt-to-income (%), Lender's past denial rate, Lender size (log applications), Underwriting system, Preapproval.
- Decision threshold for the audit: flag applications with predicted probability ≥ 0.282 (the share flagged in 2024 equals that year's actual denial rate).

## Fairness evaluation (held-out year)
| Characteristic | Actual denial-rate gap | Model flag-rate gap | Equalized-odds difference |
|---|---|---|---|
| Race and ethnicity | 14.7 pts | 15.4 pts | 14.9 pts |
| Sex | 8.0 pts | 7.6 pts | 6.1 pts |
| Age | 2.2 pts | 1.4 pts | 3.6 pts |

- Largest race/ethnicity gap: Black applicants were denied 24.2% of the time vs. 10.8% for Non-Hispanic White applicants (2.24×; approval adverse impact ratio 0.85).
- For Black applicants the odds of denial were 2.64× those of Non-Hispanic White applicants before adjustment and 1.83× (95% CI 1.61–2.08) after adjusting for the credit factors in the model's score. The credit factors HMDA records account for about 38% of the gap on the log-odds scale; the rest is statistically significant and would be a reason for a closer review with data HMDA does not contain (credit scores and histories).

## Limitations and ethical considerations
- HMDA has no credit scores, credit histories, assets or employment data, so differences that remain after adjustment cannot be attributed to discrimination; they mark where a closer review would start.
- The model learns past lending decisions, including any bias in them. It must not be used to make decisions.
- Fields such as debt-to-income can be reported as not applicable when a lender did not rely on them (often on files denied early), so missing values carry some information about the decision process.
- Lender identity is used as a feature; lenders that serve different communities may differ in denial rates for reasons that are themselves worth auditing.
- Race and ethnicity are as reported in HMDA: usually self-reported, but for some in-person applications where the applicant declined, recorded by the lender from visual observation or surname. Applicants with no information form a separate 'Not provided' group.
- Adjusting the model with protected characteristics to equalize outcomes would itself be disparate treatment under ECOA, so no such mitigation is applied here.
