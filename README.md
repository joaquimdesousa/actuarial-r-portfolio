# Actuarial & Financial Risk Analytics Portfolio

**Author:** Joaquim De Sousa
**Language:** R
**Focus:** Insurance pricing, credit risk, market risk


---

## Repository Structure

```
actuarial-r-portfolio/
├── README.md                          ← You are here
├── utils.R                            ← Shared utilities: validation, metrics, logging
├── claims_frequency_severity.R        ← Project 1: Insurance claims pricing
├── credit_risk_scoring.R              ← Project 2: Credit risk PD modelling
├── market_risk_simulation.R           ← Project 3: Market risk VaR/CVaR
├── portfolio_walkthrough.Rmd          ← Stakeholder-ready narrative (renders to HTML)
├── tests/
│   └── test_core.R                    ← 40+ unit & integration tests
├── outputs/                           ← Generated CSVs and plots (gitignored)
│   ├── claims/
│   ├── credit/
│   └── market/
└── docs/
    └── index.html                     ← Interactive portfolio showcase (GitHub Pages)
```

---

## The Three Projects

### 1. Insurance Claims Frequency & Severity

**Business question:** What is the expected claims cost per policy, and which segments are riskiest?

| Step | What happens |
|------|-------------|
| **v0 (wrong)** | Poisson GLM without exposure offset — treats 6-month and 12-month policies identically, biasing frequency estimates |
| **v1 (fix)** | Add `offset(log(exposure))` to model claim *rates* instead of raw counts |
| **v2 (final)** | Formal overdispersion test → switch to Negative Binomial if warranted |

**Severity modelling** uses a Gamma GLM with covariates (not a single scalar), plus a truncated lognormal fit via `actuar` and a three-distribution comparison (Lognormal / Gamma / Weibull) using AIC. Q-Q diagnostics confirm distributional fit.

**Key techniques:** Poisson & Negative Binomial GLM, exposure offsets, Gamma GLM, truncated distribution fitting, `fitdistrplus` distribution comparison, overdispersion testing, collective risk model (E[N] × E[X])

---

### 2. Credit Risk Scoring & Portfolio Monitoring

**Business question:** What is each loan applicant's probability of default?

| Step | What happens |
|------|-------------|
| **v0 (wrong)** | Evaluate AUC on *training* data — inflated performance due to data leakage |
| **v1 (fix)** | Proper 75/25 train/test split with holdout evaluation |
| **v2 (final)** | LASSO regularisation via `glmnet` for variable selection + parsimony |

**Feature engineering** includes log-income, DTI², delinquency flags, credit history binning, and loan-to-income ratio. Variable selection uses **Information Value / Weight of Evidence** (IV/WoE) — the standard credit scoring methodology.

**Validation suite:** AUC, Gini coefficient, KS statistic, Brier score, decile calibration table, calibration plot, precision-recall curve, ROC curve, risk band construction with odds ratios.

**Key techniques:** Logistic regression, LASSO (glmnet), WoE/IV, feature engineering, ROC/AUC, calibration analysis, risk banding, Basel Expected Loss (PD × LGD × EAD)

---

### 3. Market Risk Simulation & Stress Testing

**Business question:** How much could a diversified portfolio lose under normal and stressed conditions?

| Step | What happens |
|------|-------------|
| **v0 (wrong)** | Simulate asset returns independently — understates tail risk by ignoring co-movement |
| **v1 (fix)** | Cholesky decomposition preserves correlation structure between equity, bonds, and commodities |
| **v2 (final)** | GARCH(1,1) models volatility clustering; simulation uses conditional volatility from the current regime |

**Stress testing** applies three economically motivated scenarios — GFC 2008, COVID 2020, and Stagflation — each with calibrated mean and volatility shocks. **Kupiec backtest** validates the VaR model against historical breaches (Basel requirement). **Tail dependence analysis** compares empirical crash co-occurrence to Gaussian copula predictions.

**Key techniques:** Monte Carlo simulation (50,000 paths), Cholesky decomposition, GARCH(1,1), EWMA fallback, VaR/CVaR at 95/99/99.5%, 10-day horizon scaling (Basel), component VaR, Kupiec POF test, copula tail dependence

---

## Shared Infrastructure (`utils.R`)

All three projects use a common utility module providing:

- **Logging:** `log_info()` / `log_warn()` with timestamps for audit trails
- **Data validation:** `assert_cols()`, `assert_no_negative()`, `assert_no_na()`, `assert_range()`
- **Statistical metrics:** `gini_coefficient()`, `ks_statistic()`, `brier_score()`, `rmse()`, `mae()`, `winsorise()`
- **Model monitoring:** `population_stability_index()` — detects distribution drift for model retraining
- **Feature selection:** `information_value()` — WoE/IV computation for credit scoring variables
- **Model diagnostics:** `overdispersion_test()`, `print_model_summary()`
- **Output management:** `ensure_dir()`, `save_csv()`, `save_plot()`
- **Consistent theming:** `theme_portfolio()` for all ggplot visualisations

---

## Test Suite

40+ tests covering unit, edge case, and integration scenarios:

```r
# Run the full test suite
testthat::test_dir("tests")
```

Coverage includes all utility functions (validation, statistics, output management), edge cases (NA handling, invalid inputs, boundary conditions), and integration tests that verify each project pipeline runs end-to-end and produces structurally correct outputs.

---

## Quick Start

### Prerequisites

R ≥ 4.1 with the following packages:

```r
install.packages(c(
  "tidyverse", "MASS", "actuar", "fitdistrplus", "broom",    # Project 1
  "glmnet", "pROC",                                            # Project 2
  "PerformanceAnalytics", "rugarch",                            # Project 3 (rugarch optional)
  "testthat", "rmarkdown", "scales"                             # Testing & reporting
))
```

### Run Everything

```r
# Run all three projects (generates outputs/ directory)
source("claims_frequency_severity.R")
source("credit_risk_scoring.R")
source("market_risk_simulation.R")

# Run the test suite
testthat::test_dir("tests")

# Render the stakeholder walkthrough to HTML
rmarkdown::render("portfolio_walkthrough.Rmd")
```

---

## Technical Skills Demonstrated

| Category | Techniques |
|----------|-----------|
| **Statistical Modelling** | GLM (Poisson, NegBin, Gamma, Logistic), LASSO, distribution fitting |
| **Risk Metrics** | VaR, CVaR, AUC, Gini, KS, Brier, PSI, IV/WoE |
| **Simulation** | Monte Carlo, Cholesky decomposition, GARCH(1,1), EWMA |
| **Actuarial** | Exposure offsets, truncated distributions, collective risk model, severity modelling |
| **Validation** | Kupiec backtest, calibration analysis, overdispersion test, Q-Q diagnostics |
| **Software Engineering** | Defensive programming, unit testing (testthat), logging, DRY utilities |
| **R Packages** | tidyverse, glmnet, pROC, actuar, fitdistrplus, rugarch, MASS, broom |

---

