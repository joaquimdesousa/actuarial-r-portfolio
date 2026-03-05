# =============================================================================
# credit_risk_scoring.R
# Project 2: Credit Risk Scoring & Portfolio Monitoring
# =============================================================================
# Author:  Joaquim De Sousa
# Purpose: Build a probability-of-default (PD) model for loan applicants,
#          validate it properly, and construct a risk-banded portfolio view
#          suitable for credit committee reporting.
#
# Narrative (v0 → failure → fix):
#   v0: Evaluate AUC on the TRAINING set → inflated performance (leakage).
#   v1: Proper train/test split with holdout evaluation.
#   v2: Add feature engineering (WoE-style binning, log transforms),
#       LASSO regularisation for variable selection, and comprehensive
#       discrimination + calibration diagnostics.
#
# Key packages: tidyverse, glmnet (LASSO), pROC (ROC curves), broom
#
# Outputs (saved to outputs/credit/):
#   - calibration_table.csv     : decile calibration (predicted vs observed)
#   - portfolio_summary.csv     : risk band summary for credit committee
#   - model_coefficients.csv    : final model coefficients with odds ratios
#   - roc_curve.png             : ROC curve comparison (v0 vs final)
#   - calibration_plot.png      : predicted vs observed default rate by decile
#   - discrimination_metrics.txt: AUC, Gini, KS, Brier summary
# =============================================================================

# --- Load dependencies --------------------------------------------------------
suppressPackageStartupMessages({
  library(tidyverse)    # data manipulation + ggplot2
  library(glmnet)       # LASSO / elastic net regularisation
  library(pROC)         # ROC curves and AUC
  library(broom)        # tidy model summaries
})
source("utils.R")       # shared helpers: assertions, Gini, KS, Brier, logging

# --- Configuration ------------------------------------------------------------
OUTPUT_DIR   <- "outputs/credit/"
SEED         <- 123
N_APPLICANTS <- 12000
TRAIN_PROP   <- 0.75                     # 75% train, 25% test

set.seed(SEED)
ensure_dir(OUTPUT_DIR)

log_info("=== Project 2: Credit Risk Scoring & Portfolio Monitoring ===")

# =============================================================================
# SECTION 0 — DATA SIMULATION
# =============================================================================
# Simulate a realistic loan portfolio with known default probabilities.
# Variables mirror what a bank would have at application: income, DTI,
# credit history, recent delinquencies, and loan purpose.

log_info("Simulating loan application data (n =", N_APPLICANTS, ")")

loans <- tibble(
  applicant_id = seq_len(N_APPLICANTS),

  # Annual income: log-normal (right-skewed, realistic for AU market)
  income = round(exp(rnorm(N_APPLICANTS, mean = log(75000), sd = 0.50)), 0),

  # Debt-to-income ratio: truncated normal in [0.01, 0.90]
  # Higher DTI → more leveraged → higher default risk
  dti = pmin(pmax(rnorm(N_APPLICANTS, mean = 0.28, sd = 0.12), 0.01), 0.90),

  # Years of credit history: longer history → more data → lower risk
  credit_history_years = pmax(0, round(rnorm(N_APPLICANTS, mean = 6, sd = 3))),

  # Number of delinquencies in past 12 months (Poisson distributed)
  delinq_12m = rpois(N_APPLICANTS, lambda = 0.35),

  # Loan purpose: Small business loans carry higher default risk
  purpose = sample(c("Auto", "Personal", "HomeImprovement", "SmallBusiness"),
                   N_APPLICANTS, replace = TRUE, prob = c(0.25, 0.35, 0.25, 0.15)),

  # Loan amount: correlated with income (higher income → larger loans)
  loan_amount = round(income * runif(N_APPLICANTS, 0.1, 0.5), 0)
)

# TRUE default probability (hidden from model)
# Logistic DGP with economically interpretable coefficients:
#   - Higher DTI increases default risk (β = +0.9)
#   - Higher income decreases risk (β = -0.000006 per dollar)
#   - Longer credit history decreases risk (β = -0.10 per year)
#   - Recent delinquencies increase risk (β = +0.25 per event)
#   - Small business loans carry extra risk (β = +0.35)
true_logit <- -3.2 +
  0.9  * loans$dti -
  0.000006 * loans$income -
  0.10 * loans$credit_history_years +
  0.25 * loans$delinq_12m +
  ifelse(loans$purpose == "SmallBusiness", 0.35, 0.0)

true_pd       <- 1 / (1 + exp(-true_logit))                 # inverse logit
loans$default <- rbinom(N_APPLICANTS, size = 1, prob = true_pd)

log_info("Default rate:", round(mean(loans$default) * 100, 2), "%")
log_info("Default count:", sum(loans$default), "/", N_APPLICANTS)

# =============================================================================
# SECTION 1 — DATA QUALITY & FEATURE ENGINEERING
# =============================================================================

log_info("Running data quality checks and feature engineering")

assert_cols(loans, c("applicant_id", "income", "dti", "credit_history_years",
                      "delinq_12m", "purpose", "default"))
assert_no_na(loans, c("income", "dti", "default"))
assert_range(loans$dti, 0, 1, "debt-to-income ratio")

# Feature engineering: create transformed variables that improve model fit
# These transforms are standard in credit scoring:
loans_fe <- loans %>%
  mutate(
    # Log-income: linearises the relationship with default
    # (doubling income from $50k→$100k matters more than $200k→$250k)
    log_income = log(income),

    # DTI squared: allows non-linear DTI effect (risk accelerates at high DTI)
    dti_sq = dti^2,

    # Delinquency flag: binary "any delinquency" captures the main signal
    # (difference between 0 and 1 delinquency matters more than 3 vs 4)
    has_delinq = as.integer(delinq_12m > 0),

    # Credit history bins: non-linear effect (new-to-credit is risky;
    # difference between 10 and 15 years matters less)
    credit_hist_bin = cut(credit_history_years,
                          breaks = c(-Inf, 2, 5, 10, Inf),
                          labels = c("0-2yr", "3-5yr", "6-10yr", "10yr+"),
                          ordered_result = TRUE),

    # Loan-to-income ratio: another leverage indicator
    lti = loan_amount / income
  )

# =============================================================================
# SECTION 1b — WEIGHT OF EVIDENCE & INFORMATION VALUE
# =============================================================================
# WoE/IV is the standard feature selection methodology in credit scoring.
# It measures each variable's predictive power for distinguishing defaults
# from non-defaults. This section uses the information_value() function
# from utils.R.

log_info("Computing Information Value for all features")

# Compute IV for each numeric feature
iv_results <- list()
for (feat_name in c("income", "dti", "credit_history_years", "delinq_12m", "loan_amount", "lti")) {
  iv_res <- information_value(loans_fe[[feat_name]], loans_fe$default, n_bins = 10)
  iv_results[[feat_name]] <- iv_res$iv_total
}

# Compute IV for categorical feature
iv_purpose <- information_value(as.factor(loans_fe$purpose), loans_fe$default)
iv_results[["purpose"]] <- iv_purpose$iv_total

# Compile IV summary
iv_summary <- tibble(
  Feature = names(iv_results),
  IV = round(unlist(iv_results), 4)
) %>%
  arrange(desc(IV)) %>%
  mutate(
    Strength = case_when(
      IV < 0.02 ~ "Not useful",
      IV < 0.10 ~ "Weak",
      IV < 0.30 ~ "Medium",
      IV < 0.50 ~ "Strong",
      TRUE       ~ "Suspicious (check)"
    )
  )

cat("\n=== Information Value Summary ===\n")
print(iv_summary)
log_info("Top predictors by IV:", paste(iv_summary$Feature[1:3], collapse = ", "))

# =============================================================================
# SECTION 2 — v0 (DELIBERATELY WRONG): Evaluate on Training Data
# =============================================================================
# WHY THIS IS WRONG: When we evaluate model performance on the same data
# used to fit the model, we measure memorisation, not generalisation.
# The AUC will be optimistically biased because the model has already
# "seen" these outcomes. In a real bank, this could lead to approving
# loans that default because the model appeared better than it truly is.

log_info("Fitting v0 model (evaluate on training data — LEAKAGE)")

v0_fit  <- glm(default ~ income + dti + credit_history_years + delinq_12m + purpose,
               data = loans_fe, family = binomial(link = "logit"))
v0_prob <- predict(v0_fit, type = "response")
v0_auc  <- as.numeric(pROC::auc(loans_fe$default, v0_prob))

log_warn("v0 AUC (training set — INFLATED):", round(v0_auc, 4))

# =============================================================================
# SECTION 3 — v1 (FIX): Proper Train/Test Split
# =============================================================================
# WHY: Holding out 25% of data for evaluation gives an unbiased estimate
# of how the model will perform on future, unseen applicants. This is
# the minimum acceptable methodology for any production credit model.

log_info("Splitting data:", TRAIN_PROP * 100, "% train /",
         (1 - TRAIN_PROP) * 100, "% test")

train_idx <- sample(seq_len(N_APPLICANTS), size = floor(TRAIN_PROP * N_APPLICANTS))
train     <- loans_fe[train_idx, ]
test      <- loans_fe[-train_idx, ]

log_info("Train set:", nrow(train), "rows (",
         round(mean(train$default) * 100, 2), "% default)")
log_info("Test set: ", nrow(test), "rows (",
         round(mean(test$default) * 100, 2), "% default)")

# ----- v1a: Logistic regression with engineered features ----------------------
v1_fit <- glm(
  default ~ log_income + dti + dti_sq + credit_history_years +
            delinq_12m + has_delinq + purpose + lti,
  data   = train,
  family = binomial(link = "logit")
)

print_model_summary(v1_fit, "v1: Logistic Regression (engineered features)")

# =============================================================================
# SECTION 4 — v2 (ADVANCED): LASSO Regularisation
# =============================================================================
# WHY: LASSO (L1 penalty) simultaneously performs variable selection and
# regularisation. It shrinks weak predictors' coefficients to exactly zero,
# producing a sparser, more interpretable model. This is increasingly
# standard in credit scoring where regulators expect parsimony.

log_info("Fitting v2 model (LASSO regularisation via glmnet)")

# Prepare model matrix (glmnet requires matrix input, not formula)
x_train <- model.matrix(
  ~ log_income + dti + dti_sq + credit_history_years +
    delinq_12m + has_delinq + purpose + lti,
  data = train
)[, -1]                                                      # drop intercept column

x_test <- model.matrix(
  ~ log_income + dti + dti_sq + credit_history_years +
    delinq_12m + has_delinq + purpose + lti,
  data = test
)[, -1]

# Cross-validated LASSO: finds optimal λ that minimises CV deviance
cv_lasso <- cv.glmnet(
  x      = x_train,
  y      = train$default,
  family = "binomial",
  alpha  = 1,                                                # 1 = LASSO, 0 = Ridge
  nfolds = 10,
  type.measure = "deviance"
)

# Use lambda.1se (more regularised than lambda.min) for parsimony
# This is the "one standard error rule" — select the simplest model
# whose CV error is within 1 SE of the minimum
optimal_lambda <- cv_lasso$lambda.1se
log_info("Optimal lambda (1SE rule):", round(optimal_lambda, 6))

# Extract non-zero coefficients to see what LASSO kept
lasso_coefs <- coef(cv_lasso, s = "lambda.1se")
n_nonzero   <- sum(lasso_coefs[-1] != 0)                    # exclude intercept
log_info("LASSO retained", n_nonzero, "of",
         ncol(x_train), "features")

cat("\nLASSO coefficients (lambda.1se):\n")
print(round(as.matrix(lasso_coefs), 6))

# =============================================================================
# SECTION 5 — MODEL EVALUATION ON HOLDOUT SET
# =============================================================================
# ALL metrics computed on the TEST set to avoid leakage.

log_info("Evaluating models on holdout test set")

# Predictions from each model
prob_v1    <- predict(v1_fit, newdata = test, type = "response")
prob_lasso <- as.numeric(predict(cv_lasso, newx = x_test,
                                  s = "lambda.1se", type = "response"))

# ----- Discrimination metrics -------------------------------------------------
# AUC: area under ROC curve; probability that model ranks a random default
#       higher than a random non-default. 0.5 = random, 1.0 = perfect.
auc_v1    <- as.numeric(pROC::auc(test$default, prob_v1))
auc_lasso <- as.numeric(pROC::auc(test$default, prob_lasso))

# Gini: = 2×AUC - 1. Preferred by many credit risk teams and regulators.
gini_v1    <- gini_coefficient(test$default, prob_v1)
gini_lasso <- gini_coefficient(test$default, prob_lasso)

# KS: maximum separation between default and non-default CDFs.
# Higher = better discrimination. Typical good models: 30-50%.
ks_v1    <- ks_statistic(test$default, prob_v1)
ks_lasso <- ks_statistic(test$default, prob_lasso)

# Brier: mean squared prediction error. Lower = better calibration.
brier_v1    <- brier_score(test$default, prob_v1)
brier_lasso <- brier_score(test$default, prob_lasso)

# Print comprehensive comparison
cat("\n", strrep("=", 60), "\n")
cat(" DISCRIMINATION & CALIBRATION METRICS (Test Set)\n")
cat(strrep("=", 60), "\n")
cat(sprintf("%-25s %12s %12s %12s\n", "Metric", "v0 (leaked)", "v1 (logistic)", "v2 (LASSO)"))
cat(strrep("-", 60), "\n")
cat(sprintf("%-25s %12.4f %12.4f %12.4f\n", "AUC",   v0_auc, auc_v1, auc_lasso))
cat(sprintf("%-25s %12s %12.4f %12.4f\n",   "Gini",  "N/A",  gini_v1, gini_lasso))
cat(sprintf("%-25s %12s %12.4f %12.4f\n",   "KS",    "N/A",  ks_v1, ks_lasso))
cat(sprintf("%-25s %12s %12.4f %12.4f\n",   "Brier", "N/A",  brier_v1, brier_lasso))
cat(strrep("=", 60), "\n\n")

# Save metrics to text file
metrics_text <- paste(
  "Credit Risk Model — Discrimination Metrics (Test Set)",
  paste(strrep("=", 50)),
  sprintf("v0 AUC (training — LEAKED):   %.4f", v0_auc),
  sprintf("v1 AUC (logistic, test):      %.4f", auc_v1),
  sprintf("v2 AUC (LASSO, test):         %.4f", auc_lasso),
  "",
  sprintf("v1 Gini: %.4f  |  v2 Gini: %.4f", gini_v1, gini_lasso),
  sprintf("v1 KS:   %.4f  |  v2 KS:   %.4f", ks_v1, ks_lasso),
  sprintf("v1 Brier: %.4f |  v2 Brier: %.4f", brier_v1, brier_lasso),
  sep = "\n"
)
writeLines(metrics_text, paste0(OUTPUT_DIR, "discrimination_metrics.txt"))

# ----- ROC curve plot ---------------------------------------------------------
roc_v1    <- pROC::roc(test$default, prob_v1, quiet = TRUE)
roc_lasso <- pROC::roc(test$default, prob_lasso, quiet = TRUE)

roc_data <- bind_rows(
  tibble(fpr = 1 - roc_v1$specificities,    tpr = roc_v1$sensitivities,
         model = paste0("v1 Logistic (AUC=", round(auc_v1, 3), ")")),
  tibble(fpr = 1 - roc_lasso$specificities,  tpr = roc_lasso$sensitivities,
         model = paste0("v2 LASSO (AUC=", round(auc_lasso, 3), ")"))
)

p_roc <- ggplot(roc_data, aes(x = fpr, y = tpr, colour = model)) +
  geom_line(linewidth = 0.9) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey60") +
  scale_colour_manual(values = c("steelblue", "darkred")) +
  labs(
    title    = "ROC Curve Comparison: Logistic vs LASSO",
    subtitle = "Both evaluated on holdout test set (no leakage)",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)",
    colour = "Model",
    caption = "Portfolio Project 2 | Joaquim De Sousa"
  ) +
  theme_portfolio() +
  coord_equal()

save_plot(p_roc, paste0(OUTPUT_DIR, "roc_curve.png"))

# =============================================================================
# SECTION 6 — CALIBRATION ANALYSIS
# =============================================================================
# A well-discriminating model can still be poorly calibrated (predicted
# probabilities don't match observed rates). This matters for capital
# calculations where PD estimates feed directly into Expected Loss = PD × LGD × EAD.

log_info("Computing calibration table")

# Use the better-performing model for calibration
best_prob <- if (auc_lasso >= auc_v1) prob_lasso else prob_v1
best_name <- if (auc_lasso >= auc_v1) "LASSO" else "Logistic"

calibration <- test %>%
  mutate(
    predicted_pd = best_prob,
    decile       = ntile(predicted_pd, 10)                   # 10 equal-sized buckets
  ) %>%
  group_by(decile) %>%
  summarise(
    n_applicants    = n(),
    n_defaults      = sum(default),
    observed_rate   = round(mean(default), 4),               # actual default rate
    avg_predicted   = round(mean(predicted_pd), 4),          # model's average PD
    min_predicted   = round(min(predicted_pd), 4),
    max_predicted   = round(max(predicted_pd), 4),
    .groups = "drop"
  ) %>%
  mutate(
    # Calibration error per bucket: |predicted - observed|
    abs_error = round(abs(avg_predicted - observed_rate), 4)
  )

cat("\n=== Calibration Table (", best_name, ") ===\n")
print(calibration)

save_csv(calibration, paste0(OUTPUT_DIR, "calibration_table.csv"))

# ----- Calibration plot -------------------------------------------------------
p_cal <- ggplot(calibration, aes(x = avg_predicted, y = observed_rate)) +
  geom_point(aes(size = n_applicants), colour = "steelblue", alpha = 0.8) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey50") +
  geom_smooth(method = "lm", se = FALSE, colour = "darkred", linewidth = 0.7) +
  scale_size_continuous(range = c(3, 8), guide = "none") +
  labs(
    title    = paste("Calibration Plot:", best_name, "Model"),
    subtitle = "Perfect calibration lies on the diagonal; dot size = bucket count",
    x = "Average predicted PD",
    y = "Observed default rate",
    caption = "Portfolio Project 2 | Joaquim De Sousa"
  ) +
  theme_portfolio() +
  coord_equal(xlim = c(0, max(calibration$avg_predicted) * 1.1),
              ylim = c(0, max(calibration$observed_rate) * 1.1))

save_plot(p_cal, paste0(OUTPUT_DIR, "calibration_plot.png"))

# ----- Precision-Recall curve ------------------------------------------------
# PR curves are more informative than ROC when classes are imbalanced.
# In credit risk, defaults are typically 2-8% of applications, making
# precision (of those we flag, how many actually default?) crucial.

log_info("Computing precision-recall curve")

# Compute precision and recall at various thresholds
thresholds <- seq(0.01, 0.99, by = 0.01)
pr_data <- map_dfr(thresholds, function(t) {
  pred_class <- as.integer(best_prob >= t)
  tp <- sum(pred_class == 1 & test$default == 1)
  fp <- sum(pred_class == 1 & test$default == 0)
  fn <- sum(pred_class == 0 & test$default == 1)
  tibble(
    threshold = t,
    precision = ifelse(tp + fp > 0, tp / (tp + fp), NA_real_),
    recall    = ifelse(tp + fn > 0, tp / (tp + fn), NA_real_)
  )
}) %>%
  filter(!is.na(precision), !is.na(recall))

p_pr <- ggplot(pr_data, aes(x = recall, y = precision)) +
  geom_line(colour = "steelblue", linewidth = 0.9) +
  geom_hline(yintercept = mean(test$default), linetype = "dashed",
             colour = "grey50", linewidth = 0.5) +
  annotate("text", x = 0.8, y = mean(test$default) + 0.02,
           label = "Baseline (prevalence)", colour = "grey50", size = 3) +
  labs(
    title    = "Precision-Recall Curve",
    subtitle = paste0("Model: ", best_name, " | Higher area = better at identifying defaults"),
    x = "Recall (sensitivity)", y = "Precision (positive predictive value)",
    caption = "Portfolio Project 2 | Joaquim De Sousa"
  ) +
  theme_portfolio() +
  coord_cartesian(ylim = c(0, 1))

save_plot(p_pr, paste0(OUTPUT_DIR, "precision_recall.png"))

# =============================================================================
# SECTION 7 — RISK BAND CONSTRUCTION & PORTFOLIO SUMMARY
# =============================================================================
# Segment applicants into risk bands for underwriting decisions.
# These bands would typically be reviewed quarterly by credit committee.

log_info("Constructing risk bands")

# Define risk bands using predicted PD thresholds
# Thresholds based on quantiles of the PD distribution
cutoffs <- quantile(best_prob, probs = c(0.60, 0.85, 0.95))

test_out <- test %>%
  mutate(
    predicted_pd = best_prob,
    risk_band = case_when(
      predicted_pd <= cutoffs[1] ~ "1-Low",
      predicted_pd <= cutoffs[2] ~ "2-Medium",
      predicted_pd <= cutoffs[3] ~ "3-High",
      TRUE                        ~ "4-Very High"
    )
  )

portfolio <- test_out %>%
  group_by(risk_band) %>%
  summarise(
    n_applicants        = n(),
    pct_of_portfolio    = round(n() / nrow(test_out) * 100, 1),
    observed_default_rate = round(mean(default), 4),
    avg_predicted_pd    = round(mean(predicted_pd), 4),
    avg_income          = round(mean(income), 0),
    avg_dti             = round(mean(dti), 3),
    avg_loan_amount     = round(mean(loan_amount), 0),
    # Expected Loss assuming 40% LGD (Loss Given Default) — Basel standard
    expected_loss_pct   = round(mean(predicted_pd) * 0.40 * 100, 2),
    .groups = "drop"
  )

cat("\n=== Portfolio Risk Band Summary ===\n")
print(portfolio)

save_csv(portfolio, paste0(OUTPUT_DIR, "portfolio_summary.csv"))

# ----- Save model coefficients with odds ratios --------------------------------
if (best_name == "Logistic") {
  coef_table <- tidy(v1_fit, conf.int = TRUE, exponentiate = FALSE) %>%
    mutate(
      odds_ratio = exp(estimate),                            # OR = exp(β)
      or_lower   = exp(conf.low),
      or_upper   = exp(conf.high)
    ) %>%
    mutate(across(where(is.numeric), ~ round(., 4)))

  cat("\n=== Model Coefficients with Odds Ratios ===\n")
  print(coef_table %>% select(term, estimate, odds_ratio, or_lower, or_upper, p.value))
  save_csv(coef_table, paste0(OUTPUT_DIR, "model_coefficients.csv"))
}

# =============================================================================
# SECTION 8 — FINAL SUMMARY
# =============================================================================

cat("\n", strrep("=", 60), "\n")
cat(" PROJECT 2 COMPLETE: Credit Risk Scoring\n")
cat(strrep("=", 60), "\n")
cat("Best model:         ", best_name, "\n")
cat("AUC (test):         ", round(max(auc_v1, auc_lasso), 4), "\n")
cat("Gini (test):        ", round(max(gini_v1, gini_lasso), 4), "\n")
cat("KS (test):          ", round(max(ks_v1, ks_lasso), 4), "\n")
cat("v0 AUC (LEAKED):    ", round(v0_auc, 4),
    " → shows", round((v0_auc - max(auc_v1, auc_lasso)) * 100, 1),
    "pp inflation\n")
cat("Risk bands:          4 (Low / Medium / High / Very High)\n")
cat("Outputs saved to:   ", OUTPUT_DIR, "\n")
cat(strrep("=", 60), "\n")
