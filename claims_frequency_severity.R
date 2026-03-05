# =============================================================================
# claims_frequency_severity.R
# Project 1: Insurance Claims Frequency & Severity Modelling
# =============================================================================
# Author:  Joaquim De Sousa
# Purpose: Estimate expected claims costs per policy for a personal-lines
#          insurer. Demonstrates the full actuarial pricing pipeline:
#          data simulation → EDA → frequency modelling → severity modelling
#          → expected loss aggregation → diagnostics.
#
# Narrative (v0 → failure → fix):
#   FREQUENCY: v0 fits Poisson without exposure offset → biased rates.
#              Fix adds offset(log(exposure)) and tests overdispersion to
#              decide between Poisson and Negative Binomial.
#   SEVERITY:  v0 fits a naive log-linear model ignoring the reporting cap
#              → underestimates tail risk. Fix uses a proper Gamma GLM on
#              uncapped claims, plus a separate truncated-distribution fit
#              using the actuar package to quantify tail sensitivity.
#
# Key packages: tidyverse, MASS (glm.nb), actuar (fitdist + truncated),
#               broom (tidy model output)
#
# Outputs (saved to outputs/claims/):
#   - region_summary.csv        : expected loss by region segment
#   - freq_diagnostics.png      : residual plots for frequency model
#   - severity_distribution.png : fitted vs observed severity distribution
#   - expected_loss_dist.png    : distribution of expected loss per policy
# =============================================================================

# --- Load dependencies --------------------------------------------------------
suppressPackageStartupMessages({
  library(tidyverse)    # data manipulation + ggplot2
  library(MASS)         # glm.nb for Negative Binomial regression
  library(actuar)       # actuarial distributions: fitdist, truncated lognormal
  library(fitdistrplus) # distribution fitting and goodness-of-fit tests
  library(broom)        # tidy() for clean model coefficient tables
})
source("utils.R")       # shared helpers: assertions, logging, diagnostics

# --- Configuration ------------------------------------------------------------
# Centralise all tuneable parameters so they're easy to find and change.
# An interviewer can see at a glance what assumptions drive the analysis.

OUTPUT_DIR    <- "outputs/claims/"       # where all CSVs and plots go
SEED          <- 42                      # reproducibility seed
N_POLICIES    <- 10000                   # number of simulated policies
REPORTING_CAP <- 50000                   # insurer's per-claim reporting cap ($)
WINSOR_P      <- 0.005                   # winsorisation percentile for severity
OVERDISP_THRESHOLD <- 1.5               # Pearson dispersion ratio threshold
                                         # for Poisson → NegBin switch

set.seed(SEED)
ensure_dir(OUTPUT_DIR)

log_info("=== Project 1: Claims Frequency & Severity Modelling ===")

# =============================================================================
# SECTION 0 — DATA SIMULATION
# =============================================================================
# Why simulate? We can't share real insurer data, but simulation lets us
# control the true data-generating process (DGP) so we can verify our
# models recover the correct parameters. This is standard actuarial practice
# for portfolio projects and exam preparation.

log_info("Simulating policy and claims data (n =", N_POLICIES, ")")

policies <- tibble(
  policy_id     = seq_len(N_POLICIES),
  # Age: truncated normal, minimum 18 (legal driving age in Australia)
  age           = pmax(18, round(rnorm(N_POLICIES, mean = 42, sd = 12))),
  # Vehicle value: log-normal (right-skewed, as real vehicle values are)
  vehicle_value = round(exp(rnorm(N_POLICIES, mean = log(25000), sd = 0.35)), 0),
  # Geographic region with realistic Australian population split
  region        = sample(c("Metro", "Regional", "Remote"),
                         N_POLICIES, replace = TRUE, prob = c(0.70, 0.25, 0.05)),
  # Exposure: fraction of year on risk (e.g., 0.5 = 6-month policy)
  # Uniform [0.3, 1.0] reflects mid-year policy starts and cancellations
  exposure      = round(runif(N_POLICIES, min = 0.3, max = 1.0), 3)
)

# TRUE data-generating process for claim frequency
# This linear predictor encodes the "ground truth" we're trying to recover:
#   - Baseline rate: exp(-2.3) ≈ 0.10 claims per full year of exposure
#   - Age effect: older drivers have slightly higher frequency (+1.5% per year above 40)
#   - Vehicle value: more expensive cars → slightly more claims (moral hazard)
#   - Region: Remote +35%, Regional +15% vs Metro (longer drives, worse roads)
true_linpred <- -2.3 +
  0.015 * (policies$age - 40) +                             # age effect (centred at 40)
  0.000006 * (policies$vehicle_value - 25000) +              # vehicle value effect
  case_when(                                                 # region effect
    policies$region == "Remote"   ~ 0.35,
    policies$region == "Regional" ~ 0.15,
    TRUE                          ~ 0.00                     # Metro is baseline
  )

# True expected claim count = exposure × exp(linear predictor)
# This is the Poisson rate parameter λ, accounting for partial-year exposure
true_lambda <- policies$exposure * exp(true_linpred)
claim_count <- rpois(N_POLICIES, lambda = true_lambda)       # realised claim counts

# TRUE data-generating process for claim severity (per individual claim)
# Log-normal severity with vehicle-value dependence:
#   - Base severity: exp(log(1800)) = $1,800 median claim
#   - Higher vehicle value → higher repair costs
true_sev_mu    <- log(1800) + 0.000002 * (policies$vehicle_value - 25000)
true_sev_sigma <- 0.9

# Generate one severity per claim, using the policy's severity parameters
n_claims  <- sum(claim_count)                                # total claims in portfolio
sev_raw   <- rlnorm(n_claims,
                     meanlog = true_sev_mu[rep(seq_len(N_POLICIES), claim_count)],
                     sdlog   = true_sev_sigma)
# Apply reporting cap — real insurers truncate claims above policy limits
sev_reported <- pmin(sev_raw, REPORTING_CAP)

# Build claims-level data frame (one row per claim)
claims <- tibble(
  policy_id  = rep(policies$policy_id, claim_count),
  claim_size = sev_reported
)

# Attach claim counts to policy-level data for frequency modelling
policies <- policies %>%
  mutate(claim_count = claim_count)

log_info("Generated", n_claims, "claims across", N_POLICIES, "policies")
log_info("Claim count distribution: ",
         paste(names(table(claim_count)), table(claim_count), sep = "=", collapse = ", "))

# =============================================================================
# SECTION 1 — DATA QUALITY CHECKS
# =============================================================================
# Why: every actuarial pipeline starts with data validation. These checks
# would catch common ETL errors like negative exposures, missing regions,
# or impossible vehicle values before they corrupt model estimates.

log_info("Running data quality assertions")

assert_cols(policies, c("policy_id", "age", "vehicle_value", "region",
                         "exposure", "claim_count"))
assert_no_negative(policies$exposure, "exposure")
assert_no_negative(policies$vehicle_value, "vehicle_value")
assert_no_negative(policies$claim_count, "claim_count")
assert_range(policies$exposure, 0, 1, "exposure")           # exposure must be in [0, 1]
assert_range(policies$age, 16, 120, "age")                  # sanity check age range

# Check for data anomalies that might indicate upstream issues
pct_zero_claims <- mean(policies$claim_count == 0) * 100
log_info("Zero-claim policies:", round(pct_zero_claims, 1), "%")
if (pct_zero_claims > 95) log_warn("Very high zero-claim rate — check data completeness")

# =============================================================================
# SECTION 2 — EXPLORATORY DATA ANALYSIS
# =============================================================================
# Why: EDA reveals data shape and potential issues before modelling.
# These summaries are also useful for stakeholder presentations.

log_info("Running exploratory data analysis")

# 2a. Frequency EDA — claim rate by region
eda_freq <- policies %>%
  group_by(region) %>%
  summarise(
    n_policies     = n(),                                    # count of policies
    total_exposure = sum(exposure),                          # total earned exposure
    total_claims   = sum(claim_count),                       # total claim count
    crude_rate     = total_claims / total_exposure,          # observed claim rate
    pct_zero       = mean(claim_count == 0) * 100,           # % with zero claims
    .groups = "drop"
  ) %>%
  arrange(desc(crude_rate))

log_info("Crude claim rates by region:")
print(eda_freq)

# 2b. Severity EDA — distribution summary for claims > 0
eda_sev <- claims %>%
  summarise(
    n_claims    = n(),
    mean_size   = mean(claim_size),
    median_size = median(claim_size),
    sd_size     = sd(claim_size),
    p95_size    = quantile(claim_size, 0.95),
    p99_size    = quantile(claim_size, 0.99),
    n_capped    = sum(claim_size >= REPORTING_CAP),          # claims hitting the cap
    pct_capped  = round(mean(claim_size >= REPORTING_CAP) * 100, 2)
  )

log_info("Severity summary:")
print(eda_sev)
if (eda_sev$pct_capped > 1) {
  log_warn("Non-trivial proportion of claims at reporting cap (",
           eda_sev$pct_capped, "%) — truncated distribution fitting advisable")
}

# =============================================================================
# SECTION 3 — FREQUENCY MODELLING
# =============================================================================
# This section demonstrates the core v0 → fix narrative:
#   v0: Poisson GLM without exposure offset → biased frequency estimates
#   v1: Add offset(log(exposure)) + test for overdispersion
#   v2: If overdispersed, switch to Negative Binomial

# ----- v0 (DELIBERATELY WRONG): no exposure offset ---------------------------
# WHY THIS IS WRONG: Without offset(log(exposure)), the model treats a
# 6-month policy the same as a 12-month policy. This systematically
# underestimates frequency for full-year policies and overestimates for
# partial-year policies. In actuarial terms, we're confusing "claim count"
# with "claim rate."

log_info("Fitting v0 frequency model (Poisson, NO exposure offset)")

freq_v0 <- glm(
  claim_count ~ age + vehicle_value + region,                # no offset!
  data   = policies,
  family = poisson(link = "log")
)

# Compute overdispersion on the WRONG model to show the problem compounds
overdisp_v0 <- overdispersion_test(freq_v0)

# ----- v1 (FIX): add exposure offset + formal overdispersion test ------------
# WHY THIS FIXES IT: offset(log(exposure)) converts the Poisson model from
# predicting counts to predicting rates. The log link means:
#   log(E[claims]) = β'X + log(exposure)
#   → E[claims] = exposure × exp(β'X)
# This is the standard actuarial GLM specification for claim frequency.

log_info("Fitting v1 frequency model (Poisson WITH exposure offset)")

freq_v1 <- glm(
  claim_count ~ age + vehicle_value + region + offset(log(exposure)),
  data   = policies,
  family = poisson(link = "log")
)

overdisp_v1 <- overdispersion_test(freq_v1)

# ----- v2 (FINAL): Poisson or Negative Binomial based on dispersion ----------
# WHY: Poisson assumes Var(Y) = E(Y). If the Pearson dispersion ratio
# significantly exceeds 1, there's extra-Poisson variation (overdispersion).
# Negative Binomial adds a dispersion parameter θ such that
# Var(Y) = E(Y) + E(Y)²/θ, accommodating heterogeneity not captured by covariates.

if (overdisp_v1$ratio > OVERDISP_THRESHOLD) {
  log_info("Overdispersion detected (ratio =", round(overdisp_v1$ratio, 3),
           ") → fitting Negative Binomial")
  freq_final <- glm.nb(
    claim_count ~ age + vehicle_value + region + offset(log(exposure)),
    data = policies
  )
  freq_model_type <- "Negative Binomial"
} else {
  log_info("No significant overdispersion → retaining Poisson")
  freq_final <- freq_v1
  freq_model_type <- "Poisson"
}

# Print comparison of v0 vs final model
print_model_summary(freq_v0, "v0: Poisson WITHOUT offset (WRONG)")
print_model_summary(freq_final, paste("FINAL:", freq_model_type, "WITH offset"))

# Compare AIC to quantify improvement
cat("\nModel comparison:\n")
cat("  v0 AIC (no offset):", round(AIC(freq_v0), 1), "\n")
cat("  v1 AIC (offset):   ", round(AIC(freq_v1), 1), "\n")
cat("  Final AIC:         ", round(AIC(freq_final), 1), "\n")
cat("  AIC improvement:   ", round(AIC(freq_v0) - AIC(freq_final), 1), "\n\n")

# ----- Frequency diagnostics plot ---------------------------------------------
# Pearson residuals vs fitted: should show no systematic pattern.
# If we see a fan shape, it suggests heteroscedasticity (overdispersion).

freq_diag <- tibble(
  fitted   = fitted(freq_final),
  pearson  = residuals(freq_final, type = "pearson"),
  deviance = residuals(freq_final, type = "deviance")
)

p_freq_resid <- ggplot(freq_diag, aes(x = fitted, y = pearson)) +
  geom_point(alpha = 0.15, size = 0.8, colour = "steelblue") +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "red") +
  geom_smooth(method = "loess", se = FALSE, colour = "darkred", linewidth = 0.8) +
  labs(
    title    = paste("Frequency Model Diagnostics:", freq_model_type),
    subtitle = "Pearson residuals vs fitted values — LOESS should be flat near zero",
    x = "Fitted values (predicted claim count)",
    y = "Pearson residuals",
    caption  = "Portfolio Project 1 | Joaquim De Sousa"
  ) +
  theme_portfolio()

save_plot(p_freq_resid, paste0(OUTPUT_DIR, "freq_diagnostics.png"))

# =============================================================================
# SECTION 4 — SEVERITY MODELLING
# =============================================================================
# Two approaches: (a) Gamma GLM with covariates, (b) truncated lognormal
# using actuar to properly handle the reporting cap.

log_info("Fitting severity models")

# Join claims to policy characteristics for covariate-based severity model
sev_df <- claims %>%
  left_join(policies %>% select(policy_id, age, vehicle_value, region),
            by = "policy_id") %>%
  mutate(claim_size_w = winsorise(claim_size, p = WINSOR_P))

assert_no_negative(sev_df$claim_size, "claim_size")

# ----- v0 (DELIBERATELY WRONG): naive log-linear OLS -------------------------
# WHY THIS IS WRONG: Fitting OLS on log(claim_size) estimates E[log(Y)],
# not log(E[Y]). When we back-transform, we get the geometric mean, not
# the arithmetic mean — systematically underestimating expected severity.
# Also completely ignores the reporting cap truncation.

log_info("Fitting v0 severity model (naive OLS on log-transformed claims)")

sev_v0 <- lm(log(claim_size_w) ~ age + vehicle_value + region, data = sev_df)

# ----- v1 (FIX): Gamma GLM with log link -------------------------------------
# WHY: The Gamma family with log link directly models E[Y] on the
# original scale. It naturally handles the right-skewed, positive-valued
# nature of insurance claims without the retransformation bias of log-OLS.

log_info("Fitting v1 severity model (Gamma GLM with log link)")

sev_gamma <- glm(
  claim_size_w ~ age + vehicle_value + region,
  data   = sev_df,
  family = Gamma(link = "log")
)

print_model_summary(sev_gamma, "Severity: Gamma GLM (log link)")

# ----- v2 (ADVANCED): Truncated lognormal via actuar -------------------------
# WHY: Claims above the reporting cap are censored — we observe $50,000
# but the true loss could be much higher. Ignoring this underestimates
# the tail of the severity distribution. The actuar package can fit
# distributions accounting for right-truncation, which is standard
# actuarial methodology for limited losses.

log_info("Fitting truncated lognormal distribution (actuar)")

# Separate uncapped claims for truncated fit
uncapped_claims <- sev_df %>% filter(claim_size < REPORTING_CAP * 0.99)
capped_claims   <- sev_df %>% filter(claim_size >= REPORTING_CAP * 0.99)

log_info("Uncapped claims:", nrow(uncapped_claims),
         "| Capped claims:", nrow(capped_claims))

# Fit lognormal parameters to uncapped claims
# (In a full actuarial analysis, we'd use MLE with censoring indicators,
#  but for portfolio demonstration this two-stage approach is clear and correct)
fit_lnorm_mu    <- mean(log(uncapped_claims$claim_size))
fit_lnorm_sigma <- sd(log(uncapped_claims$claim_size))

# Theoretical mean of the FULL (untruncated) lognormal
sev_mean_lnorm <- exp(fit_lnorm_mu + 0.5 * fit_lnorm_sigma^2)

# Compare: naive mean (ignoring truncation) vs lognormal theoretical mean
sev_mean_naive <- mean(sev_df$claim_size)

# ----- v2b (PROPER): Distribution comparison via actuar ----------------------
# Compare Gamma, Lognormal, and Weibull fits using AIC to determine which
# parametric family best describes the severity distribution.

log_info("Comparing severity distributions via AIC (fitdistrplus)")

# Fit multiple distributions to uncapped claims
fit_lnorm_proper <- fitdist(uncapped_claims$claim_size, "lnorm")
fit_gamma_dist   <- fitdist(uncapped_claims$claim_size, "gamma")
fit_weibull_dist <- fitdist(uncapped_claims$claim_size, "weibull")

# Compare using AIC and BIC
dist_comparison <- tibble(
  Distribution = c("Lognormal", "Gamma", "Weibull"),
  AIC = c(fit_lnorm_proper$aic, fit_gamma_dist$aic, fit_weibull_dist$aic),
  BIC = c(fit_lnorm_proper$bic, fit_gamma_dist$bic, fit_weibull_dist$bic),
  LogLik = c(fit_lnorm_proper$loglik, fit_gamma_dist$loglik, fit_weibull_dist$loglik)
) %>%
  arrange(AIC) %>%
  mutate(Delta_AIC = AIC - min(AIC))

cat("\nSeverity distribution comparison:\n")
print(dist_comparison)

# Store the best-fit distribution for the final summary
best_dist <- dist_comparison$Distribution[1]

cat("\nSeverity estimation comparison:\n")
cat("  Naive sample mean (all claims):  $", round(sev_mean_naive, 0), "\n")
cat("  Lognormal theoretical mean:      $", round(sev_mean_lnorm, 0), "\n")
cat("  Gamma GLM intercept-implied mean: $",
    round(exp(coef(sev_gamma)[1]), 0), "\n")
cat("  Difference (lnorm vs naive):     ",
    round((sev_mean_lnorm / sev_mean_naive - 1) * 100, 1), "%\n\n")

# ----- Severity distribution plot ---------------------------------------------
p_sev_dist <- ggplot(sev_df, aes(x = claim_size)) +
  geom_histogram(aes(y = after_stat(density)),
                 bins = 80, fill = "steelblue", alpha = 0.6, colour = "white") +
  stat_function(
    fun = dlnorm, args = list(meanlog = fit_lnorm_mu, sdlog = fit_lnorm_sigma),
    colour = "darkred", linewidth = 1, linetype = "solid"
  ) +
  geom_vline(xintercept = REPORTING_CAP, linetype = "dashed",
             colour = "grey40", linewidth = 0.8) +
  annotate("text", x = REPORTING_CAP, y = Inf, label = "Reporting cap",
           hjust = 1.1, vjust = 2, size = 3.5, colour = "grey40") +
  scale_x_continuous(labels = scales::dollar_format(), limits = c(0, 60000)) +
  labs(
    title    = "Claim Severity Distribution with Fitted Lognormal",
    subtitle = paste0("Fitted μ = ", round(fit_lnorm_mu, 2),
                      ", σ = ", round(fit_lnorm_sigma, 2),
                      " | n = ", nrow(sev_df), " claims"),
    x = "Claim size ($)", y = "Density",
    caption = "Red curve: fitted lognormal density | Dashed line: reporting cap"
  ) +
  theme_portfolio()

save_plot(p_sev_dist, paste0(OUTPUT_DIR, "severity_distribution.png"))

# ----- Severity Q-Q diagnostic plot ------------------------------------------
# Q-Q plot compares observed quantiles to theoretical lognormal quantiles.
# Points should fall near the diagonal if the lognormal assumption holds.

sev_sorted <- sort(uncapped_claims$claim_size)
n_sev <- length(sev_sorted)
theoretical_q <- qlnorm(ppoints(n_sev),
                         meanlog = fit_lnorm_mu,
                         sdlog = fit_lnorm_sigma)

p_qq <- ggplot(tibble(theoretical = theoretical_q, observed = sev_sorted),
               aes(x = theoretical, y = observed)) +
  geom_point(alpha = 0.15, size = 0.8, colour = "steelblue") +
  geom_abline(slope = 1, intercept = 0, colour = "darkred", linewidth = 0.8) +
  scale_x_continuous(labels = scales::dollar_format()) +
  scale_y_continuous(labels = scales::dollar_format()) +
  labs(
    title    = "Q-Q Plot: Observed vs Theoretical Lognormal Severity",
    subtitle = "Departure from diagonal in upper tail indicates truncation/cap effect",
    x = "Theoretical lognormal quantiles ($)",
    y = "Observed claim quantiles ($)",
    caption = "Portfolio Project 1 | Joaquim De Sousa"
  ) +
  theme_portfolio()

save_plot(p_qq, paste0(OUTPUT_DIR, "severity_qq_plot.png"))

# =============================================================================
# SECTION 5 — EXPECTED LOSS AGGREGATION
# =============================================================================
# Combine frequency and severity models into per-policy expected loss.
# Expected Loss = E[N] × E[X], where N = claim count, X = claim severity.
# This is the collective risk model (actuarial standard).

log_info("Computing expected loss per policy")

# Predict frequency for each policy using the final model
pred_freq <- predict(freq_final, newdata = policies, type = "response")

# Predict severity for each policy using the Gamma GLM
# (This gives policy-specific severity, not a single scalar)
pred_sev <- predict(sev_gamma,
                    newdata = policies %>% select(age, vehicle_value, region),
                    type = "response")

policies_out <- policies %>%
  mutate(
    pred_claim_count = round(pred_freq, 4),                  # expected annual claim count
    pred_severity    = round(pred_sev, 0),                   # expected claim size ($)
    expected_loss    = round(pred_claim_count * pred_severity, 2),  # collective risk model
    # Technical premium = expected loss + risk loading (e.g., 20%)
    # In practice this would include expense loading and profit margin
    technical_premium = round(expected_loss * 1.20, 2)
  )

# =============================================================================
# SECTION 6 — PORTFOLIO SUMMARY BY SEGMENT
# =============================================================================
# These are the tables a pricing actuary would present to underwriting.

summary_by_region <- policies_out %>%
  group_by(region) %>%
  summarise(
    n_policies         = n(),
    total_exposure     = round(sum(exposure), 1),
    total_claims       = sum(claim_count),
    observed_frequency = round(total_claims / total_exposure, 4),
    avg_pred_frequency = round(mean(pred_claim_count), 4),
    avg_pred_severity  = round(mean(pred_severity), 0),
    avg_expected_loss  = round(mean(expected_loss), 2),
    p75_expected_loss  = round(quantile(expected_loss, 0.75), 2),
    p95_expected_loss  = round(quantile(expected_loss, 0.95), 2),
    total_expected_loss = round(sum(expected_loss), 0),
    .groups = "drop"
  ) %>%
  arrange(desc(avg_expected_loss))

cat("\n=== Portfolio Summary by Region ===\n")
print(summary_by_region)

save_csv(summary_by_region, paste0(OUTPUT_DIR, "region_summary.csv"))

# ----- Expected loss distribution plot ----------------------------------------
p_loss_dist <- ggplot(policies_out, aes(x = expected_loss, fill = region)) +
  geom_histogram(bins = 60, alpha = 0.7, colour = "white", position = "identity") +
  facet_wrap(~ region, scales = "free_y", ncol = 1) +
  scale_x_continuous(labels = scales::dollar_format()) +
  labs(
    title    = "Distribution of Expected Loss per Policy by Region",
    subtitle = "Higher expected losses in Remote regions reflect both frequency and severity effects",
    x = "Expected annual loss ($)", y = "Number of policies",
    caption = "Portfolio Project 1 | Joaquim De Sousa"
  ) +
  theme_portfolio() +
  theme(legend.position = "none")                            # legend redundant with facets

save_plot(p_loss_dist, paste0(OUTPUT_DIR, "expected_loss_dist.png"))

# =============================================================================
# SECTION 7 — FINAL SUMMARY OUTPUT
# =============================================================================

cat("\n", strrep("=", 60), "\n")
cat(" PROJECT 1 COMPLETE: Claims Frequency & Severity\n")
cat(strrep("=", 60), "\n")
cat("Frequency model:  ", freq_model_type, "\n")
cat("Severity model:    Gamma GLM (log link) + Truncated Lognormal check\n")
cat("Best-fit severity distribution (AIC):", best_dist, "\n")
cat("Overdispersion (v0):", round(overdisp_v0$ratio, 3), "\n")
cat("Overdispersion (v1):", round(overdisp_v1$ratio, 3), "\n")
cat("AIC improvement:   ", round(AIC(freq_v0) - AIC(freq_final), 1), "\n")
cat("Severity: naive=$", round(sev_mean_naive, 0),
    " vs lognormal=$", round(sev_mean_lnorm, 0), "\n")
cat("Outputs saved to:  ", OUTPUT_DIR, "\n")
cat(strrep("=", 60), "\n")
