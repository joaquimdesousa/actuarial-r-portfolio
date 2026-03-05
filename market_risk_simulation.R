# =============================================================================
# market_risk_simulation.R
# Project 3: Market Risk Simulation & Stress Testing
# =============================================================================
# Author:  Joaquim De Sousa
# Purpose: Assess portfolio risk for a diversified investment portfolio
#          using Monte Carlo simulation, compute VaR/CVaR at multiple
#          confidence levels, and run economically motivated stress tests.
#
# Narrative (v0 → failure → fix):
#   v0: Simulate asset returns independently → understates tail risk
#       because it ignores the fact that assets crash together in crises.
#   v1: Use Cholesky decomposition to generate correlated returns from
#       the empirical correlation matrix → captures co-movement.
#   v2: Apply GARCH(1,1) to model volatility clustering (returns are
#       not i.i.d. — volatility tends to persist), then layer on
#       economically motivated stress scenarios.
#
# Key packages: tidyverse, PerformanceAnalytics (risk metrics),
#               rugarch (GARCH modelling)
#
# Outputs (saved to outputs/market/):
#   - risk_summary.csv          : VaR/CVaR comparison across scenarios
#   - var_comparison.png        : VaR/CVaR bar chart by scenario
#   - return_distributions.png  : portfolio return distributions
#   - correlation_heatmap.png   : asset correlation matrix
#   - garch_volatility.png      : GARCH conditional volatility over time
# =============================================================================

# --- Load dependencies --------------------------------------------------------
suppressPackageStartupMessages({
  library(tidyverse)
  library(PerformanceAnalytics)          # VaR, CVaR, chart functions
})

# Attempt to load rugarch for GARCH modelling; if unavailable, skip GARCH
has_rugarch <- requireNamespace("rugarch", quietly = TRUE)
if (has_rugarch) {
  library(rugarch)
} else {
  message("Note: rugarch not installed — GARCH section will use simple EWMA volatility")
}

source("utils.R")

# --- Configuration ------------------------------------------------------------
OUTPUT_DIR   <- "outputs/market/"
SEED         <- 456
N_SIM        <- 50000                    # Monte Carlo simulations (professional level)
T_HIST       <- 2000                     # historical trading days (~8 years)
CONF_LEVELS  <- c(0.95, 0.99, 0.995)    # VaR confidence levels
HORIZON_DAYS <- c(1, 10)                 # 1-day (internal) and 10-day (Basel)

set.seed(SEED)
ensure_dir(OUTPUT_DIR)

log_info("=== Project 3: Market Risk Simulation & Stress Testing ===")

# =============================================================================
# SECTION 0 — DATA SIMULATION
# =============================================================================
# Simulate daily returns for a 3-asset portfolio. In production this would
# come from market data APIs (quantmod / Bloomberg), but simulation lets
# us control the true correlation structure for verification.

log_info("Simulating", T_HIST, "days of historical returns for 3 assets")

assets <- c("Equity", "Bonds", "Commodities")

# Expected daily returns (annualised: Equity ~7.5%, Bonds ~3.8%, Commodities ~5%)
mu <- c(0.0003, 0.00015, 0.0002)

# Daily volatilities (annualised: Equity ~19%, Bonds ~6.3%, Commodities ~14.3%)
sigma <- c(0.012, 0.004, 0.009)

# Correlation matrix — reflects realistic asset relationships:
#   Equity-Bonds: -0.25 (flight to safety during equity downturns)
#   Equity-Commodities: +0.20 (both driven by economic growth)
#   Bonds-Commodities: -0.10 (inflation expectations)
R_true <- matrix(c(
   1.00, -0.25,  0.20,
  -0.25,  1.00, -0.10,
   0.20, -0.10,  1.00
), nrow = 3, byrow = TRUE)
colnames(R_true) <- rownames(R_true) <- assets

# Verify correlation matrix is positive definite (required for Cholesky)
eigen_vals <- eigen(R_true, only.values = TRUE)$values
stopifnot(all(eigen_vals > 0))                               # must be positive definite
log_info("Correlation matrix eigenvalues:", paste(round(eigen_vals, 4), collapse = ", "))

# Generate correlated returns via Cholesky decomposition
# Method: Z ~ N(0,I) → L'Z ~ N(0,R) where R = LL' (Cholesky factorisation)
L_chol <- chol(R_true)                                       # upper triangular Cholesky factor
Z_hist <- matrix(rnorm(T_HIST * 3), ncol = 3)               # independent standard normals
corr_Z <- Z_hist %*% L_chol                                 # apply correlation structure

# Scale to correct mean and volatility: r = μ + σ * z
hist_returns <- sweep(corr_Z, 2, sigma, `*`)                # scale by volatility
hist_returns <- sweep(hist_returns, 2, mu, `+`)             # shift by mean
colnames(hist_returns) <- assets

# Convert to data frame for analysis
returns_df <- as.data.frame(hist_returns) %>%
  mutate(day = row_number())

# Portfolio weights (60/30/10 — typical balanced allocation)
weights <- c(Equity = 0.60, Bonds = 0.30, Commodities = 0.10)

# Historical portfolio returns
returns_df$portfolio <- as.numeric(as.matrix(returns_df[, assets]) %*% weights)

log_info("Historical return stats:")
log_info("  Mean daily portfolio return:", round(mean(returns_df$portfolio), 6))
log_info("  Std dev daily portfolio return:", round(sd(returns_df$portfolio), 6))

# =============================================================================
# SECTION 1 — v0 (WRONG): Independent Simulation
# =============================================================================
# WHY THIS IS WRONG: Simulating each asset independently ignores the
# correlation structure. During market stress, correlations spike —
# equity and commodities drop together, bonds rally. Independent
# simulation underestimates how bad a joint drawdown can be because
# it allows unrealistically good diversification benefits.

log_info("v0: Running Monte Carlo with INDEPENDENT asset returns")

Z_indep <- matrix(rnorm(N_SIM * 3), ncol = 3)               # independent normals
rets_indep <- sweep(Z_indep, 2, sigma, `*`)                 # scale by vol
rets_indep <- sweep(rets_indep, 2, mu, `+`)                 # shift by mean
port_indep <- as.numeric(rets_indep %*% weights)             # portfolio return

# =============================================================================
# SECTION 2 — v1 (FIX): Correlated Simulation
# =============================================================================
# WHY: Cholesky decomposition of the correlation matrix preserves the
# co-movement structure. When equity drops, bonds tend to rise (negative
# correlation), providing realistic diversification. But commodities also
# drop with equity (positive correlation), concentrating losses.

log_info("v1: Running Monte Carlo with CORRELATED asset returns")

Z_corr <- matrix(rnorm(N_SIM * 3), ncol = 3) %*% L_chol    # correlated via Cholesky
rets_corr <- sweep(Z_corr, 2, sigma, `*`)
rets_corr <- sweep(rets_corr, 2, mu, `+`)
port_corr <- as.numeric(rets_corr %*% weights)

# =============================================================================
# SECTION 3 — v2 (ADVANCED): GARCH-Based Volatility
# =============================================================================
# WHY: Real financial returns exhibit volatility clustering — large moves
# tend to follow large moves. GARCH(1,1) captures this by modelling
# conditional variance as a function of past shocks and past variance.
# This produces fatter tails and more realistic risk estimates than
# assuming constant volatility.

if (has_rugarch) {
  log_info("v2: Fitting GARCH(1,1) to equity returns for conditional volatility")

  # Fit GARCH(1,1) to equity returns (the dominant risk driver)
  garch_spec <- ugarchspec(
    variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
    mean.model     = list(armaOrder = c(0, 0), include.mean = TRUE),
    distribution.model = "std"                               # Student-t for fat tails
  )

  garch_fit <- ugarchfit(spec = garch_spec, data = returns_df$Equity)

  # Extract conditional volatility (time-varying σ)
  cond_vol <- sigma(garch_fit)

  # Use the last-day conditional volatility for forward simulation
  # (reflects current market regime)
  garch_vol_equity <- as.numeric(tail(cond_vol, 1))

  log_info("GARCH estimated current equity vol:",
           round(garch_vol_equity, 6), "vs unconditional:", round(sigma[1], 6))

  # Scale simulation volatility by GARCH ratio
  vol_scale <- garch_vol_equity / sigma[1]                   # ratio vs unconditional
  sigma_garch <- sigma
  sigma_garch[1] <- garch_vol_equity                         # update equity vol

  # Re-simulate with GARCH-adjusted volatility
  Z_garch <- matrix(rnorm(N_SIM * 3), ncol = 3) %*% L_chol
  rets_garch <- sweep(Z_garch, 2, sigma_garch, `*`)
  rets_garch <- sweep(rets_garch, 2, mu, `+`)
  port_garch <- as.numeric(rets_garch %*% weights)

  # Volatility plot
  vol_df <- tibble(day = seq_len(T_HIST), cond_vol = as.numeric(cond_vol))
  p_vol <- ggplot(vol_df, aes(x = day, y = cond_vol)) +
    geom_line(colour = "steelblue", linewidth = 0.4) +
    geom_hline(yintercept = sigma[1], linetype = "dashed", colour = "red") +
    annotate("text", x = T_HIST * 0.02, y = sigma[1] * 1.08,
             label = "Unconditional vol", colour = "red", hjust = 0, size = 3) +
    labs(
      title    = "GARCH(1,1) Conditional Volatility — Equity",
      subtitle = "Volatility clustering: periods of high vol persist",
      x = "Trading day", y = "Conditional daily volatility",
      caption = "Portfolio Project 3 | Joaquim De Sousa"
    ) +
    theme_portfolio()

  save_plot(p_vol, paste0(OUTPUT_DIR, "garch_volatility.png"))

} else {
  log_warn("rugarch not available — using EWMA volatility estimate instead")

  # EWMA (Exponentially Weighted Moving Average) as fallback
  # λ = 0.94 is the RiskMetrics standard
  ewma_lambda <- 0.94
  ewma_var <- numeric(T_HIST)
  ewma_var[1] <- var(returns_df$Equity)
  for (t in 2:T_HIST) {
    ewma_var[t] <- ewma_lambda * ewma_var[t-1] +
                   (1 - ewma_lambda) * returns_df$Equity[t-1]^2
  }
  garch_vol_equity <- sqrt(tail(ewma_var, 1))

  sigma_garch <- sigma
  sigma_garch[1] <- garch_vol_equity

  Z_garch <- matrix(rnorm(N_SIM * 3), ncol = 3) %*% L_chol
  rets_garch <- sweep(Z_garch, 2, sigma_garch, `*`)
  rets_garch <- sweep(rets_garch, 2, mu, `+`)
  port_garch <- as.numeric(rets_garch %*% weights)
}

# =============================================================================
# SECTION 4 — VaR AND CVaR COMPUTATION
# =============================================================================
# VaR (Value-at-Risk): the loss threshold not exceeded with probability α.
#   "We are α% confident the portfolio won't lose more than VaR in one day."
# CVaR (Conditional VaR / Expected Shortfall): the expected loss given
#   that we're in the worst (1-α) tail. CVaR is always >= VaR and is
#   a coherent risk measure (satisfies subadditivity, unlike VaR).

log_info("Computing VaR and CVaR at multiple confidence levels")

#' Compute VaR and CVaR from a return vector
#' @param returns Numeric vector of portfolio returns
#' @param alpha   Confidence level (e.g., 0.99)
#' @return Named list with VaR and CVaR (both positive for losses)
compute_var_cvar <- function(returns, alpha) {
  threshold <- quantile(returns, 1 - alpha)                  # (1-α) quantile
  var_val   <- -as.numeric(threshold)                        # negate: loss is positive
  cvar_val  <- -mean(returns[returns <= threshold])          # expected loss in tail
  list(VaR = var_val, CVaR = cvar_val)
}

# Build comprehensive risk summary across all scenarios and confidence levels
risk_results <- list()

for (alpha in CONF_LEVELS) {
  # Independent (v0)
  rc_indep <- compute_var_cvar(port_indep, alpha)
  risk_results <- c(risk_results, list(tibble(
    scenario = "v0: Independent", confidence = alpha,
    VaR = rc_indep$VaR, CVaR = rc_indep$CVaR)))

  # Correlated (v1)
  rc_corr <- compute_var_cvar(port_corr, alpha)
  risk_results <- c(risk_results, list(tibble(
    scenario = "v1: Correlated", confidence = alpha,
    VaR = rc_corr$VaR, CVaR = rc_corr$CVaR)))

  # GARCH-adjusted (v2)
  rc_garch <- compute_var_cvar(port_garch, alpha)
  risk_results <- c(risk_results, list(tibble(
    scenario = "v2: GARCH-adjusted", confidence = alpha,
    VaR = rc_garch$VaR, CVaR = rc_garch$CVaR)))
}

risk_summary <- bind_rows(risk_results) %>%
  mutate(
    VaR  = round(VaR, 6),
    CVaR = round(CVaR, 6),
    # Scale to 10-day horizon using square-root-of-time rule
    # (standard under Basel framework for market risk capital)
    VaR_10d  = round(VaR * sqrt(10), 6),
    CVaR_10d = round(CVaR * sqrt(10), 6)
  )

cat("\n=== Risk Summary: VaR and CVaR ===\n")
print(risk_summary)

# =============================================================================
# SECTION 5 — STRESS TESTING
# =============================================================================
# Three economically motivated stress scenarios based on historical events.
# Each scenario adjusts the simulation parameters (mean and volatility)
# to reflect what happened during that crisis.

log_info("Running stress test scenarios")

stress_scenarios <- list(
  # Scenario 1: 2008 GFC-style equity crash
  # Equity: mean drops sharply, volatility doubles
  # Bonds: mean rises (flight to safety), vol increases modestly
  # Commodities: mean drops, vol increases
  "GFC_2008" = list(
    mu_shift  = c(-0.003, +0.001, -0.002),                  # daily mean shifts
    vol_scale = c(2.5, 1.5, 2.0)                            # volatility multipliers
  ),

  # Scenario 2: 2020 COVID crash
  # Faster, sharper drawdown; bonds initially sell off too (liquidity crisis)
  "COVID_2020" = list(
    mu_shift  = c(-0.004, -0.0005, -0.003),
    vol_scale = c(3.0, 2.0, 2.5)
  ),

  # Scenario 3: Stagflation (1970s-style)
  # Equity stagnates, bonds sell off (rising rates), commodities rally
  "Stagflation" = list(
    mu_shift  = c(-0.001, -0.002, +0.001),
    vol_scale = c(1.5, 2.0, 1.8)
  )
)

stress_results <- list()

for (scenario_name in names(stress_scenarios)) {
  sc <- stress_scenarios[[scenario_name]]

  # Adjust parameters
  mu_stressed    <- mu + sc$mu_shift
  sigma_stressed <- sigma * sc$vol_scale

  # Simulate with stressed parameters (still correlated)
  Z_stress <- matrix(rnorm(N_SIM * 3), ncol = 3) %*% L_chol
  rets_stress <- sweep(Z_stress, 2, sigma_stressed, `*`)
  rets_stress <- sweep(rets_stress, 2, mu_stressed, `+`)
  port_stress <- as.numeric(rets_stress %*% weights)

  # Compute risk metrics
  for (alpha in CONF_LEVELS) {
    rc <- compute_var_cvar(port_stress, alpha)
    stress_results <- c(stress_results, list(tibble(
      scenario   = paste0("Stress: ", scenario_name),
      confidence = alpha,
      VaR = round(rc$VaR, 6), CVaR = round(rc$CVaR, 6),
      VaR_10d = round(rc$VaR * sqrt(10), 6),
      CVaR_10d = round(rc$CVaR * sqrt(10), 6)
    )))
  }
}

stress_summary <- bind_rows(stress_results)

# Combine baseline and stress results
full_risk_summary <- bind_rows(risk_summary, stress_summary)

cat("\n=== Full Risk Summary (Including Stress Tests) ===\n")
print(full_risk_summary)

save_csv(full_risk_summary, paste0(OUTPUT_DIR, "risk_summary.csv"))

# =============================================================================
# SECTION 6 — COMPONENT VaR (Risk Attribution)
# =============================================================================
# Component VaR decomposes portfolio VaR into contributions from each asset.
# This tells the portfolio manager WHERE the risk is coming from.
# Component VaR = weight_i × σ_i × ρ(r_i, r_portfolio) × z_α / σ_portfolio

log_info("Computing component VaR (risk attribution)")

# Use correlated simulation for attribution
port_sd <- sd(port_corr)

comp_var_99 <- sapply(seq_along(assets), function(i) {
  asset_ret <- rets_corr[, i]
  rho_ip    <- cor(asset_ret, port_corr)                     # asset-portfolio correlation
  weights[i] * sigma[i] * rho_ip * qnorm(0.99) / port_sd * (-quantile(port_corr, 0.01))
})
names(comp_var_99) <- assets

comp_var_pct <- round(comp_var_99 / sum(comp_var_99) * 100, 1)

cat("\n=== Component VaR (99%) — Risk Attribution ===\n")
for (i in seq_along(assets)) {
  cat(sprintf("  %-15s %6.1f%%\n", assets[i], comp_var_pct[i]))
}

# =============================================================================
# SECTION 6b — VaR BACKTESTING (Kupiec Test)
# =============================================================================
# WHY: Computing VaR without backtesting is like building a model without
# validation. The Kupiec (1995) test checks whether the observed number of
# VaR breaches is consistent with the expected number under the model.
# Basel regulations require banks to backtest their VaR models.

log_info("Running Kupiec backtest on historical returns")

#' Kupiec Proportion of Failures (POF) test for VaR backtesting
#' @param returns Numeric vector of historical portfolio returns
#' @param var_estimate Scalar VaR estimate (positive number = loss)
#' @param alpha Confidence level (e.g., 0.99)
#' @return Named list with n_breaches, expected_breaches, breach_rate,
#'         lr_statistic (likelihood ratio), p_value, and verdict
kupiec_test <- function(returns, var_estimate, alpha) {
  n <- length(returns)
  n_breaches <- sum(returns < -var_estimate)  # count days loss exceeds VaR
  expected_rate <- 1 - alpha
  expected_breaches <- n * expected_rate
  observed_rate <- n_breaches / n

  # Likelihood ratio test statistic
  # LR = -2 * ln[ (1-p)^(n-x) * p^x / (1-x/n)^(n-x) * (x/n)^x ]
  # where p = expected rate, x = breaches, n = observations
  p0 <- expected_rate
  p1 <- max(observed_rate, 1e-10)  # avoid log(0)

  if (n_breaches == 0) {
    lr_stat <- -2 * (n * log(1 - p0) - n * log(1))
  } else if (n_breaches == n) {
    lr_stat <- -2 * (n * log(p0) - n * log(1))
  } else {
    lr_stat <- -2 * ((n - n_breaches) * log(1 - p0) + n_breaches * log(p0) -
                      (n - n_breaches) * log(1 - p1) - n_breaches * log(p1))
  }

  p_value <- 1 - pchisq(lr_stat, df = 1)

  verdict <- case_when(
    p_value > 0.05 ~ "PASS: VaR model is adequate (fail to reject H0)",
    p_value > 0.01 ~ "WARNING: Marginal — model may need recalibration",
    TRUE            ~ "FAIL: VaR model is rejected — systematic under/overestimation"
  )

  list(
    n_obs = n, n_breaches = n_breaches,
    expected_breaches = round(expected_breaches, 1),
    breach_rate = round(observed_rate, 4),
    expected_rate = expected_rate,
    lr_statistic = round(lr_stat, 4),
    p_value = round(p_value, 4),
    verdict = verdict
  )
}

# Backtest using historical portfolio returns and correlated VaR estimate
var_99_corr <- compute_var_cvar(port_corr, 0.99)$VaR

backtest_result <- kupiec_test(returns_df$portfolio, var_99_corr, alpha = 0.99)

cat("\n=== Kupiec VaR Backtest (99% confidence) ===\n")
cat("  Observations:       ", backtest_result$n_obs, "\n")
cat("  VaR breaches:       ", backtest_result$n_breaches,
    "(expected:", backtest_result$expected_breaches, ")\n")
cat("  Breach rate:        ", backtest_result$breach_rate,
    "(expected:", backtest_result$expected_rate, ")\n")
cat("  LR statistic:       ", backtest_result$lr_statistic, "\n")
cat("  p-value:            ", backtest_result$p_value, "\n")
cat("  Verdict:            ", backtest_result$verdict, "\n\n")

# =============================================================================
# SECTION 6c — COPULA FRAMEWORK NOTE
# =============================================================================
# The Cholesky decomposition used in this simulation implicitly assumes a
# GAUSSIAN COPULA — all dependence is captured by the linear correlation
# matrix. In reality, assets exhibit tail dependence: correlations spike
# during crises (financial contagion). A more advanced approach would use:
#   - t-copula (captures symmetric tail dependence)
#   - Clayton copula (captures lower tail dependence — crashes)
#   - Gumbel copula (captures upper tail dependence)
#
# Below we demonstrate awareness by computing empirical tail dependence
# and comparing to what the Gaussian copula would predict.

log_info("Computing empirical vs Gaussian tail dependence coefficients")

# Empirical lower tail dependence: P(Commodities < q | Equity < q) vs Gaussian
q_threshold <- 0.05  # look at worst 5% of days

equity_below_q <- returns_df$Equity <= quantile(returns_df$Equity, q_threshold)
commod_below_q <- returns_df$Commodities <= quantile(returns_df$Commodities, q_threshold)

# Empirical conditional probability (tail dependence proxy)
empirical_tail_dep <- mean(commod_below_q[equity_below_q])

# Under Gaussian copula, lower tail dependence coefficient λ_L = 0
# (Gaussian copula has zero tail dependence asymptotically)
# But for finite samples, we can compute the theoretical conditional prob
gaussian_tail_dep <- q_threshold  # under independence
# Under Gaussian copula with ρ = 0.20:
rho_eq_com <- R_true["Equity", "Commodities"]
gaussian_conditional <- pnorm(
  (qnorm(q_threshold) - rho_eq_com * qnorm(q_threshold)) / sqrt(1 - rho_eq_com^2)
)

cat("=== Tail Dependence Analysis (Equity-Commodities) ===\n")
cat("  Empirical P(Commod < Q5 | Equity < Q5):  ", round(empirical_tail_dep, 4), "\n")
cat("  Gaussian copula prediction:               ", round(gaussian_conditional, 4), "\n")
cat("  Ratio (>1 suggests excess tail dependence):",
    round(empirical_tail_dep / gaussian_conditional, 2), "\n")
cat("  Note: Real market data typically shows ratio > 1,\n")
cat("        motivating t-copula or Clayton copula for risk modelling.\n\n")

# =============================================================================
# SECTION 7 — VISUALISATIONS
# =============================================================================

# ----- Correlation heatmap ----------------------------------------------------
cor_empirical <- cor(returns_df[, assets])

cor_long <- as.data.frame(as.table(cor_empirical)) %>%
  rename(Asset1 = Var1, Asset2 = Var2, Correlation = Freq)

p_corr <- ggplot(cor_long, aes(x = Asset1, y = Asset2, fill = Correlation)) +
  geom_tile(colour = "white", linewidth = 1) +
  geom_text(aes(label = round(Correlation, 2)), size = 4.5, fontface = "bold") +
  scale_fill_gradient2(low = "#B2182B", mid = "white", high = "#2166AC",
                       midpoint = 0, limits = c(-1, 1)) +
  labs(
    title = "Asset Correlation Matrix (Empirical from Simulated History)",
    subtitle = "Negative Equity-Bonds correlation provides diversification benefit",
    caption = "Portfolio Project 3 | Joaquim De Sousa"
  ) +
  theme_portfolio() +
  theme(axis.title = element_blank())

save_plot(p_corr, paste0(OUTPUT_DIR, "correlation_heatmap.png"))

# ----- VaR comparison bar chart -----------------------------------------------
var_compare <- full_risk_summary %>%
  filter(confidence == 0.99) %>%
  select(scenario, VaR, CVaR) %>%
  pivot_longer(cols = c(VaR, CVaR), names_to = "metric", values_to = "value")

p_var <- ggplot(var_compare, aes(x = reorder(scenario, value), y = value, fill = metric)) +
  geom_col(position = "dodge", alpha = 0.85, width = 0.7) +
  scale_fill_manual(values = c("VaR" = "steelblue", "CVaR" = "darkred")) +
  coord_flip() +
  scale_y_continuous(labels = scales::percent_format(accuracy = 0.01)) +
  labs(
    title    = "VaR and CVaR Comparison at 99% Confidence",
    subtitle = "Independence assumption understates risk; stress scenarios reveal tail vulnerability",
    x = NULL, y = "Daily loss (as % of portfolio)",
    fill = "Metric",
    caption = "Portfolio Project 3 | Joaquim De Sousa"
  ) +
  theme_portfolio()

save_plot(p_var, paste0(OUTPUT_DIR, "var_comparison.png"))

# ----- Portfolio return distributions -----------------------------------------
dist_data <- bind_rows(
  tibble(returns = port_indep, scenario = "v0: Independent"),
  tibble(returns = port_corr,  scenario = "v1: Correlated"),
  tibble(returns = port_garch, scenario = "v2: GARCH-adjusted")
)

p_dist <- ggplot(dist_data, aes(x = returns, fill = scenario)) +
  geom_density(alpha = 0.4, colour = NA) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50") +
  scale_x_continuous(labels = scales::percent_format(accuracy = 0.1)) +
  labs(
    title    = "Portfolio Return Distributions: v0 vs v1 vs v2",
    subtitle = "Correlated and GARCH models produce fatter left tails (more realistic)",
    x = "Daily portfolio return", y = "Density", fill = "Scenario",
    caption = "Portfolio Project 3 | Joaquim De Sousa"
  ) +
  theme_portfolio()

save_plot(p_dist, paste0(OUTPUT_DIR, "return_distributions.png"))

# =============================================================================
# SECTION 8 — FINAL SUMMARY
# =============================================================================

cat("\n", strrep("=", 60), "\n")
cat(" PROJECT 3 COMPLETE: Market Risk Simulation\n")
cat(strrep("=", 60), "\n")
cat("Simulations:       ", format(N_SIM, big.mark = ","), "\n")
cat("Assets:            ", paste(assets, collapse = ", "), "\n")
cat("Weights:           ", paste(weights, collapse = "/"), "\n")
cat("Confidence levels: ", paste(CONF_LEVELS, collapse = ", "), "\n")
cat("Stress scenarios:  ", paste(names(stress_scenarios), collapse = ", "), "\n")
cat("GARCH available:   ", has_rugarch, "\n")
cat("Backtest verdict:  ", backtest_result$verdict, "\n")
cat("Copula analysis:   Gaussian copula vs empirical tail dependence computed\n")
cat("Outputs saved to:  ", OUTPUT_DIR, "\n")
cat(strrep("=", 60), "\n")
