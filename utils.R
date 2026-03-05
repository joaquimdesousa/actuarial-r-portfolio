# =============================================================================
# utils.R — Shared Helper Functions for Finance & Risk Analytics Portfolio
# =============================================================================
# Author:  Joaquim De Sousa
# Purpose: Centralised utility functions used across all three portfolio
#          projects (claims modelling, credit risk, market risk). Provides
#          data validation, statistical helpers, model diagnostics, logging,
#          and output management so every project stays DRY and auditable.
#
# Design philosophy:
#   - Every public function has roxygen-style docs (param / return / examples)
#   - Defensive programming: fail fast with clear error messages
#   - Functions are pure where possible (no hidden side effects)
#   - Logging is opt-in via `log_info()` / `log_warn()` for traceability
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
})

# =============================================================================
# SECTION 1 — LOGGING
# =============================================================================
# Why: reproducible analysis needs a paper trail. These lightweight wrappers
# timestamp every message so we can trace what happened and when during a run.

#' Log an informational message with timestamp
#' @param ... Character strings to concatenate into the message
#' @return Invisible NULL; prints to stderr so stdout stays clean for output
#' @examples log_info("Model fitting started for", model_type)
log_info <- function(...) {
  msg <- paste("[INFO", format(Sys.time(), "%H:%M:%S"), "]", ...)
  message(msg)                # message() writes to stderr, not stdout
  invisible(NULL)
}

#' Log a warning message with timestamp
#' @param ... Character strings to concatenate into the warning
#' @return Invisible NULL; prints to stderr
#' @examples log_warn("Overdispersion detected:", round(ratio, 2))
log_warn <- function(...) {
  msg <- paste("[WARN", format(Sys.time(), "%H:%M:%S"), "]", ...)
  message(msg)
  invisible(NULL)
}

# =============================================================================
# SECTION 2 — DATA VALIDATION (Defensive Programming)
# =============================================================================
# Why: garbage-in-garbage-out is the #1 cause of wrong model results.
# These assertions run at the start of every pipeline to catch data issues
# before they propagate into fitted models. Each one gives a specific,
# actionable error message.

#' Assert that a data frame contains all required columns
#' @param df    A data.frame or tibble to check
#' @param cols  Character vector of required column names
#' @return Invisible TRUE if all columns exist; stops with error otherwise
#' @examples assert_cols(policies, c("policy_id", "age", "exposure"))
assert_cols <- function(df, cols) {
  stopifnot(is.data.frame(df))                             # guard: must be a data frame
  missing <- setdiff(cols, names(df))                       # which required cols are absent?
  if (length(missing) > 0) {
    stop("Missing required columns: ", paste(missing, collapse = ", "),
         "\nAvailable columns: ", paste(names(df), collapse = ", "))
  }
  invisible(TRUE)
}

#' Assert no negative values in a numeric vector
#' @param x    Numeric vector to check
#' @param name Human-readable name for error messages (e.g. "exposure")
#' @return Invisible TRUE if no negatives; stops with error otherwise
#' @examples assert_no_negative(policies$exposure, "exposure")
assert_no_negative <- function(x, name = "value") {
  n_neg <- sum(x < 0, na.rm = TRUE)                        # count negatives, ignoring NA
 if (n_neg > 0) {
    stop(n_neg, " negative value(s) detected in '", name,
         "'. Min = ", min(x, na.rm = TRUE),
         ". Check data cleaning upstream.")
  }
  invisible(TRUE)
}

#' Assert no missing values in specified columns
#' @param df   Data frame to check
#' @param cols Character vector of column names that must be complete
#' @return Invisible TRUE; stops if any column has NAs
#' @examples assert_no_na(loans, c("income", "dti", "default"))
assert_no_na <- function(df, cols) {
  for (col in cols) {
    n_na <- sum(is.na(df[[col]]))                           # count NAs per column
    if (n_na > 0) {
      stop("Column '", col, "' has ", n_na, " missing value(s) (",
           round(100 * n_na / nrow(df), 1), "% of rows). ",
           "Impute or filter before modelling.")
    }
  }
  invisible(TRUE)
}

#' Assert a vector falls within an expected numeric range
#' @param x    Numeric vector
#' @param lo   Minimum allowed value (inclusive)
#' @param hi   Maximum allowed value (inclusive)
#' @param name Human-readable label for error messages
#' @return Invisible TRUE; stops if any values fall outside [lo, hi]
#' @examples assert_range(loans$dti, 0, 1, "debt-to-income ratio")
assert_range <- function(x, lo, hi, name = "value") {
  out_of_range <- sum(x < lo | x > hi, na.rm = TRUE)
  if (out_of_range > 0) {
    stop(out_of_range, " value(s) in '", name, "' outside [", lo, ", ", hi, "]. ",
         "Observed range: [", round(min(x, na.rm = TRUE), 4), ", ",
         round(max(x, na.rm = TRUE), 4), "]")
  }
  invisible(TRUE)
}

# =============================================================================
# SECTION 3 — STATISTICAL HELPERS
# =============================================================================
# Why: these are the building-block calculations that appear repeatedly
# across claims, credit, and market risk projects. Centralising them avoids
# copy-paste errors and ensures consistent definitions.

#' Winsorise a numeric vector at specified percentiles
#'
#' Replaces values below the p-th percentile with that percentile's value
#' and values above the (1-p)-th percentile likewise. This reduces the
#' influence of extreme outliers without dropping observations entirely.
#'
#' @param x Numeric vector (NAs are preserved)
#' @param p Lower tail probability for clipping (default 0.01 = 1st percentile)
#' @return Numeric vector of same length with extremes clipped
#' @examples
#'   winsorise(c(1, 2, 3, 100, NA), p = 0.25)
#'   # Clips 1 up to Q25 and 100 down to Q75; NA stays NA
winsorise <- function(x, p = 0.01) {
  stopifnot(is.numeric(x), p > 0, p < 0.5)                 # guard: p must be in (0, 0.5)
  q <- quantile(x, probs = c(p, 1 - p), na.rm = TRUE)      # compute clipping thresholds
  pmin(pmax(x, q[1]), q[2])                                 # clip both tails; NAs pass through
}

#' Root Mean Squared Error
#' @param y    Numeric vector of observed values
#' @param yhat Numeric vector of predicted values (same length as y)
#' @return Scalar RMSE value
#' @examples rmse(c(1, 2, 3), c(1.1, 2.2, 2.8))
rmse <- function(y, yhat) {
  stopifnot(length(y) == length(yhat))                      # guard: vectors must match
 sqrt(mean((y - yhat)^2, na.rm = TRUE))
}

#' Mean Absolute Error
#' @param y    Numeric vector of observed values
#' @param yhat Numeric vector of predicted values
#' @return Scalar MAE value
mae <- function(y, yhat) {
  stopifnot(length(y) == length(yhat))
  mean(abs(y - yhat), na.rm = TRUE)
}

#' Gini coefficient from predicted probabilities and binary outcome
#'
#' Gini = 2 * AUC - 1. A standard discrimination metric in credit scoring
#' that ranges from 0 (random model) to 1 (perfect model). Used because
#' regulators and credit committees are more familiar with Gini than raw AUC.
#'
#' @param actual    Binary 0/1 outcome vector
#' @param predicted Predicted probability vector
#' @return Scalar Gini coefficient
#' @examples gini_coefficient(c(0,0,1,1), c(0.1, 0.3, 0.8, 0.9))
gini_coefficient <- function(actual, predicted) {
  stopifnot(all(actual %in% c(0, 1)))                       # guard: must be binary
  # Rank-based Gini (avoids dependency on pROC for this single calculation)
  n <- length(actual)
  ord <- order(predicted, decreasing = TRUE)                 # sort by predicted desc
  actual_sorted <- actual[ord]
  cum_actual <- cumsum(actual_sorted) / sum(actual_sorted)   # cumulative % of positives
  cum_pop <- seq_len(n) / n                                  # cumulative % of population
  gini_val <- sum(cum_actual[-1] * diff(cum_pop)) -          # area under Lorenz curve
              sum(cum_pop[-1] * diff(cum_actual))
  gini_val                                                   # already in [0, 1] range
}

#' Kolmogorov-Smirnov statistic for binary classification
#'
#' Maximum separation between cumulative distribution of predicted
#' probabilities for positive vs negative cases. Used in credit risk
#' model validation alongside AUC and Gini.
#'
#' @param actual    Binary 0/1 outcome vector
#' @param predicted Predicted probability vector
#' @return Scalar KS statistic in [0, 1]
ks_statistic <- function(actual, predicted) {
  stopifnot(all(actual %in% c(0, 1)))
  # Empirical CDFs for each class
  pred_pos <- predicted[actual == 1]                         # predictions where default = 1
  pred_neg <- predicted[actual == 0]                         # predictions where default = 0
  # KS = max |F_pos(x) - F_neg(x)| over all thresholds
  all_vals <- sort(unique(c(pred_pos, pred_neg)))
  cdf_pos <- ecdf(pred_pos)(all_vals)                        # CDF of defaults
  cdf_neg <- ecdf(pred_neg)(all_vals)                        # CDF of non-defaults
  max(abs(cdf_pos - cdf_neg))                                # max separation
}

#' Brier score for probability calibration
#'
#' Mean squared difference between predicted probabilities and binary
#' outcomes. Ranges from 0 (perfect) to 1 (worst). Decomposes into
#' calibration + discrimination + uncertainty components.
#'
#' @param actual    Binary 0/1 outcome vector
#' @param predicted Predicted probability vector in [0, 1]
#' @return Scalar Brier score
brier_score <- function(actual, predicted) {
  stopifnot(all(actual %in% c(0, 1)))
  stopifnot(all(predicted >= 0 & predicted <= 1))            # probabilities must be valid
  mean((predicted - actual)^2)
}

#' Population Stability Index for model monitoring
#'
#' PSI measures the distributional shift between expected (baseline) and
#' actual (monitoring period) variables. Used to detect model drift and
#' trigger retraining decisions. Formula:
#' PSI = Σ (actual_pct - expected_pct) × ln(actual_pct / expected_pct)
#'
#' Interpretation:
#'   - PSI < 0.1:      Stable distribution, no action required
#'   - PSI 0.1-0.25:   Moderate shift, investigate before using model
#'   - PSI > 0.25:     Significant shift, recommend model retraining
#'
#' @param expected  Numeric vector of expected/baseline values
#' @param actual    Numeric vector of actual/monitoring period values (same length)
#' @param n_bins    Number of quantile bins for continuous data (default 10)
#' @return Scalar PSI value (non-negative)
#' @examples
#'   expected <- rnorm(1000, mean = 5, sd = 2)
#'   actual   <- rnorm(1000, mean = 5.5, sd = 2)  # slight shift
#'   population_stability_index(expected, actual, n_bins = 10)
population_stability_index <- function(expected, actual, n_bins = 10) {
  stopifnot(is.numeric(expected), is.numeric(actual))        # guard: must be numeric
  stopifnot(length(expected) == length(actual))              # guard: same length
  stopifnot(n_bins > 1)                                       # guard: need at least 2 bins

  # Create binned categories using quantiles of the expected distribution
  # This ensures stable bin definitions across expected vs actual
  breaks <- quantile(expected, probs = seq(0, 1, length.out = n_bins + 1),
                     na.rm = TRUE)
  breaks[1] <- -Inf                                            # ensure left tail included
  breaks[length(breaks)] <- Inf                                # ensure right tail included

  # Bin both distributions using the same breakpoints
  exp_binned <- cut(expected, breaks = breaks, include.lowest = TRUE)
  act_binned <- cut(actual, breaks = breaks, include.lowest = TRUE)

  # Calculate bin proportions (with small epsilon to avoid log(0))
  eps <- 1e-6
  exp_pct <- table(exp_binned) / length(expected)             # proportion per bin (expected)
  act_pct <- table(act_binned) / length(actual)               # proportion per bin (actual)

  # Ensure both have same bins (in case actual is sparse in some bins)
  all_bins <- union(names(exp_pct), names(act_pct))
  exp_pct <- exp_pct[all_bins]
  act_pct <- act_pct[all_bins]
  exp_pct[is.na(exp_pct)] <- eps                              # fill missing expected bins
  act_pct[is.na(act_pct)] <- eps                              # fill missing actual bins

  # PSI formula: sum of (actual_pct - expected_pct) * ln(actual_pct / expected_pct)
  psi_val <- sum((as.numeric(act_pct) - as.numeric(exp_pct)) *
                 log(as.numeric(act_pct) / as.numeric(exp_pct)))

  psi_val                                                     # return scalar PSI
}

#' Information Value for credit scoring variable selection
#'
#' IV quantifies the predictive power of a feature for a binary target.
#' For numeric features, values are binned into quantiles. For each bin,
#' Weight of Evidence (WoE) and Information Value components are calculated.
#'
#' Formula (per bin i):
#'   WoE_i = ln(% events_i / % non-events_i)
#'   IV_i = (% events_i - % non-events_i) × WoE_i
#'   Total IV = Σ IV_i across all bins
#'
#' Interpretation:
#'   - IV < 0.02:        Not useful for predicting target
#'   - IV 0.02-0.1:      Weak predictive power
#'   - IV 0.1-0.3:       Medium predictive power (typical for strong variables)
#'   - IV 0.3-0.5:       Strong predictive power (watch for overfitting)
#'   - IV > 0.5:         Suspicious / likely data leakage or unnatural pattern
#'
#' @param feature Numeric or factor vector (feature to evaluate)
#' @param target  Binary 0/1 outcome vector (same length as feature)
#' @param n_bins  Number of quantile bins for numeric features (default 10)
#' @return List with:
#'   - iv_total: Scalar IV value
#'   - woe_table: Data frame with columns (bin, pct_events, pct_non_events, woe, iv)
#' @examples
#'   feature <- rnorm(500, mean = 0, sd = 1)
#'   target  <- rbinom(500, size = 1, prob = 0.3)
#'   result <- information_value(feature, target, n_bins = 10)
#'   result$iv_total
#'   head(result$woe_table)
information_value <- function(feature, target, n_bins = 10) {
  stopifnot(is.numeric(target) | is.logical(target))         # guard: target must be binary
  stopifnot(all(target %in% c(0, 1)))                        # guard: binary only
  stopifnot(length(feature) == length(target))               # guard: same length
  stopifnot(n_bins > 1)                                       # guard: need at least 2 bins

  # For numeric features, create quantile bins; for factors, use levels directly
  if (is.numeric(feature)) {
    # Bin numeric feature by quantiles; handle ties by using unique() if bins collapse
    breaks <- quantile(feature, probs = seq(0, 1, length.out = n_bins + 1),
                       na.rm = TRUE)
    breaks <- unique(breaks)                                  # remove duplicate breaks (ties)
    feature_binned <- cut(feature, breaks = breaks, include.lowest = TRUE,
                          dig.lab = 4)                        # more digits for readability
  } else if (is.factor(feature)) {
    feature_binned <- feature
  } else {
    stop("feature must be numeric or factor")
  }

  # Create a contingency table: bins × binary target
  # Rows = bins, Cols = target levels (0, 1)
  cont_table <- table(feature_binned, target)

  # Calculate distribution of events (1s) and non-events (0s) within each bin
  n_events <- cont_table[, "1"]                              # count of events per bin
  n_non_events <- cont_table[, "0"]                          # count of non-events per bin
  total_events <- sum(n_events)                              # total count of events
  total_non_events <- sum(n_non_events)                      # total count of non-events

  # Percentage of events and non-events within each bin
  pct_events <- n_events / total_events
  pct_non_events <- n_non_events / total_non_events

  # Add epsilon to avoid log(0) in WoE calculation
  eps <- 1e-6
  pct_events <- pmax(pct_events, eps)
  pct_non_events <- pmax(pct_non_events, eps)

  # Weight of Evidence (WoE) per bin: ln(% of events / % of non-events)
  woe <- log(pct_events / pct_non_events)

  # Information Value component per bin: (% events - % non-events) × WoE
  iv_component <- (pct_events - pct_non_events) * woe

  # Total IV is sum of IV components across all bins
  iv_total <- sum(iv_component)

  # Assemble results into a readable table
  woe_table <- tibble(
    bin = names(n_events),
    pct_events = as.numeric(pct_events),
    pct_non_events = as.numeric(pct_non_events),
    woe = as.numeric(woe),
    iv = as.numeric(iv_component)
  )

  list(iv_total = iv_total, woe_table = woe_table)            # return both total and table
}

# =============================================================================
# SECTION 4 — OUTPUT MANAGEMENT
# =============================================================================
# Why: hardcoding output paths is fragile and pollutes the project root.
# These helpers ensure every run writes to a clean, timestamped directory
# so outputs are traceable and old results aren't silently overwritten.

#' Ensure output directory exists, creating it if necessary
#' @param dir Path to the output directory
#' @return The directory path (invisible), used for chaining
#' @examples ensure_dir("outputs/claims/")
ensure_dir <- function(dir) {
  if (!dir.exists(dir)) {
    dir.create(dir, recursive = TRUE)                        # recursive handles nested paths
    log_info("Created output directory:", dir)
  }
  invisible(dir)
}

#' Save a data frame to CSV with logging
#'
#' Wrapper around write.csv that logs what was saved and where,
#' creating the parent directory if it doesn't exist.
#'
#' @param df       Data frame to save
#' @param path     File path for the CSV
#' @param ...      Additional arguments passed to write.csv
#' @return Invisible path string
#' @examples save_csv(summary_table, "outputs/claims/region_summary.csv")
save_csv <- function(df, path, ...) {
  ensure_dir(dirname(path))                                  # make sure parent dir exists
  write.csv(df, path, row.names = FALSE, ...)
  log_info("Saved", nrow(df), "rows to", path)
  invisible(path)
}

#' Save a ggplot to file with logging and consistent defaults
#' @param plot   A ggplot object
#' @param path   Output file path (.png, .pdf, etc.)
#' @param width  Width in inches (default 8)
#' @param height Height in inches (default 5)
#' @param dpi    Resolution (default 300 for print quality)
#' @return Invisible path string
save_plot <- function(plot, path, width = 8, height = 5, dpi = 300) {
  ensure_dir(dirname(path))
  ggsave(path, plot, width = width, height = height, dpi = dpi)
  log_info("Saved plot to", path)
  invisible(path)
}

# =============================================================================
# SECTION 5 — MODEL DIAGNOSTICS (Reusable Across Projects)
# =============================================================================
# Why: every GLM in this portfolio needs residual checks. Centralising
# the diagnostic functions means consistent formatting and no code duplication.

#' Print a tidy model summary with key goodness-of-fit statistics
#' @param model A fitted glm or glm.nb object
#' @param label Human-readable model name for the header
#' @return Invisible NULL; prints formatted summary
print_model_summary <- function(model, label = "Model") {
  cat("\n", strrep("=", 60), "\n")
  cat(" ", label, "\n")
  cat(strrep("=", 60), "\n")
  cat("Family:       ", family(model)$family, "\n")
  cat("Link:         ", family(model)$link, "\n")
  cat("AIC:          ", round(AIC(model), 1), "\n")
  cat("BIC:          ", round(BIC(model), 1), "\n")
  cat("Deviance:     ", round(deviance(model), 1), "\n")
  cat("Df residual:  ", df.residual(model), "\n")
  cat("Dispersion:   ", round(sum(residuals(model, "pearson")^2) / df.residual(model), 3), "\n")
  cat(strrep("-", 60), "\n")
  print(tidy(model) %>% mutate(across(where(is.numeric), ~ round(., 4))))
  cat(strrep("=", 60), "\n\n")
  invisible(NULL)
}

#' Compute overdispersion ratio for a Poisson GLM
#'
#' Ratio = sum(Pearson residuals^2) / residual df.
#' Values > 1 suggest overdispersion; values > ~1.5 strongly suggest
#' switching to Negative Binomial or quasi-Poisson.
#'
#' @param model A fitted glm object with family = poisson
#' @return Named list with ratio, p_value (from chi-sq test), and recommendation
overdispersion_test <- function(model) {
  pearson_chi2 <- sum(residuals(model, type = "pearson")^2)  # Pearson chi-squared
  df_resid <- df.residual(model)
  ratio <- pearson_chi2 / df_resid                           # dispersion estimate
  p_value <- pchisq(pearson_chi2, df_resid, lower.tail = FALSE)  # formal test
  recommendation <- case_when(
    ratio < 1.2 ~ "No significant overdispersion — Poisson is adequate",
    ratio < 2.0 ~ "Moderate overdispersion — consider Negative Binomial",
    TRUE         ~ "Strong overdispersion — Negative Binomial recommended"
  )
  log_info("Overdispersion test: ratio =", round(ratio, 3),
           "| p =", format.pval(p_value, digits = 3))
  list(ratio = ratio, p_value = p_value, recommendation = recommendation)
}

# =============================================================================
# SECTION 6 — THEME AND PLOTTING DEFAULTS
# =============================================================================
# Why: consistent visual identity across all portfolio outputs.
# Using a shared theme means every plot looks professional and the
# interviewer sees a cohesive body of work.

#' Portfolio ggplot theme — clean, professional, presentation-ready
#' @return A ggplot theme object
theme_portfolio <- function() {
  theme_minimal(base_size = 11, base_family = "") +
    theme(
      plot.title       = element_text(face = "bold", size = 13, margin = margin(b = 8)),
      plot.subtitle    = element_text(colour = "grey40", size = 10, margin = margin(b = 12)),
      plot.caption     = element_text(colour = "grey60", size = 8, hjust = 0),
      panel.grid.minor = element_blank(),                    # remove minor gridlines
      panel.grid.major = element_line(colour = "grey90"),    # subtle major gridlines
      strip.text       = element_text(face = "bold"),        # facet labels
      legend.position  = "bottom"                            # consistent legend placement
    )
}
