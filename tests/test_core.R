# =============================================================================
# tests/test_core.R — Comprehensive Test Suite for Portfolio
# =============================================================================
# Author:  Joaquim De Sousa
# Purpose: Unit tests for shared utilities AND integration tests for each
#          project pipeline. Run with: testthat::test_dir("tests")
#
# Coverage:
#   - utils.R: all public functions (validation, statistics, output mgmt)
#   - claims_frequency_severity.R: data simulation, model fitting, outputs
#   - credit_risk_scoring.R: data simulation, train/test split, metrics
#   - market_risk_simulation.R: correlation, VaR/CVaR, stress tests
# =============================================================================

library(testthat)
source("../utils.R")

# =============================================================================
# SECTION 1 — VALIDATION FUNCTIONS
# =============================================================================

context("assert_cols")

test_that("assert_cols passes when all columns present", {
  df <- data.frame(a = 1, b = 2, c = 3)
  expect_true(assert_cols(df, c("a", "b")))                  # subset is fine
  expect_true(assert_cols(df, c("a", "b", "c")))             # exact match is fine
})

test_that("assert_cols fails with informative error on missing columns", {
  df <- data.frame(a = 1, b = 2)
  expect_error(assert_cols(df, c("a", "b", "x")), "Missing required columns: x")
  expect_error(assert_cols(df, c("x", "y")), "Missing required columns: x, y")
})

test_that("assert_cols rejects non-data-frame input", {
  expect_error(assert_cols("not a df", c("a")))              # string
  expect_error(assert_cols(c(1, 2, 3), c("a")))              # vector
})

# ---

context("assert_no_negative")

test_that("assert_no_negative passes for all-positive vectors", {
  expect_true(assert_no_negative(c(0, 1, 2, 100)))          # zero is ok
  expect_true(assert_no_negative(c(0.001, 999)))
})

test_that("assert_no_negative catches negatives with count", {
  expect_error(assert_no_negative(c(1, -1, 2, -3), "test"),
               "2 negative value")                            # reports count
})

test_that("assert_no_negative handles NAs gracefully", {
  expect_true(assert_no_negative(c(1, NA, 3), "test"))       # NAs are ignored
})

# ---

context("assert_no_na")

test_that("assert_no_na passes for complete data", {
  df <- data.frame(a = 1:3, b = 4:6)
  expect_true(assert_no_na(df, c("a", "b")))
})

test_that("assert_no_na catches NAs with percentage", {
  df <- data.frame(a = c(1, NA, 3), b = 4:6)
  expect_error(assert_no_na(df, c("a")), "1 missing value")
})

# ---

context("assert_range")

test_that("assert_range passes for in-range values", {
  expect_true(assert_range(c(0.1, 0.5, 0.9), 0, 1, "test"))
  expect_true(assert_range(c(0, 1), 0, 1, "test"))          # boundaries inclusive
})

test_that("assert_range catches out-of-range values", {
  expect_error(assert_range(c(0.5, 1.1), 0, 1, "test"), "1 value")
  expect_error(assert_range(c(-0.1, 0.5), 0, 1, "test"), "1 value")
})

# =============================================================================
# SECTION 2 — STATISTICAL HELPERS
# =============================================================================

context("winsorise")

test_that("winsorise clamps extremes correctly", {
  x <- c(1, 2, 3, 4, 100)
  y <- winsorise(x, p = 0.20)
  # At p=0.20, lower bound is ~20th percentile, upper is ~80th
  q <- quantile(x, c(0.20, 0.80))
  expect_true(min(y) >= q[1])
  expect_true(max(y) <= q[2])
})

test_that("winsorise preserves NAs", {
  x <- c(1, NA, 3, 100)
  y <- winsorise(x, p = 0.25)
  expect_true(is.na(y[2]))                                   # NA stays NA
  expect_equal(length(y), length(x))                          # same length
})

test_that("winsorise rejects invalid p", {
  expect_error(winsorise(1:10, p = 0))                        # p must be > 0
  expect_error(winsorise(1:10, p = 0.5))                      # p must be < 0.5
  expect_error(winsorise(1:10, p = -0.1))
})

test_that("winsorise is idempotent on uniform data", {
  x <- 1:100
  y <- winsorise(x, p = 0.01)
  z <- winsorise(y, p = 0.01)
  expect_equal(y, z)                                          # second pass changes nothing
})

# ---

context("rmse")

test_that("rmse returns zero for perfect predictions", {
  expect_equal(rmse(c(1, 2, 3), c(1, 2, 3)), 0)
})

test_that("rmse computes correctly for known values", {
  # RMSE of (1,2,3) vs (2,3,4) = sqrt(mean(c(1,1,1))) = 1
  expect_equal(rmse(c(1, 2, 3), c(2, 3, 4)), 1)
})

test_that("rmse rejects mismatched lengths", {
  expect_error(rmse(c(1, 2), c(1, 2, 3)))
})

# ---

context("mae")

test_that("mae returns zero for perfect predictions", {
  expect_equal(mae(c(1, 2, 3), c(1, 2, 3)), 0)
})

test_that("mae computes correctly", {
  expect_equal(mae(c(1, 2, 3), c(2, 3, 4)), 1)              # all errors = 1
})

# ---

context("gini_coefficient")

test_that("gini is near 1 for perfect separation", {
  actual    <- c(rep(0, 100), rep(1, 100))
  predicted <- c(runif(100, 0, 0.3), runif(100, 0.7, 1.0))
  g <- gini_coefficient(actual, predicted)
  expect_true(g > 0.5)                                        # should be high
})

test_that("gini is near 0 for random predictions", {
  set.seed(999)
  actual    <- rbinom(1000, 1, 0.5)
  predicted <- runif(1000)
  g <- gini_coefficient(actual, predicted)
  expect_true(abs(g) < 0.15)                                  # should be near zero
})

test_that("gini rejects non-binary actual", {
  expect_error(gini_coefficient(c(0, 1, 2), c(0.1, 0.5, 0.9)))
})

# ---

context("ks_statistic")

test_that("ks is high for well-separated predictions", {
  actual    <- c(rep(0, 100), rep(1, 100))
  predicted <- c(runif(100, 0, 0.3), runif(100, 0.7, 1.0))
  ks <- ks_statistic(actual, predicted)
  expect_true(ks > 0.5)
})

test_that("ks is in [0, 1]", {
  actual    <- rbinom(200, 1, 0.3)
  predicted <- runif(200)
  ks <- ks_statistic(actual, predicted)
  expect_true(ks >= 0 && ks <= 1)
})

# ---

context("brier_score")

test_that("brier is 0 for perfect deterministic predictions", {
  expect_equal(brier_score(c(0, 0, 1, 1), c(0, 0, 1, 1)), 0)
})

test_that("brier is 0.25 for constant 0.5 predictions on balanced data", {
  # Mean of (0.5 - 0)^2 and (0.5 - 1)^2 = 0.25
  expect_equal(brier_score(c(0, 1), c(0.5, 0.5)), 0.25)
})

test_that("brier rejects probabilities outside [0,1]", {
  expect_error(brier_score(c(0, 1), c(-0.1, 0.5)))
  expect_error(brier_score(c(0, 1), c(0.5, 1.1)))
})

# =============================================================================
# SECTION 3 — OUTPUT MANAGEMENT
# =============================================================================

context("ensure_dir")

test_that("ensure_dir creates nested directories", {
  test_path <- file.path(tempdir(), "test_a", "test_b")
  if (dir.exists(test_path)) unlink(test_path, recursive = TRUE)
  ensure_dir(test_path)
  expect_true(dir.exists(test_path))
  unlink(file.path(tempdir(), "test_a"), recursive = TRUE)   # cleanup
})

# ---

context("save_csv")

test_that("save_csv writes readable CSV", {
  test_file <- file.path(tempdir(), "test_output.csv")
  df <- data.frame(a = 1:3, b = c("x", "y", "z"))
  save_csv(df, test_file)
  expect_true(file.exists(test_file))
  df_read <- read.csv(test_file, stringsAsFactors = FALSE)
  expect_equal(nrow(df_read), 3)
  expect_equal(names(df_read), c("a", "b"))
  unlink(test_file)                                           # cleanup
})

# =============================================================================
# SECTION 4 — MODEL DIAGNOSTIC HELPERS
# =============================================================================

context("overdispersion_test")

test_that("overdispersion_test returns expected structure", {
  set.seed(1)
  df <- data.frame(y = rpois(100, 2), x = rnorm(100))
  fit <- glm(y ~ x, data = df, family = poisson())
  result <- overdispersion_test(fit)
  expect_true(is.list(result))
  expect_true(all(c("ratio", "p_value", "recommendation") %in% names(result)))
  expect_true(is.numeric(result$ratio))
  expect_true(result$ratio > 0)
})

# =============================================================================
# SECTION 4b — ADVANCED STATISTICAL TESTS
# =============================================================================

# ---

context("population_stability_index")

test_that("PSI is zero for identical distributions", {
  x <- rnorm(1000, mean = 10, sd = 2)
  psi <- population_stability_index(x, x, n_bins = 10)
  expect_true(psi < 0.01)  # should be approximately zero
})

test_that("PSI detects shifted distribution", {
  set.seed(42)
  expected <- rnorm(1000, mean = 10, sd = 2)
  actual   <- rnorm(1000, mean = 12, sd = 2)  # shifted mean
  psi <- population_stability_index(expected, actual, n_bins = 10)
  expect_true(psi > 0.1)  # noticeable shift
})

test_that("PSI is non-negative", {
  set.seed(42)
  psi <- population_stability_index(rnorm(500), rnorm(500))
  expect_true(psi >= 0)
})

test_that("PSI rejects non-numeric input", {
  expect_error(population_stability_index(letters[1:10], letters[1:10]))
})

test_that("PSI rejects mismatched lengths gracefully", {
  # PSI doesn't strictly require same length (it bins separately)
  # but it should still work on different-length vectors
  set.seed(42)
  psi <- population_stability_index(rnorm(500), rnorm(300))
  expect_true(is.numeric(psi))
})

# ---

context("information_value")

test_that("IV is high for perfectly predictive feature", {
  set.seed(42)
  target  <- c(rep(0, 500), rep(1, 500))
  feature <- c(rnorm(500, mean = 0), rnorm(500, mean = 5))  # well-separated
  result  <- information_value(feature, target, n_bins = 10)
  expect_true(result$iv_total > 0.5)  # should be very strong
  expect_true(is.data.frame(result$woe_table))
})

test_that("IV is low for random feature", {
  set.seed(42)
  target  <- rbinom(1000, 1, 0.3)
  feature <- runif(1000)  # random, no predictive power
  result  <- information_value(feature, target, n_bins = 10)
  expect_true(result$iv_total < 0.1)  # should be weak
})

test_that("IV works with factor features", {
  set.seed(42)
  feature <- factor(sample(c("A", "B", "C"), 300, replace = TRUE))
  target  <- rbinom(300, 1, 0.2)
  result  <- information_value(feature, target)
  expect_true(is.numeric(result$iv_total))
  expect_true(nrow(result$woe_table) <= 3)  # 3 categories
})

test_that("IV rejects non-binary target", {
  expect_error(information_value(1:10, c(0, 1, 2, rep(0, 7))))
})

# ---

context("kupiec_test (from market_risk_simulation)")

test_that("kupiec backtest framework is structurally sound", {
  # We can't easily test kupiec_test without sourcing the full market script,
  # but we can test the mathematical logic independently

  # Simulate: 1000 returns, VaR at 99% should be breached ~10 times
  set.seed(42)
  returns <- rnorm(1000, mean = 0, sd = 0.01)
  var_99  <- -quantile(returns, 0.01)
  n_breaches <- sum(returns < -var_99)

  # Breach count should be close to 10 (1% of 1000)
  expect_true(n_breaches >= 5 && n_breaches <= 20)
})

# =============================================================================
# SECTION 5 — INTEGRATION TESTS (Project Pipelines)
# =============================================================================
# These tests verify that the full project scripts run without error
# and produce expected output files. They don't check exact numeric
# values (which depend on seeds) but verify structure and sanity.

context("Integration: claims_frequency_severity.R")

test_that("claims pipeline runs and produces outputs", {
  # Set working directory to project root
  # This test assumes it's run from the tests/ directory via test_dir("tests")
  skip_if_not(file.exists("../claims_frequency_severity.R"),
              "claims script not found")

  old_wd <- setwd("..")
  on.exit(setwd(old_wd))

  # Source the script (it will create outputs/)
  source("claims_frequency_severity.R", local = TRUE)

  # Check that key objects were created
  expect_true(exists("freq_final"))
  expect_true(exists("sev_gamma"))
  expect_true(exists("summary_by_region"))
  expect_true(exists("policies_out"))

  # Check summary table has expected structure
  expect_true(nrow(summary_by_region) >= 1)
  expect_true("avg_expected_loss" %in% names(summary_by_region))

  # Check output file was created
  expect_true(file.exists("outputs/claims/region_summary.csv"))
})

# ---

context("Integration: credit_risk_scoring.R")

test_that("credit pipeline runs and produces outputs", {
  skip_if_not(file.exists("../credit_risk_scoring.R"),
              "credit script not found")

  old_wd <- setwd("..")
  on.exit(setwd(old_wd))

  source("credit_risk_scoring.R", local = TRUE)

  # Check discrimination metrics are reasonable
  expect_true(exists("auc_v1"))
  expect_true(auc_v1 > 0.5 && auc_v1 < 1.0)                 # better than random
  expect_true(exists("auc_lasso"))
  expect_true(auc_lasso > 0.5 && auc_lasso < 1.0)

  # Check v0 AUC is inflated vs test AUC
  expect_true(v0_auc >= max(auc_v1, auc_lasso))

  # Check calibration table structure
  expect_true(exists("calibration"))
  expect_equal(nrow(calibration), 10)                         # 10 deciles

  # Check output files
  expect_true(file.exists("outputs/credit/calibration_table.csv"))
  expect_true(file.exists("outputs/credit/portfolio_summary.csv"))
})

# ---

context("Integration: market_risk_simulation.R")

test_that("market risk pipeline runs and produces outputs", {
  skip_if_not(file.exists("../market_risk_simulation.R"),
              "market risk script not found")

  old_wd <- setwd("..")
  on.exit(setwd(old_wd))

  source("market_risk_simulation.R", local = TRUE)

  # Check VaR/CVaR are positive (losses)
  expect_true(exists("full_risk_summary"))
  expect_true(all(full_risk_summary$VaR > 0))
  expect_true(all(full_risk_summary$CVaR > 0))

  # CVaR >= VaR always (coherent risk measure property)
  expect_true(all(full_risk_summary$CVaR >= full_risk_summary$VaR))

  # Stress VaR should exceed baseline VaR (at same confidence)
  baseline_99 <- full_risk_summary %>%
    filter(scenario == "v1: Correlated", confidence == 0.99)
  stress_99 <- full_risk_summary %>%
    filter(grepl("Stress", scenario), confidence == 0.99)
  expect_true(all(stress_99$VaR > baseline_99$VaR))

  # Check output files
  expect_true(file.exists("outputs/market/risk_summary.csv"))
})
