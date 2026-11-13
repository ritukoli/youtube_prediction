suppressPackageStartupMessages(library(ranger))
source("R/00_config.R")

dir.create(MODEL_DIR, showWarnings = FALSE)

df <- load_data()


set.seed(SEED)
channels       <- unique(as.character(df[[GROUP_COL]]))
train_channels <- sample(channels, floor(TRAIN_FRACTION * length(channels)))
in_train       <- as.character(df[[GROUP_COL]]) %in% train_channels
train <- df[in_train, ]
test  <- df[!in_train, ]
cat("Training videos:", nrow(train), "| Test videos:", nrow(test), "\n")

evaluate <- function(actual, pred) {
  data.frame(
    MAE    = round(mean(abs(actual - pred)), 2),
    RMSE   = round(sqrt(mean((actual - pred)^2)), 2),
    R2     = round(1 - sum((actual - pred)^2) /
                     sum((actual - mean(actual))^2), 3),
    R2_log = round(1 - sum((log1p(actual) - log1p(pred))^2) /
                     sum((log1p(actual) - mean(log1p(actual)))^2), 3)
  )
}


make_x <- function(data) {
  x <- data[, FEATURES, drop = FALSE]
  for (f in LOG_FEATURES) x[[f]] <- log1p(x[[f]])
  x
}

make_model_data <- function(data, target) {
  out <- make_x(data)
  out$target_log <- log1p(data[[target]])
  out
}

formula_all <- as.formula(paste("target_log ~", paste(FEATURES, collapse = " + ")))

all_metrics <- list()

for (target in TARGETS) {
  cat("\n=== Target:", target, "===\n")

  tr <- train[!is.na(train[[target]]), ]
  te <- test[!is.na(test[[target]]), ]

  for (col in CAT_FEATURES) te <- te[te[[col]] %in% tr[[col]], ]
  if (nrow(te) == 0) stop("No test rows left for target '", target, "'.")
  cat("Rows used - train:", nrow(tr), "| test:", nrow(te), "\n")

  train_m <- make_model_data(tr, target)
  test_x  <- make_x(te)
  actual  <- te[[target]]

  lm_fit <- lm(formula_all, data = train_m)
  rf_fit <- ranger(formula_all, data = train_m, num.trees = 500,
                   importance = "impurity", quantreg = TRUE, seed = SEED)

  lm_pred <- pmax(expm1(predict(lm_fit, newdata = test_x)), 0)
  rf_pred <- pmax(expm1(predict(rf_fit, data = test_x)$predictions), 0)

  m_lm <- evaluate(actual, lm_pred)
  m_rf <- evaluate(actual, rf_pred)

  use_rf <- m_rf$MAE <= m_lm$MAE
  best   <- if (use_rf) "random_forest" else "linear_regression"
  cat("Best model for", target, ":", best, "\n")

  resid_q <- as.numeric(quantile(residuals(lm_fit), INTERVAL))
  if (use_rf) {
    q <- predict(rf_fit, data = test_x, type = "quantiles",
                 quantiles = INTERVAL)$predictions
    low  <- expm1(q[, 1])
    high <- expm1(q[, 2])
  } else {
    p_log <- predict(lm_fit, newdata = test_x)
    low   <- expm1(p_log + resid_q[1])
    high  <- expm1(p_log + resid_q[2])
  }
  coverage <- mean(actual >= low & actual <= high)
  cat("Share of test videos inside the", paste0(INTERVAL * 100, "%", collapse = "-"),
      "range:", round(coverage, 3), "\n")

  tbl <- rbind(
    cbind(target = target, model = "linear_regression", m_lm),
    cbind(target = target, model = "random_forest",     m_rf)
  )
  tbl$Range_coverage <- NA_real_
  tbl$Range_coverage[tbl$model == best] <- round(coverage, 3)
  print(tbl, row.names = FALSE)
  all_metrics[[target]] <- tbl

  bundle <- list(
    target       = target,
    type         = if (use_rf) "ranger" else "lm",
    model        = if (use_rf) rf_fit else lm_fit,
    features     = FEATURES,
    num_features = NUM_FEATURES,
    cat_features = CAT_FEATURES,
    log_features = LOG_FEATURES,
    levels       = lapply(df[CAT_FEATURES], levels),  # allowed dropdown values
    interval     = INTERVAL,
    resid_q      = resid_q,
    metrics      = tbl,
    importance   = if (use_rf) sort(rf_fit$variable.importance, decreasing = TRUE) else NULL
  )

  saveRDS(bundle, file.path(MODEL_DIR, paste0(target, "_model.rds")))
  cat("Saved", file.path(MODEL_DIR, paste0(target, "_model.rds")), "\n")
}

# Save the metrics and dropdown options for the Streamlit app 
metrics_df <- do.call(rbind, all_metrics)
write.csv(metrics_df, file.path(MODEL_DIR, "metrics.csv"), row.names = FALSE)

choices <- lapply(df[CAT_FEATURES], levels)
writeLines(jsonlite::toJSON(choices), file.path(MODEL_DIR, "choices.json"))

cat("\nTraining finished. Models saved in", MODEL_DIR, "\n")