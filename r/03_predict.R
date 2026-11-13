suppressPackageStartupMessages({
  library(jsonlite)
  library(ranger)
})

script_dir <- function() {
  f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  if (length(f) > 0) dirname(normalizePath(f[1])) else "R"
}
MODEL_DIR <- file.path(dirname(script_dir()), "models")

build_newdata <- function(b, input) {
  missing <- setdiff(b$features, names(input))
  if (length(missing) > 0) {
    stop("Missing input field(s): ", paste(missing, collapse = ", "))
  }

  vals <- list()
  for (f in b$num_features) {
    v <- suppressWarnings(as.numeric(input[[f]]))
    if (length(v) != 1 || is.na(v)) stop("Field '", f, "' must be a single number")
    if (v < 0) stop("Field '", f, "' cannot be negative")
    vals[[f]] <- if (f %in% b$log_features) log1p(v) else v
  }
  for (f in b$cat_features) {
    v <- as.character(input[[f]])
    if (length(v) != 1 || !(v %in% b$levels[[f]])) {
      stop("Unknown value for '", f, "': ", paste(v, collapse = ""),
           ". Allowed: ", paste(b$levels[[f]], collapse = ", "))
    }
    vals[[f]] <- factor(v, levels = b$levels[[f]])
  }

  do.call(data.frame, c(vals[b$features],
                        list(stringsAsFactors = FALSE, check.names = FALSE)))
}

predict_one <- function(b, newdata) {
  if (is.null(b$interval)) {
    stop("This model file is from an older version. Re-run: Rscript R/02_train.R")
  }

  if (b$type == "ranger") {
    point <- predict(b$model, data = newdata)$predictions
    q <- predict(b$model, data = newdata, type = "quantiles",
                 quantiles = b$interval)$predictions
    low  <- q[1, 1]
    high <- q[1, 2]
  } else {
    point <- predict(b$model, newdata = newdata)
    low   <- point + b$resid_q[1]
    high  <- point + b$resid_q[2]
  }

  back <- function(z) max(round(expm1(as.numeric(z)[1])), 0)
  list(point = back(point), low = back(low), high = back(high))
}

run <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 1) {
    stop("Usage: Rscript R/03_predict.R '<json string>'  or  <path to json file>")
  }

  txt <- if (file.exists(args[1])) {
    paste(readLines(args[1], warn = FALSE), collapse = "\n")
  } else {
    args[1]
  }
  input <- as.list(fromJSON(txt))

  files <- list.files(MODEL_DIR, pattern = "_model\\.rds$", full.names = TRUE)
  if (length(files) == 0) {
    stop("No models found in ", MODEL_DIR, ". Run: Rscript R/02_train.R")
  }

  out <- list()
  for (f in files) {
    b <- readRDS(f)
    r <- predict_one(b, build_newdata(b, input))
    out[[paste0("predicted_", b$target)]]          <- r$point
    out[[paste0("predicted_", b$target, "_low")]]  <- r$low
    out[[paste0("predicted_", b$target, "_high")]] <- r$high
  }

  cat(toJSON(out, auto_unbox = TRUE))
}

tryCatch(run(), error = function(e) {
  message("ERROR: ", conditionMessage(e))
  quit(status = 1)
})