DATA_PATH <- "data/youtube_dataset.csv"
MODEL_DIR <- "models"
PLOT_DIR  <- "plots"

TARGETS <- c("views", "comments", "likes")

NUM_FEATURES <- c("subscribers", "duration_sec", "title_length")
CAT_FEATURES <- c("category")
FEATURES     <- c(NUM_FEATURES, CAT_FEATURES)

LOG_FEATURES <- c("subscribers", "duration_sec")

SHORTS_MAX_SECONDS <- 60

GROUP_COL <- "channel_id"

EXTRA_COLS <- c("video_type")

SEED           <- 42
TRAIN_FRACTION <- 0.8

INTERVAL <- c(0.1, 0.9)


parse_iso_duration <- function(x) {
  part <- function(unit) {
    found <- grepl(paste0("[0-9]+", unit), x)
    out <- rep(0, length(x))
    out[found] <- as.numeric(sub(paste0(".*?([0-9]+)", unit, ".*"), "\\1",
                                 x[found], perl = TRUE))
    out
  }
  part("D") * 86400 + part("H") * 3600 + part("M") * 60 + part("S")
}

load_data <- function() {
  if (!file.exists(DATA_PATH)) {
    stop("Cannot find ", DATA_PATH,
         ". Run the script from the project root folder.", call. = FALSE)
  }

  df <- read.csv(DATA_PATH, stringsAsFactors = FALSE, encoding = "UTF-8")
  n_raw <- nrow(df)

  raw_needed <- c("title", "channel_id", "duration", "views", "likes",
                  "comments", "subscribers", "search_category")
  missing <- setdiff(raw_needed, names(df))
  if (length(missing) > 0) {
    stop("These columns are not in your CSV: ", paste(missing, collapse = ", "),
         "\nYour CSV has: ", paste(names(df), collapse = ", "), call. = FALSE)
  }

  df <- unique(df)

  for (col in c("views", "likes", "comments", "subscribers")) {
    df[[col]] <- suppressWarnings(as.numeric(df[[col]]))
  }

  df$duration_sec <- parse_iso_duration(df$duration)
  df$title_length <- nchar(df$title, type = "chars", allowNA = TRUE)
  df$category     <- as.factor(trimws(df$search_category))
  df$video_type   <- ifelse(df$duration_sec <= SHORTS_MAX_SECONDS,
                            "Shorts", "Long-form")

  df <- df[, c(TARGETS, FEATURES, GROUP_COL, EXTRA_COLS), drop = FALSE]

  df <- df[complete.cases(df[, c(FEATURES, GROUP_COL)]), , drop = FALSE]
  df <- df[df$duration_sec > 0, , drop = FALSE]
  for (t in TARGETS) df <- df[is.na(df[[t]]) | df[[t]] >= 0, , drop = FALSE]
  df <- droplevels(df)

  if (nrow(df) < 20) {
    stop("Only ", nrow(df), " usable rows after cleaning. Need at least 20.",
         call. = FALSE)
  }

  cat("Loaded", n_raw, "rows,", nrow(df), "usable after cleaning.\n")
  for (t in TARGETS) {
    cat("  rows with", t, ":", sum(!is.na(df[[t]])), "\n")
  }
  df
}