# 00_config.R
# The ONE file you edit to change features/targets.
# 01_eda.R and 02_train.R load it with: source("R/00_config.R")
# Always run the scripts from the project root folder (youtube-predictor/).
#
# Your CSV has one row per video with these columns:
#   video_id, title, channel_id, category_id, duration (ISO 8601, e.g. PT9M24S),
#   views, likes, comments, subscribers, search_category

# ---- Paths -----------------------------------------------------------------
DATA_PATH <- "data/youtube_dataset.csv"
MODEL_DIR <- "models"
PLOT_DIR  <- "plots"

# ---- What we predict (one model per target) --------------------------------
TARGETS <- c("views", "comments")

# ---- Model inputs ----------------------------------------------------------
# These are created from the raw CSV columns in load_data() below:
#   subscribers   <- subscribers
#   duration_sec  <- duration (converted from "PT9M24S" to seconds)
#   title_length  <- number of characters in title
#   category      <- search_category
NUM_FEATURES <- c("subscribers", "duration_sec", "title_length")
CAT_FEATURES <- c("category")
FEATURES     <- c(NUM_FEATURES, CAT_FEATURES)

# Very skewed inputs are log-transformed (log1p) before modelling
LOG_FEATURES <- c("subscribers", "duration_sec")

# Videos this short or shorter are labelled "Shorts" (used for EDA only)
SHORTS_MAX_SECONDS <- 60

# Videos from the same channel are kept together when splitting train/test,
# otherwise the model can "recognise" a channel it already saw in training
GROUP_COL <- "channel_id"

# Extra columns kept for the EDA plots (not used as model inputs)
EXTRA_COLS <- c("likes", "video_type")

# ---- Training settings -----------------------------------------------------
SEED           <- 42
TRAIN_FRACTION <- 0.8

# The app shows a likely range: the 10th to 90th percentile of what similar
# videos in the data got (an 80% range)
INTERVAL <- c(0.1, 0.9)

# ---- Helpers ---------------------------------------------------------------

# "PT1H3M6S" -> 3786 seconds. Also handles "PT54S", "PT9M24S", "P1DT2H".
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

# Load the CSV, build the features and clean the data
load_data <- function() {
  if (!file.exists(DATA_PATH)) {
    stop("Cannot find ", DATA_PATH,
         ". Run the script from the project root folder.", call. = FALSE)
  }

  # encoding = "UTF-8" keeps emojis / Hindi titles intact so title length is right
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

  # Numbers
  for (col in c("views", "likes", "comments", "subscribers")) {
    df[[col]] <- suppressWarnings(as.numeric(df[[col]]))
  }

  # Features
  df$duration_sec <- parse_iso_duration(df$duration)
  df$title_length <- nchar(df$title, type = "chars", allowNA = TRUE)
  df$category     <- as.factor(trimws(df$search_category))
  df$video_type   <- ifelse(df$duration_sec <= SHORTS_MAX_SECONDS,
                            "Shorts", "Long-form")

  # Keep only what we use
  df <- df[, c(TARGETS, FEATURES, GROUP_COL, EXTRA_COLS), drop = FALSE]

  # Drop rows with missing/invalid inputs. Missing *targets* (e.g. videos with
  # comments turned off) are kept here and dropped per-target during training.
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