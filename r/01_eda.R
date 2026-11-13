# 01_eda.R
# Exploratory data analysis. Prints summary stats and saves plots to plots/.
# Run from the project root:  Rscript R/01_eda.R

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
})
source("R/00_config.R")

dir.create(PLOT_DIR, showWarnings = FALSE)
df <- load_data()

save_plot <- function(p, name) {
  suppressWarnings(ggsave(file.path(PLOT_DIR, name), p,
                          width = 8, height = 5, dpi = 120))
  cat("Saved", file.path(PLOT_DIR, name), "\n")
}

# ---- Summary statistics ----------------------------------------------------
cat("\n--- Summary ---\n")
print(summary(df[, c(TARGETS, NUM_FEATURES, "likes")]))

# ---- 1. Distribution of views (log scale because views are very skewed) ---
p <- ggplot(df, aes(x = views)) +
  geom_histogram(bins = 40, fill = "steelblue", colour = "white") +
  scale_x_log10() +
  labs(title = "Distribution of views (log scale)", x = "Views", y = "Videos")
save_plot(p, "01_views_distribution.png")

# ---- 2. Subscribers vs views ----------------------------------------------
p <- ggplot(df, aes(x = subscribers, y = views)) +
  geom_point(alpha = 0.4, colour = "steelblue") +
  geom_smooth(method = "lm", formula = y ~ x, colour = "firebrick") +
  scale_x_log10() + scale_y_log10() +
  labs(title = "Subscribers vs views (log-log)", x = "Subscribers", y = "Views")
save_plot(p, "02_subscribers_vs_views.png")

# ---- 3. Duration vs views --------------------------------------------------
p <- ggplot(df, aes(x = duration_sec, y = views)) +
  geom_point(alpha = 0.4, colour = "darkorange") +
  geom_smooth(method = "loess", formula = y ~ x, colour = "firebrick") +
  scale_x_log10() + scale_y_log10() +
  labs(title = "Video duration vs views (log-log)",
       x = "Duration (seconds)", y = "Views")
save_plot(p, "03_duration_vs_views.png")

# ---- 4. Likes vs comments --------------------------------------------------
p <- ggplot(filter(df, !is.na(comments), comments > 0, likes > 0),
            aes(x = likes, y = comments)) +
  geom_point(alpha = 0.4, colour = "darkgreen") +
  geom_smooth(method = "lm", formula = y ~ x, colour = "firebrick") +
  scale_x_log10() + scale_y_log10() +
  labs(title = "Likes vs comments (log-log)", x = "Likes", y = "Comments")
save_plot(p, "04_likes_vs_comments.png")

# ---- 5. Which categories get more views? -----------------------------------
cat_stats <- df %>%
  group_by(category) %>%
  summarise(videos = n(),
            median_views = median(views),
            mean_views = round(mean(views)),
            median_comments = median(comments, na.rm = TRUE),
            .groups = "drop") %>%
  arrange(desc(median_views))
cat("\n--- By category ---\n")
print(as.data.frame(cat_stats), row.names = FALSE)

p <- ggplot(cat_stats, aes(x = reorder(category, median_views), y = median_views)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Median views by category", x = "Category", y = "Median views")
save_plot(p, "05_views_by_category.png")

# ---- 6. Shorts vs long-form -------------------------------------------------
type_stats <- df %>%
  group_by(video_type) %>%
  summarise(videos = n(), median_views = median(views),
            median_comments = median(comments, na.rm = TRUE), .groups = "drop")
cat("\n--- Shorts vs long-form ---\n")
print(as.data.frame(type_stats), row.names = FALSE)

p <- ggplot(df, aes(x = video_type, y = views, fill = video_type)) +
  geom_boxplot(show.legend = FALSE) +
  scale_y_log10() +
  labs(title = "Views: Shorts vs long-form (log scale)", x = NULL, y = "Views")
save_plot(p, "06_shorts_vs_longform.png")

# ---- 7. Correlation heatmap (log scale, since the data is very skewed) -----
num_cols <- c(TARGETS, NUM_FEATURES, "likes")
cor_mat <- cor(log1p(df[, num_cols]), use = "pairwise.complete.obs")
cat("\n--- Correlation matrix (log1p of each variable) ---\n")
print(round(cor_mat, 2))

cor_df <- as.data.frame(as.table(cor_mat))
names(cor_df) <- c("var1", "var2", "correlation")
p <- ggplot(cor_df, aes(x = var1, y = var2, fill = correlation)) +
  geom_tile() +
  geom_text(aes(label = round(correlation, 2)), size = 3) +
  scale_fill_gradient2(low = "firebrick", mid = "white", high = "steelblue",
                       limits = c(-1, 1)) +
  labs(title = "Correlation between variables (log scale)", x = NULL, y = NULL) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
save_plot(p, "07_correlation.png")

cat("\nEDA finished. Plots are in", PLOT_DIR, "\n")