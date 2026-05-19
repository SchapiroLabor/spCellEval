# Shared utilities for R clustering methods.
# Sourced by run_phenograph.R, run_flowsom.R, run_fusesom.R.

granularity <- c("level1" = "level_1_cell_type", "level2" = "level_2_cell_type", "level3" = "cell_type")

greedy_f1_score <- function(df, true_label_col, predicted_cluster_col, tie_strategy = "first") {
  contingency <- table(df[[true_label_col]], df[[predicted_cluster_col]])
  clusters <- unique(df[[predicted_cluster_col]])

  cluster_to_label <- list()
  for (cluster in as.character(clusters)) {
    if (!cluster %in% colnames(contingency)) { cluster_to_label[[cluster]] <- NA; next }
    col_counts <- contingency[, cluster]
    top_labels <- names(col_counts[col_counts == max(col_counts)])
    cluster_to_label[[cluster]] <- if (length(top_labels) == 1 || tie_strategy == "first")
      top_labels[1] else sample(top_labels, 1)
  }

  mapped_predictions <- sapply(as.character(df[[predicted_cluster_col]]),
    function(c) { lbl <- cluster_to_label[[c]]; if (is.null(lbl) || is.na(lbl)) "unmapped" else lbl })

  list(mapped_predictions = mapped_predictions, mapping = cluster_to_label)
}

safe_write_csv <- function(df, path) {
  dir <- dirname(path)
  if (!dir.exists(dir)) dir.create(dir, recursive = TRUE)
  write.csv(df, path)
}

# Uses globals: greedy_f1_score, df, granularity, clustering, iteration, args, original_cols
greedy_assignment_save <- function() {
  for (k in granularity) {
    results <- greedy_f1_score(df, k, clustering, tie_strategy = "random")
    output <- select(df, all_of(original_cols), clustering)
    output$predicted_phenotype <- as.vector(results$mapped_predictions)
    output <- rename(output, true_phenotype = k)
    safe_write_csv(output, paste0(args$output_path, "/", clustering,
                                  "/", names(granularity[granularity == k]),
                                  "/predictions_", iteration, ".csv"))
  }
}