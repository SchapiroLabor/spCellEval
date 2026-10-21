# Shared utilities for R clustering methods.
# Sourced by run_phenograph.R, run_flowsom.R, run_fusesom.R.

# Optional transformation of the markers (applied after normalization), as in the Leiden script
transform_markers <- function(marker_df, transform, cofactor) {
  if (transform == "log1p") return(log1p(marker_df))
  if (transform == "arcsinh") return(asinh(marker_df / cofactor))
  marker_df
}

# Optional PCA (30 PCs, or n_markers - 1 if fewer markers) and/or Harmony integration on sample_id
# (on the PCs if PCA, otherwise on the markers), as in the Leiden script
embed_markers <- function(marker_df, meta_df, pca, harmony) {
  x <- as.matrix(marker_df)
  if (pca)
    x <- prcomp(x)$x[, seq_len(min(30, ncol(x) - 1))]
  if (harmony)
    # sample_id as factor, so numeric sample ids are treated as batch labels
    x <- harmony::RunHarmony(x, data.frame(sample_id = factor(meta_df$sample_id)), "sample_id", verbose = FALSE)
  x
}

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
  write.csv(df, path, row.names = FALSE)
}

# Uses globals: greedy_f1_score, df, granularity, clustering, iteration, args, original_cols
# folder is the method folder the predictions are saved in (default: the clustering column name)
greedy_assignment_save <- function(folder = clustering) {
  for (k in granularity) {
    results <- greedy_f1_score(df, k, clustering, tie_strategy = "random")
    output <- select(df, all_of(original_cols), all_of(clustering))
    output$predicted_phenotype <- as.vector(results$mapped_predictions)
    output <- rename(output, true_phenotype = all_of(k))
    safe_write_csv(output, paste0(args$output_path, "/", folder,
                                  "/", names(granularity[granularity == k]),
                                  "/predictions_", iteration, ".csv"))
  }
}