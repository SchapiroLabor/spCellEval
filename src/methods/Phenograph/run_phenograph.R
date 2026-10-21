#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  suppressWarnings(library(argparse))
  suppressWarnings(library(readr))
  suppressWarnings(library(tidyverse))
  suppressWarnings(library(Rphenograph))
  suppressWarnings(library(logging))
})

# Resolve utils dir relative to this script
script_dir <- dirname(normalizePath(
  sub("--file=", "", commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))])
))
utils_dir <- normalizePath(file.path(script_dir, "..", "utils"))
source(file.path(utils_dir, "clustering_utils.R"))

#### Args ####
parser <- ArgumentParser(description = "Cluster data with PhenoGraph")
parser$add_argument("-i", "--input", dest = "input", required = TRUE,
                    help = "Path to input CSV file")
parser$add_argument("-s", "--split_col_name", dest = "split_col_name", required = TRUE,
                    help = "First non-marker column; all columns before it are used as markers (e.g. cell_id)")
parser$add_argument("-e", "--exclude_markers", dest = "exclude_markers", nargs = "+",
                    help = "Marker names to exclude from clustering (e.g. DAPI)")
parser$add_argument("-o", "--output", dest = "output_path", required = TRUE,
                    help = "Path to output folder")
parser$add_argument("-k", "--k_phenograph", dest = "k_phenograph",
                    type = "integer", default = 30,
                    help = "k for PhenoGraph (default: 30, the Rphenograph default)")
parser$add_argument("-l", "--log", dest = "log", default = "off",
                    choices = c("short", "long", "off"), help = "Logging level (default: off)")
parser$add_argument("-it", "--iterations", dest = "iterations", type = "integer", default = 5,
                    help = "Number of iterations (default: 5)")
parser$add_argument("-p", "--PCA", dest = "PCA", action = "store_true",
                    help = "Cluster on the first 30 PCs instead of the markers (default: FALSE)")
parser$add_argument("-ha", "--harmony", dest = "harmony", action = "store_true",
                    help = "Harmony integration on sample_id; on the PCs with -p, otherwise on the markers (default: FALSE)")
parser$add_argument("-t", "--transform", dest = "transform", default = "none", choices = c("none", "log1p", "arcsinh"),
                    help = "Transformation applied to the markers after normalization (default: none)")
parser$add_argument("-cf", "--cofactor", dest = "cofactor", type = "double", default = 5,
                    help = "Cofactor for arcsinh, i.e. arcsinh(x / cofactor) (default: 5)")
parser$add_argument("-n", "--normalization", dest = "normalization", default = FALSE,
                    help = "Row normalization before clustering (default: FALSE)")
parser$add_argument("-sc", "--subcluster_times", dest = "subcluster_times", type = "integer", default = 0,
                    help = "Number of recursive subclustering rounds applied to the resulting clusters (default: 0 = off)")
parser$add_argument("-sm", "--subcluster_min_cells", dest = "subcluster_min_cells", type = "integer", default = 10,
                    help = "Minimal number of cells a cluster needs to be subclustered; always at least small_k + 1 (default: 10)")
parser$add_argument("-sk", "--subcluster_small_k", dest = "subcluster_small_k", type = "integer", default = 10,
                    help = "k used for subclustering clusters with <= k cells (default: 10)")
args <- parser$parse_args()

# Only cell_type (level3) is evaluated
granularity <- c("level3" = "cell_type")

#### Logging ####
if (args$log != "off") {
  if (!dir.exists(paste0(args$output_path, "/logs")))
    dir.create(paste0(args$output_path, "/logs"), recursive = TRUE)
  basicConfig()
  addHandler(writeToFile,
    file = paste0(args$output_path, "/logs/", Sys.Date(), "_phenograph.log"),
    level = "INFO",
    formatter = function(record) sprintf("%s - %s - %s",
      format(Sys.time(), "%Y-%m-%d %H:%M:%S"), record$levelname, record$msg)
  )
}

#### Data ####
df <- read_csv(args$input, show_col_types = FALSE)
original_cols <- colnames(df)
# Markers are all columns before split_col_name, minus excluded ones
markers <- original_cols[seq_len(match(args$split_col_name, original_cols) - 1)]
marker_df <- select(df, all_of(setdiff(markers, args$exclude_markers)))
if (args$normalization == TRUE)
  marker_df <- as.data.frame(t(apply(marker_df, 1, function(x) x / sum(x))))
marker_df <- transform_markers(marker_df, args$transform, args$cofactor)
# Optionally replace the markers by PCs and/or Harmony-integrated values (also used for subclustering)
set.seed(0)
if (args$PCA || args$harmony)
  marker_df <- embed_markers(marker_df, df, args$PCA, args$harmony)

#### Subclustering ####
# Each round re-runs PhenoGraph on every cluster with >= min_cells (kNN graph rebuilt on the subset),
# new labels are '<parent>,<child>'. Clusters with > k cells use k, smaller ones small_k,
# clusters with <= small_k cells are kept.
subcluster <- function(marker_df, labels, times, min_cells, k, small_k) {
  min_cells <- max(min_cells, small_k + 1)
  for (round in seq_len(times)) {
    counts <- table(labels)
    for (cl in names(counts[counts >= min_cells])) {
      idx <- which(labels == cl)
      sub <- Rphenograph(marker_df[idx, ], k = if (length(idx) > k) k else small_k)
      labels[idx] <- paste0(cl, ",", membership(sub[[2]]))
    }
  }
  labels
}

#### Run ####
k_val <- args$k_phenograph
clustering <- paste0("phenograph_", k_val)
# Subclustering runs are saved in a separate method folder (e.g. 'phenograph_30_subclust3')
folder <- if (args$subcluster_times > 0) paste0(clustering, "_subclust", args$subcluster_times) else clustering
for (iteration in 0:(args$iterations - 1)) {
  # Seed per iteration: reproducible, but different between iterations
  set.seed(iteration)
  start_time <- Sys.time()
  pheno_df <- marker_df %>% Rphenograph(k = k_val)
  labels <- subcluster(marker_df, as.character(membership(pheno_df[[2]])),
                       args$subcluster_times, args$subcluster_min_cells, k_val, args$subcluster_small_k)
  end_time <- Sys.time()
  elapsed <- round(as.numeric(difftime(end_time, start_time, units = "sec")), 2)

  df[[clustering]] <- factor(labels)

  if (args$log != "off")
    loginfo("Rphenograph k = %d (Duration: %.2f min)", k_val,
            as.numeric(difftime(end_time, start_time, units = "mins")))

  text_path <- file.path(args$output_path, folder, "fold_times.txt")
  dir.create(dirname(text_path), recursive = TRUE, showWarnings = FALSE)
  con <- file(text_path, open = if (iteration == 0) "w" else "a")
  writeLines(sprintf("Fold %d train_time, %.2f", iteration, elapsed), con)
  close(con)

  greedy_assignment_save(folder)
}