#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  suppressWarnings(library(argparse))
  suppressWarnings(library(readr))
  suppressWarnings(library(dplyr))
  suppressWarnings(library(FuseSOM))
  suppressWarnings(library(logging))
})

# Resolve utils dir relative to this script
script_dir <- dirname(normalizePath(
  sub("--file=", "", commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))])
))
utils_dir <- normalizePath(file.path(script_dir, "..", "utils"))
source(file.path(utils_dir, "clustering_utils.R"))

#### Args ####
parser <- ArgumentParser(description = "Cluster data with FuseSOM")
parser$add_argument("-i", "--input", dest = "input", required = TRUE,
                    help = "Path to input CSV file")
parser$add_argument("-s", "--split_col_name", dest = "split_col_name", required = TRUE,
                    help = "First non-marker column; all columns before it are used as markers (e.g. cell_id)")
parser$add_argument("-e", "--exclude_markers", dest = "exclude_markers", nargs = "+",
                    help = "Marker names to exclude from clustering (e.g. DAPI)")
parser$add_argument("-o", "--output", dest = "output_path", required = TRUE,
                    help = "Path to output folder")
parser$add_argument("-fuse", "--fuse_clusters", dest = "fuse_clusters",
                    type = "integer", required = TRUE,
                    help = "Number of clusters for FuseSOM (FuseSOM has no default)")
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
                    help = "Minimal number of cells a cluster needs to be subclustered; always more than 10 cells per node of the small grid (default: 10)")
parser$add_argument("-sk", "--subcluster_k", dest = "subcluster_k", type = "integer", default = 2,
                    help = "Number of clusters per subclustered cluster (default: 2)")
parser$add_argument("-sg", "--subcluster_small_grid", dest = "subcluster_small_grid", type = "integer", default = 5,
                    help = "SOM grid side length used for clusters too small for FuseSOM's own grid size (default: 5, the FuseSOM minimum)")
args <- parser$parse_args()

# Only cell_type (level3) is evaluated
granularity <- c("level3" = "cell_type")

#### Logging ####
if (args$log != "off") {
  if (!dir.exists(paste0(args$output_path, "/logs")))
    dir.create(paste0(args$output_path, "/logs"), recursive = TRUE)
  basicConfig()
  addHandler(writeToFile,
    file = paste0(args$output_path, "/logs/", Sys.Date(), "_fusesom.log"),
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
# Each round re-runs FuseSOM on every cluster with >= min_cells, new labels are '<parent>,<child>'.
# The SOM grid keeps > CELLS_PER_NODE cells per node on average: FuseSOM's own grid size
# (Tracy-Widom bound, at least 5x5) if possible, otherwise small_grid x small_grid; smaller clusters are kept.
CELLS_PER_NODE <- 10
subcluster <- function(marker_df, labels, times, min_cells, k, small_grid) {
  min_cells <- max(min_cells, CELLS_PER_NODE * small_grid^2 + 1)
  for (round in seq_len(times)) {
    counts <- table(labels)
    for (cl in names(counts[counts >= min_cells])) {
      idx <- which(labels == cl)
      x <- as.matrix(marker_df[idx, ])
      grid <- max(computeGridSize(x), 5)
      if (length(idx) <= CELLS_PER_NODE * grid^2) grid <- small_grid
      child <- suppressMessages(runFuseSOM(x, numClusters = k, size = grid))$clusters
      labels[idx] <- paste0(cl, ",", sub("^cluster_", "", child))
    }
  }
  labels
}

#### Run ####
fuse_clus <- args$fuse_clusters
clustering <- paste0("FuseSOM_", fuse_clus)
# Subclustering runs are saved in a separate method folder (e.g. 'FuseSOM_10_subclust3')
folder <- if (args$subcluster_times > 0) paste0(clustering, "_subclust", args$subcluster_times) else clustering
for (iteration in 0:(args$iterations - 1)) {
  # Seed per iteration: reproducible, but different between iterations
  set.seed(iteration)
  start_time <- Sys.time()
  fuse_df <- marker_df %>% runFuseSOM(numClusters = fuse_clus)
  labels <- subcluster(marker_df, fuse_df$clusters, args$subcluster_times, args$subcluster_min_cells,
                       args$subcluster_k, args$subcluster_small_grid)
  end_time <- Sys.time()
  elapsed <- round(as.numeric(difftime(end_time, start_time, units = "sec")), 2)

  if (args$log != "off")
    loginfo("FuseSOM numClusters = %d (Duration: %.2f min)", fuse_clus,
            as.numeric(difftime(end_time, start_time, units = "mins")))

  text_path <- file.path(args$output_path, folder, "fold_times.txt")
  dir.create(dirname(text_path), recursive = TRUE, showWarnings = FALSE)
  con <- file(text_path, open = if (iteration == 0) "w" else "a")
  writeLines(sprintf("Fold %d train_time, %.2f", iteration, elapsed), con)
  close(con)

  df[[clustering]] <- factor(labels)
  greedy_assignment_save(folder)
}