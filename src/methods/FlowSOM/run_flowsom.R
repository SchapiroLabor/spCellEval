#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  suppressWarnings(library(argparse))
  suppressWarnings(library(readr))
  suppressWarnings(library(dplyr))
  suppressWarnings(library(FlowSOM))
  suppressWarnings(library(logging))
})

# Resolve utils dir relative to this script
script_dir <- dirname(normalizePath(
  sub("--file=", "", commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))])
))
utils_dir <- normalizePath(file.path(script_dir, "..", "utils"))
source(file.path(utils_dir, "clustering_utils.R"))

#### Args ####
parser <- ArgumentParser(description = "Cluster data with FlowSOM")
parser$add_argument("-i", "--input", dest = "input", required = TRUE,
                    help = "Path to input CSV file")
parser$add_argument("-s", "--split_col_name", dest = "split_col_name", required = TRUE,
                    help = "First non-marker column; all columns before it are used as markers (e.g. cell_id)")
parser$add_argument("-e", "--exclude_markers", dest = "exclude_markers", nargs = "+",
                    help = "Marker names to exclude from clustering (e.g. DAPI)")
parser$add_argument("-o", "--output", dest = "output_path", required = TRUE,
                    help = "Path to output folder")
parser$add_argument("-flow", "--flow_clusters", dest = "flow_nclus", type = "integer",
                    help = "Number of metaclusters for FlowSOM (default: number of cell types in cell_type)")
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
                    help = "Number of recursive subclustering rounds applied to the metaclusters (default: 0 = off)")
parser$add_argument("-sk", "--subcluster_k", dest = "subcluster_k", type = "integer", default = 2,
                    help = "Number of metaclusters per subclustered metacluster (default: 2)")
args <- parser$parse_args()

# Only cell_type (level3) is evaluated
granularity <- c("level3" = "cell_type")

#### Logging ####
if (args$log != "off") {
  if (!dir.exists(paste0(args$output_path, "/logs")))
    dir.create(paste0(args$output_path, "/logs"), recursive = TRUE)
  basicConfig()
  addHandler(writeToFile,
    file = paste0(args$output_path, "/logs/", Sys.Date(), "_flowsom.log"),
    level = "INFO",
    formatter = function(record) sprintf("%s - %s - %s",
      format(Sys.time(), "%Y-%m-%d %H:%M:%S"), record$levelname, record$msg)
  )
}

#### Data ####
df <- read_csv(args$input, show_col_types = FALSE)
original_cols <- colnames(df)
# Default number of metaclusters: number of ground truth cell types
if (is.null(args$flow_nclus))
  args$flow_nclus <- length(unique(df$cell_type))
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
# Each round re-runs FlowSOM on every metacluster, new labels are '<parent>,<child>'.
# The SOM grid is the largest one with >= CELLS_PER_NODE cells per node on average, at most GRID x GRID.
# Metaclusters too small for a grid with more nodes than metaclusters (ConsensusClusterPlus runs with
# maxK >= 3) are kept, e.g. < 40 cells (2 x 2 grid) for k = 2.
GRID <- 10            # FlowSOM default xdim = ydim
CELLS_PER_NODE <- 10

# Same as FlowSOM(nClus = k), but FlowSOM's consensus metaclustering fails for k = 2
# (ConsensusClusterPlus needs maxK >= 3), so maxK is raised and the k solution is taken
flowsom_metaclusters <- function(m, k, grid) {
  fsom <- BuildSOM(ReadInput(m), xdim = grid, ydim = grid, silent = TRUE)
  cc <- suppressMessages(ConsensusClusterPlus::ConsensusClusterPlus(t(fsom$map$codes), maxK = max(k, 3),
    reps = 100, pItem = 0.9, pFeature = 1, title = tempdir(), plot = "pdf", verbose = FALSE,
    clusterAlg = "hc", distance = "euclidean"))
  cc[[k]]$consensusClass[GetClusters(fsom)]
}

subcluster <- function(marker_df, labels, times, k) {
  min_cells <- CELLS_PER_NODE * ceiling(sqrt(max(k, 3) + 1))^2
  for (round in seq_len(times)) {
    counts <- table(labels)
    for (cl in names(counts[counts >= min_cells])) {
      idx <- which(labels == cl)
      grid <- min(GRID, floor(sqrt(length(idx) / CELLS_PER_NODE)))
      labels[idx] <- paste0(cl, ",", flowsom_metaclusters(as.matrix(marker_df[idx, ]), k, grid))
    }
  }
  labels
}

#### Run ####
# Subclustering runs are saved in a separate method folder ('flowsom_meta_clusters_subclust3')
folder <- if (args$subcluster_times > 0) paste0("flowsom_meta_clusters_subclust", args$subcluster_times) else "flowsom_meta_clusters"
# Harmony runs get '_harm' appended (e.g. 'flowsom_meta_clusters_subclust3_harm', 'flowsom_harm')
harm <- if (args$harmony) "_harm" else ""
folder <- paste0(folder, harm)
for (iteration in 0:(args$iterations - 1)) {
  # Seed per iteration: reproducible, but different between iterations
  set.seed(iteration)
  start_time <- Sys.time()
  flowsom_df <- FlowSOM(as.matrix(marker_df), silent = FALSE, nClus = args$flow_nclus)
  meta_labels <- subcluster(marker_df, as.character(GetMetaclusters(flowsom_df)), args$subcluster_times,
                            args$subcluster_k)
  end_time <- Sys.time()
  elapsed <- round(as.numeric(difftime(end_time, start_time, units = "sec")), 2)

  if (args$log != "off")
    loginfo("FlowSOM nClus = %d (Duration: %.2f min)", args$flow_nclus,
            as.numeric(difftime(end_time, start_time, units = "mins")))

  # Times (FlowSOM + subclustering) are only recorded for the metaclusters
  text_path <- file.path(args$output_path, folder, "level3", "fold_times.txt")
  dir.create(dirname(text_path), recursive = TRUE, showWarnings = FALSE)
  con <- file(text_path, open = if (iteration == 0) "w" else "a")
  writeLines(sprintf("Fold %d train_time, %.2f", iteration, elapsed), con)
  close(con)

  # Node clusters are never subclustered, so they are only saved in the default run
  if (args$subcluster_times == 0) {
    clustering <- "flowsom"
    df$flowsom <- GetClusters(flowsom_df)
    greedy_assignment_save(paste0(clustering, harm))
  }

  clustering <- "flowsom_meta_clusters"
  df$flowsom_meta_clusters <- factor(meta_labels)
  greedy_assignment_save(folder)
}