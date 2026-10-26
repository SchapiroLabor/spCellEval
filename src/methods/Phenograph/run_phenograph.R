#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  suppressWarnings(library(argparse))
  suppressWarnings(library(readr))
  suppressWarnings(library(dplyr))
  # Rphenograph's C++ code needs Rcpp loaded, but its NAMESPACE does not import Rcpp
  # (otherwise: "function 'enterRNGScope' not provided by package 'Rcpp'")
  suppressWarnings(library(Rcpp))
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
                    type = "integer", default = 50,
                    help = "k for PhenoGraph (default: 50, the Rphenograph default)")
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
parser$add_argument("-sck", "--subcluster_k", dest = "subcluster_k", type = "integer", default = 25,
                    help = "k used for subclustering; larger k gives fewer subclusters (default: 25)")
parser$add_argument("-sp", "--subcluster_purity", dest = "subcluster_purity", type = "double",
                    help = "Clusters whose majority cell_type fraction is >= this are not subclustered further, -sc is then the max depth. Uses the ground truth (oracle setting) (default: off)")
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
# Each round re-runs PhenoGraph on every cluster (kNN graph rebuilt on the subset), new labels are
# '<parent>,<child>'. k is the subclustering k (-sck, larger k -> fewer subclusters).
# Rphenograph needs at least k + 2 cells, clusters with fewer cells are kept.
# With purity, clusters whose cell_type purity is >= purity are kept.
subcluster <- function(marker_df, labels, times, k, true_labels, purity) {
  for (round in seq_len(times)) {
    counts <- table(labels)
    for (cl in names(counts[counts >= k + 2])) {
      idx <- which(labels == cl)
      if (!is.null(purity) && cluster_purity(true_labels[idx]) >= purity) next
      sub <- Rphenograph(marker_df[idx, ], k = k)
      labels[idx] <- paste0(cl, ",", membership(sub[[2]]))
    }
  }
  labels
}

#### Run ####
k_val <- args$k_phenograph
sub_k <- if (is.null(args$subcluster_k)) k_val else args$subcluster_k
clustering <- paste0("phenograph_", k_val)
# Subclustering runs are saved in a separate method folder (e.g. 'phenograph_30_subclust3'),
# with the subclustering k appended if it differs from k (e.g. 'phenograph_30_subclust3_k60')
folder <- if (args$subcluster_times > 0) paste0(clustering, "_subclust", args$subcluster_times) else clustering
if (args$subcluster_times > 0 && sub_k != k_val) folder <- paste0(folder, "_k", sub_k)
# and the purity if set (e.g. 'phenograph_30_subclust3_purity0_9')
if (args$subcluster_times > 0) folder <- paste0(folder, purity_suffix(args$subcluster_purity))
# Harmony runs get '_harm' appended (e.g. 'phenograph_30_subclust3_harm')
if (args$harmony) folder <- paste0(folder, "_harm")
for (iteration in 0:(args$iterations - 1)) {
  # Seed per iteration: reproducible, but different between iterations
  set.seed(iteration)
  start_time <- Sys.time()
  pheno_df <- marker_df %>% Rphenograph(k = k_val)
  labels <- subcluster(marker_df, as.character(membership(pheno_df[[2]])),
                       args$subcluster_times, sub_k, df$cell_type, args$subcluster_purity)
  end_time <- Sys.time()
  elapsed <- round(as.numeric(difftime(end_time, start_time, units = "sec")), 2)

  df[[clustering]] <- factor(labels)

  if (args$log != "off")
    loginfo("Rphenograph k = %d (Duration: %.2f min)", k_val,
            as.numeric(difftime(end_time, start_time, units = "mins")))

  text_path <- file.path(args$output_path, folder, "level3", "fold_times.txt")
  dir.create(dirname(text_path), recursive = TRUE, showWarnings = FALSE)
  con <- file(text_path, open = if (iteration == 0) "w" else "a")
  writeLines(sprintf("Fold %d train_time, %.2f", iteration, elapsed), con)
  close(con)

  greedy_assignment_save(folder)
}