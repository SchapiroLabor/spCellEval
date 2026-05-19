#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  suppressWarnings(library(argparse))
  suppressWarnings(library(readr))
  suppressWarnings(library(tidyverse))
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
parser$add_argument("-m", "--markers", dest = "markers", nargs = "+", required = TRUE,
                    help = "Marker columns to use")
parser$add_argument("-o", "--output", dest = "output_path", required = TRUE,
                    help = "Path to output folder")
parser$add_argument("-fuse", "--fuse_clusters", dest = "fuse_clusters", nargs = "+",
                    type = "integer", required = TRUE,
                    help = "List of cluster numbers for FuseSOM")
parser$add_argument("-l", "--log", dest = "log", default = "off",
                    choices = c("short", "long", "off"), help = "Logging level (default: off)")
parser$add_argument("-it", "--iterations", dest = "iterations", type = "integer", default = 5,
                    help = "Number of iterations (default: 5)")
parser$add_argument("-n", "--normalization", dest = "normalization", default = FALSE,
                    help = "Row normalization before clustering (default: FALSE)")
args <- parser$parse_args()

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
marker_df <- select(df, all_of(args$markers))
if (args$normalization == TRUE)
  marker_df <- as.data.frame(t(apply(marker_df, 1, function(x) x / sum(x))))

#### Run ####
for (iteration in 1:args$iterations) {
  for (fuse_clus in args$fuse_clusters) {
    clustering <- paste0("FuseSOM_", fuse_clus)

    start_time <- Sys.time()
    fuse_df <- marker_df %>% runFuseSOM(numClusters = fuse_clus)
    end_time <- Sys.time()
    elapsed <- round(as.numeric(difftime(end_time, start_time, units = "sec")), 2)

    if (args$log != "off")
      loginfo("FuseSOM numClusters = %d (Duration: %.2f min)", fuse_clus,
              as.numeric(difftime(end_time, start_time, units = "mins")))

    text_path <- file.path(args$output_path, clustering, "fold_times.txt")
    dir.create(dirname(text_path), recursive = TRUE, showWarnings = FALSE)
    con <- file(text_path, open = if (iteration == 1) "w" else "a")
    writeLines(sprintf("Fold %d FuseSOM_time, %.2f", iteration, elapsed), con)
    close(con)

    df[[clustering]] <- fuse_df$clusters
    greedy_assignment_save()
  }
}