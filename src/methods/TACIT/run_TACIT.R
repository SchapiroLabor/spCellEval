library(Seurat)
library(class)
library(segmented)
library(readr)
library(tidyverse)
library(future)
library(future.apply)
library(parallelly)
library(TACIT)
library(argparse)

script_dir <- dirname(normalizePath(
  sub("--file=", "", commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))])
))
source(file.path(script_dir, "..", "utils", "fold_timer.R"))
START <- Sys.time()  # fold_times.csv: the run is timed from here, after the library() calls

run_TACIT <- function(n, input_path, separate_col, scaling, log1p, decision_matrix_path, r, p, output_path) {
  data <- read_csv(input_path)
  
  CELLxFEATURE <- data %>% 
    select(cell_id, everything()) %>%
    select(- (separate_col:last_col()))
  feature_cols <- setdiff(names(CELLxFEATURE), "cell_id")
  if (!is.null(scaling)) {
    CELLxFEATURE[feature_cols] <- CELLxFEATURE[feature_cols] * scaling
  }
  if (log1p) {
    CELLxFEATURE[feature_cols] <- log1p(CELLxFEATURE[feature_cols])
  }
  data <- data %>%
    rename(true_phenotype = cell_type)
  
  TYPExMARKER <- read_csv(decision_matrix_path)
  fold_times <- c()
  for(i in 1:n) {
    start_time <- Sys.time()
    TACIT_result <- TACIT(CELLxFEATURE, TYPExMARKER, r=r, p=p)
    end_time <- Sys.time()
    fold_times[as.character(i)] <- as.numeric(difftime(end_time, start_time, units = "secs"))
    
    data <- data %>%
      mutate(predicted_phenotype = TACIT_result$TACIT) %>%
      mutate(predicted_phenotype = case_when(
        predicted_phenotype == "Others" ~ "undefined",
        TRUE ~ predicted_phenotype
      ))
    
    output_file <- file.path(output_path, paste0("predictions_", i, ".csv"))
    write_csv(data, output_file)
  }
  write_fold_times(fold_times, START, output_path)
}

main <- function() {
  parser <- ArgumentParser(description = "Run TACIT phenotype predictions")
  
  parser$add_argument("--input_path", type="character", required=TRUE,
                      help="Path to input quantification CSV file")
  parser$add_argument("--decision_matrix_path", type="character", required=TRUE,
                      help="Path to marker decision matrix CSV file")
  parser$add_argument("--separate_col", type="character", required=TRUE,
                      help="Col that separates marker columns from remaining cols. Input like this from terminal: 'colname' ")
  parser$add_argument("--scaling", type="double", default=NULL,
                      help="Scaling factor for the data. If not provided, no scaling is applied")
  parser$add_argument("--log1p", action='store_true', help="Apply log1p transformation to the data")
  parser$add_argument("-n", "--iterations", type="integer", default=5,
                      help="Number of times to run the TACIT prediction [default %(default)s]")
  parser$add_argument("-r", type="integer", default=10,
                      help="resolution. The higher the resolution, the greaterde the number of microclusters")
  parser$add_argument("-p", type="integer", default=10,
                      help="Dimension. TNumber of dimensions used for microclusters")
  parser$add_argument("--output_path", type="character", required=TRUE,
                      help="Directory to save prediction CSV files")
  args <- parser$parse_args()
  options(future.globals.maxSize = 6 * 1024^3)
  run_TACIT(args$iterations, args$input_path,args$separate_col, args$scaling, args$log1p, args$decision_matrix_path, args$r, args$p, args$output_path)
}

main()