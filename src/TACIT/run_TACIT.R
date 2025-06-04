library(Seurat)
library(class)
library(segmented)
library(readr)
library(tidyverse)
library(TACIT)
library(argparse)

run_TACIT <- function(n, input_path, decision_matrix_path, r, p, output_path) {
  data <- read_csv(input_path)
  
  CELLxFEATURE <- data %>% 
    select(cell_id, everything()) %>%
    select(- (DNA1:last_col()))
  
  data <- data %>%
    rename(true_phenotype = cell_type)
  
  TYPExMARKER <- read_csv(decision_matrix_path)
  
  for(i in 1:n) {
    TACIT_result <- TACIT(CELLxFEATURE, TYPExMARKER, r=r, p=p)
    
    data <- data %>%
      mutate(predicted_phenotype = TACIT_result[[3]]$mem) %>%
      mutate(predicted_phenotype = case_when(
        predicted_phenotype == "Others" ~ "undefined",
        TRUE ~ predicted_phenotype
      ))
    
    output_file <- file.path(output_path, paste0("predictions_", i, ".csv"))
    write_csv(data, output_file)
  }
}

main <- function() {
  parser <- ArgumentParser(description = "Run TACIT phenotype predictions")
  
  parser$add_argument("--input_path", type="character", required=TRUE,
                      help="Path to input quantification CSV file")
  parser$add_argument("--decision_matrix_path", type="character", required=TRUE,
                      help="Path to marker decision matrix CSV file")
  parser$add_argument("-n", "--iterations", type="integer", default=5,
                      help="Number of times to run the TACIT prediction [default %(default)s]")
  parser$add_argument("-r", type="integer", default=10,
                      help="resolution. The higher the resolution, the greaterde the number of microclusters")
  parser$add_argument("-p", type="integer", default=10,
                      help="Dimension. TNumber of dimensions used for microclusters")
  parser$add_argument("--output_path", type="character", required=TRUE,
                      help="Directory to save prediction CSV files")
  args <- parser$parse_args()
  
  run_TACIT(args$iterations, args$input_path, args$decision_matrix_path, args$r, args$p, args$output_path)
}

main()

