#!/usr/bin/env Rscript

# Load libraries
suppressPackageStartupMessages({
  suppressWarnings(library(argparse))
  suppressWarnings(library(readr))
  suppressWarnings(library(tidyverse))
  suppressWarnings(library(Rphenograph))
  suppressWarnings(library(FlowSOM))
  suppressWarnings(library(FuseSOM))
  suppressWarnings(library(reticulate))
  suppressWarnings(library(logging))
})

# Create parser
parser <- ArgumentParser(description = "Cluster Nimbus processed data and with Phenograph, FlowSOM and FuseSOM")

# Define arguments
parser$add_argument("-gt", "--ground_truth", dest = "ground_truth", required = TRUE, help = "Path to input quantification GT CSV file")
parser$add_argument("-i", "--input", dest = "input", required = TRUE,  help = "Path to nimbus output folder containing one folder per iteration with quantification csv. EG immucan/nimbus/outputs")
parser$add_argument("-m", "--markers",dest = "markers",  nargs = "+", help = "List of marker columns to use")
parser$add_argument("-o", "--output", dest = "output_path", required = TRUE, help = "Path to output folder")
parser$add_argument("--remove_str", dest = "remove_str", required = FALSE, default = "None", type = "character", help = "String to remove from sample_id column of quantification data to match fov column of nimbus data.")
parser$add_argument("-k", "--k_phenograph", dest = "k_phenograph", nargs = "+", type = "integer", default = c(40),
                    help = "List of k values for PhenoGraph clustering (default: 40)")
parser$add_argument("-fuse","--fuse_clusters", dest = "fuse_clusters", nargs = "+", type = "integer", required = TRUE,
                    help = "List of cluster numbers for FuseSOM")
parser$add_argument("-flow", "--flow_clusters", dest = "flow_nclus", type = "integer", required = TRUE,
                    help = "Number of metaclusters for FlowSOM")
parser$add_argument('-l', '--log', dest='log', required= FALSE, default='off', choices=c('short', 'long', 'off'), help='Logging level: short, long, or off (default: long)' )
parser$add_argument('-it', '--iterations', dest='iterations', type="integer", required= FALSE, default=5, help='Number of iterations to run (default: 5)')
parser$add_argument("-n", "--normalization", dest = "normalization", default = FALSE, help = "Should row normalization to 1 be performed before clustering. Default FALSE")

# Parse arguments
args <- parser$parse_args()

#### Logging setup ####
if (args$log != "off")
  # Ensure log directory exists
  if (!dir.exists(paste0(args$output_path, "/logs"))) 
    dir.create(paste0(args$output_path, "/logs"), recursive = TRUE)
# Configure logging
basicConfig()
addHandler(
  writeToFile,
  file = paste0(args$output_path, "/logs/", Sys.Date(), "_Nimbus_clustering_R.log"),
  level = "INFO",
  formatter = function(record) {
    sprintf("%s - %s - %s",
            format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
            record$levelname,
            record$msg)
  }
)

#### Python setup ####
use_condaenv('celllens-env')
source_python("/Users/margotchazotte/Documents/uni/PhD/pheno_benchmark/code/greedy_f1_utils.py")

if (!exists("greedy_f1_score")) {
  stop("Failed to import greedy_f1_score from Python. Check Python environment.")
}

#### Utility ####
safe_write_csv <- function(df, path) {
  dir <- dirname(path)
  if (!dir.exists(dir)) dir.create(dir, recursive = TRUE)
  write.csv(df, path)
}

greedy_assignment_save <- function() {
  for (k in granularity) {
    results <- greedy_f1_score(nimbus_df, k, clustering, tie_strategy = 'random')
    output <- select(nimbus_df, all_of(original_cols), clustering)
    output$predicted_phenotype <- as.vector(results$mapped_predictions)
    output = rename(output, true_phenotype = k)
    safe_write_csv(output, paste0(args$output_path, "/", clustering, 
                                  "/", names(granularity[granularity == k]),
                                  "/predictions_", iteration,".csv"))
  }
}

#### Config ####
granularity <- c("level1" = "level_1_cell_type","level2" ="level_2_cell_type","level3" ="cell_type")

#### Read data ####
gt <- read_csv(args$ground_truth, show_col_types = FALSE)
gt$sample_id <- as.character(gt$sample_id)

if (args$remove_str != "None") {
  gt$sample_id <- str_replace_all(gt$sample_id, args$remove_str, "")
}

#### Main ####
for (iteration in 1:args$iterations) {
  #### read and clean up data ####
  # read nimbus data
  nimbus_df <- read_csv(paste0(args$input, "/iteration_", iteration, "/nimbus_cell_table.csv"), show_col_types = FALSE)
  nimbus_df$fov <- as.character(nimbus_df$fov)
  # reduce nimbus data to only contain cells also present in chl2 data, based on labels and fov
  nimbus_df <- semi_join(nimbus_df, gt, join_by(fov == sample_id, label == cell_id))
  # add celltype column to nimbus dataframe
  nimbus_df <- select(gt, cell_id, level_1_cell_type:cell_type, sample_id) %>%
    right_join(., nimbus_df, join_by(sample_id == fov, cell_id == label))
  ####
  
  original_cols <- colnames(nimbus_df)
  
  marker_df <- select(nimbus_df, all_of(args$markers))
  
  if (args$normalization == TRUE) {
    marker_df <- as.data.frame(t(apply(marker_df, 1, function(x) x / sum(x))))
  }
  
  #### Run PhenoGraph at different k values ####
  for (k_val in args$k_phenograph) {
    clustering <- paste0('phenograph_', k_val)
    start_time <- Sys.time()
    pheno_df <- marker_df %>% Rphenograph(k = k_val)
    end_time <- Sys.time()
    elapsed <- round(as.numeric(difftime(end_time, start_time, units = "sec")), 2)
    nimbus_df[[clustering]] <- factor(membership(pheno_df[[2]]))
    
    if (args$log != "off") {
      loginfo("Rphenograph k = %d clustering (Duration: %s)",
              k_val, 
              paste0(round(as.numeric(difftime(end_time, start_time, units = "mins")), 2), " minutes"))
    }

    # Write time to txt file
    text_path <- file.path(args$output_path, clustering, "fold_times.txt")
    dir.create(dirname(text_path), recursive = TRUE, showWarnings = FALSE)
    con <- file(text_path, open = if (iteration == 1) "w" else "a")
    writeLines(sprintf("Fold %d phenograph_time, %.2f", iteration, elapsed), con)
    close(con)
    
    greedy_assignment_save()
  }
  
  #### Run FlowSOM clustering ####
  start_time <- Sys.time()
  flowsom_df <- FlowSOM(as.matrix(marker_df), silent= FALSE, nClus = args$flow_nclus)
  end_time <- Sys.time()
  elapsed <- round(as.numeric(difftime(end_time, start_time, units = "sec")), 2)
  
  if (args$log != "off") {
    loginfo("FlowSOM clustering (Duration: %s)",
            paste0(round(as.numeric(difftime(end_time, start_time, units = "mins")), 2), " minutes"))
  }
  
  # Write time to txt file
  for (cl in c("flowsom", "flowsom_meta_clusters")){
    text_path <- file.path(args$output_path, cl, "fold_times.txt")
    dir.create(dirname(text_path), recursive = TRUE, showWarnings = FALSE)
    con <- file(text_path, open = if (iteration == 1) "w" else "a")
    writeLines(sprintf("Fold %d flowsom_time, %.2f", iteration, elapsed), con)
    close(con)
  }

  clustering = "flowsom"
  nimbus_df$flowsom <- GetClusters(flowsom_df)
  greedy_assignment_save()
  
  clustering = "flowsom_meta_clusters"
  nimbus_df$flowsom_meta_clusters <- GetMetaclusters(flowsom_df)
  greedy_assignment_save()
  ####
  
  #### Run FuseSOM clustering ####
  for (fuse_clus in args$fuse_clusters) {
    clustering <- paste0('FuseSOM_', fuse_clus)
    
    start_time <- Sys.time()
    fuse_df <- marker_df %>% runFuseSOM(numClusters = fuse_clus)
    end_time <- Sys.time()
    elapsed <- round(as.numeric(difftime(end_time, start_time, units = "sec")), 2)

  text_path <- file.path(args$output_path, clustering, "fold_times.txt")
  dir.create(dirname(text_path), recursive = TRUE, showWarnings = FALSE)
  con <- file(text_path, open = if (iteration == 1) "w" else "a")
  writeLines(sprintf("Fold %d FuseSOM_time, %.2f", iteration, elapsed), con)
  close(con)
    
    nimbus_df[[clustering]] <- fuse_df$clusters
    
    if (args$log != "off") {
      loginfo("FuseSOM numClusters = %d clustering (Duration: %s)", fuse_clus, 
              paste0(round(as.numeric(difftime(end_time, start_time, units = "mins")), 2), " minutes"))
    }
    
    greedy_assignment_save()
  }
}

#exemplar run 
#chmod +x Nimbus_R_clustering.R
# ./Nimbus_R_clustering.R -gt ../chl2/cHL_2_MIBI_quantification.csv -i ../nimbus/nimbus_chl2 -m CD163 CD45RO CD28 'Na-K ATPase' -o ../nimbus/nimbus_chl2 -fuse 12 40 -flow 12

#parameters for chl2: 
# -markers CD45 CD20 'pSLP-76' 'SLP-76' 'anti-H2AX (pS139)' CD163 CD45RO CD28 'CD153 (CD30L)' Lag3 CD4 CD11c CD56 FoxP3 GATA3 'Granzyme B' 'PD-L1' CD16 'Ki-67' 'PD-1' 'Pax-5' Tox CD161 CD68 'B2-Microglobulin' CD8 CD3 HLA1 CD15 Tbet CD14 CD123 CXCR5 CD45RA 'HLA-DR' CD57 'IL-10' CD30 TIM3 RORgT TCRgd CD86 CD25 'Na-K ATPase'
# -fuse 12
# -flow 12
# --remove_str '.csv'

#parameters for immucan:
# -markers MPO SMA CD16 CD38 HLADR CD27 CD15 CD45RA CD163 B2M CD20 CD68 Ido1 CD3 LAG3 CD11c PD1 PDGFRb CD7 GrzB PDL1 TCF7 CD45RO FOXP3 ICOS CD8a CarbonicAnhydrase CD33 Ki67 VISTA CD40 CD4 CD14 Ecad CD303 CD206 cleavedPARP
# -fuse 15
# -flow 15