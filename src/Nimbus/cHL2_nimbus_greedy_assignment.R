#### install packages ####
library(readr)
library(tidyverse)
library(Rphenograph)
library(FlowSOM) 
library(reticulate)
library(logging)
####
#### Logging setup ####
basepath = "/Users/margotchazotte/Documents/uni/PhD/pheno_benchmark"
# Ensure log directory exists
if (!dir.exists(paste(basepath, "/logs", sep = ""))
) dir.create(paste(basepath, "/logs", sep = ""), recursive = TRUE)
# Configure logging
basicConfig()
addHandler(
  writeToFile,
  file = paste(basepath, "/logs/", Sys.Date(), "_clustering_nimbus_chl2_run.log", sep = ""),
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
source_python(paste(basepath, "/code/greedy_f1_utils.py", sep = ""))

#### Utility: Safe CSV Write ####
safe_write_csv <- function(df, path) {
  dir <- dirname(path)
  if (!dir.exists(dir)) dir.create(dir, recursive = TRUE)
  write.csv(df, path)
}

#### Config ####
clustering <- c("phenograph_cluster_k40", "flowsom_cluster")
granularity <- c("level1" = "level_1_cell_type","level2" ="level_2_cell_type","level3" ="cell_type")
set.seed(42)

#### Read data ####
chl2 <- read_csv("/Users/margotchazotte/Documents/uni/PhD/pheno_benchmark/chl2/cHL_2_MIBI_quantification.csv")
chl2$sample_id[chl2$sample_id == '1.csv'] <- 'fov_1'
chl2$sample_id[chl2$sample_id == '2.csv'] <- 'fov_2'
chl2$sample_id[chl2$sample_id == '3.csv'] <- 'fov_3'
chl2$sample_id[chl2$sample_id == '4.csv'] <- 'fov_4'
chl2$sample_id[chl2$sample_id == '5.csv'] <- 'fov_5'
chl2$sample_id[chl2$sample_id == '6.csv'] <- 'fov_6'

#### Main ####
for (iteration in 1:5) {
  #### read and clean up data ####
  # read data
  nimbus_chl2 <- read_csv(paste0("/Users/margotchazotte/Documents/uni/PhD/pheno_benchmark/nimbus/nimbus_chl2/it", iteration, "/nimbus_output/nimbus_cell_table.csv", sep = ""))
  # reduce nimbus data to only contain cells also present in chl2 data, based on labels and fov
  nimbus_chl2 <- semi_join(nimbus_chl2, chl2, join_by(fov == sample_id, label == cell_id))
  
  # add celltype column to nimbus dataframe
  nimbus_chl2 <- select(chl2, cell_id, level_1_cell_type:cell_type, sample_id) %>%
    right_join(., nimbus_chl2, join_by(sample_id == fov, cell_id == label))
  
  # extended data
  #write_csv(nimbus_chl2, "/Users/margotchazotte/Documents/uni/PhD/pheno_benchmark/chl2/nimbus_chl2_clustered.csv")
  #nimbus_chl2 <- read_csv("/Users/margotchazotte/Documents/uni/PhD/pheno_benchmark/chl2/nimbus_chl2_clustered.csv")
  ####
  
  #### run phenograph ####
  start_time <- Sys.time()
  pheno_nimbus_chl2 <- select(nimbus_chl2, CD14:CD3, -`Histone H3`) %>%
    Rphenograph(k = 40)
  nimbus_chl2$phenograph_cluster_k40 <- factor(membership(pheno_nimbus_chl2[[2]]))
  
  end_time <- Sys.time()
  loginfo("Rphenograph k = 40 clustering (Duration: %s)", paste0(round(as.numeric(difftime(end_time, start_time, units = "mins")), 2), " minutes"))
  
  #### run FlowSOM clustering ####
  start_time <- Sys.time()
  flowsom_nimbus_chl2 <- FlowSOM(as.matrix(select(nimbus_chl2, CD14:CD3, -`Histone H3`)), silent= FALSE, nClus = 12)
  
  end_time <- Sys.time()
  loginfo("FlowSOM clustering (Duration: %s)", paste0(round(as.numeric(difftime(end_time, start_time, units = "mins")), 2), " minutes"))
  
  nimbus_chl2$flowsom_cluster <- GetClusters(flowsom_nimbus_chl2)
  
  ####
  #### greedy assignment ####
  for (i in clustering) {
    for (k in granularity) {
      results <- greedy_f1_score(nimbus_chl2, k, i, tie_strategy = 'random')
      output <- select(nimbus_chl2, cell_id:CD3, i)
      output$predicted_phenotype <- as.vector(results$mapped_predictions)
      output = rename(output, true_phenotype = k)
      safe_write_csv(output, paste("/Users/margotchazotte/Documents/uni/PhD/pheno_benchmark/nimbus/nimbus_chl2/nimbus_", i, 
                                   "/", names(granularity[granularity == k]),
                                   "/predictions_", iteration,".csv", 
                                   sep = ""))
    }
  }
}
