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
  file = paste(basepath, "/logs/", Sys.Date(), "_clustering_nimbus_IMMUcan_run.log", sep = ""),
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
immucan <- read_csv("/Users/margotchazotte/Documents/uni/PhD/pheno_benchmark/immucan/immucan_clustered.csv")

#### Main ####
for (iteration in 1:5) {
  #### read and clean up data ####
  # read data
  nimbus_immucan <- read_csv(paste0(basepath, "/nimbus/IMMUcan/it", iteration, "/nimbus_output/nimbus_cell_table.csv", sep = ""))
  # reduce nimbus data to only contain cells also present in immucan data, based on labels and fov
  nimbus_immucan <- semi_join(nimbus_immucan, immucan, join_by(fov == sample_id, label == cell_id))
  
  # add celltype column to nimbus dataframe
  nimbus_immucan <- select(immucan, cell_id, level_1_cell_type:cell_type, sample_id) %>%
    right_join(., nimbus_immucan, join_by(sample_id == fov, cell_id == label))
  
  # extended data
  #write_csv(nimbus_immucan, "/Users/margotchazotte/Documents/uni/PhD/pheno_benchmark/immucan/nimbus_immucan_clustered.csv")
  #nimbus_immucan <- read_csv("/Users/margotchazotte/Documents/uni/PhD/pheno_benchmark/immucan/nimbus_immucan_clustered.csv")
  ####
  
  #### run phenograph ####
  start_time <- Sys.time()
  pheno_nimbus_immucan <- select(nimbus_immucan, MPO:cleavedPARP) %>%
    Rphenograph(k = 40)
  nimbus_immucan$phenograph_cluster_k40 <- factor(membership(pheno_nimbus_immucan[[2]]))
  
  end_time <- Sys.time()
  loginfo("Rphenograph k = 40 clustering (Duration: %s)", paste0(round(as.numeric(difftime(end_time, start_time, units = "mins")), 2), " minutes"))

  #### run FlowSOM clustering ####
  start_time <- Sys.time()
  flowsom_nimbus_immucan <- FlowSOM(as.matrix(select(nimbus_immucan, MPO:cleavedPARP)), silent= FALSE, nClus = 15)
  
  end_time <- Sys.time()
  loginfo("FlowSOM clustering (Duration: %s)", paste0(round(as.numeric(difftime(end_time, start_time, units = "mins")), 2), " minutes"))
  
  nimbus_immucan$flowsom_cluster <- GetClusters(flowsom_nimbus_immucan)
  
  ####
  #### greedy assignment ####
  for (i in clustering) {
    for (k in granularity) {
      results <- greedy_f1_score(nimbus_immucan, k, i, tie_strategy = 'random')
      output <- select(nimbus_immucan, cell_id:cleavedPARP, i)
      output$predicted_phenotype <- as.vector(results$mapped_predictions)
      output = rename(output, true_phenotype = k)
      safe_write_csv(output, paste(basepath, "/Users/margotchazotte/Documents/uni/PhD/pheno_benchmark/nimbus/IMMUcan/nimbus_", i, 
                                   "/", names(granularity[granularity == k]),
                                   "/predictions_", iteration,".csv", 
                                   sep = ""))
    }
  }
}
