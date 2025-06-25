#### install packages ####
library(readr)
library(tidyverse)
library(umap)
library(ggplot2)
library(Rphenograph)
library(FlowSOM) 
library(reticulate)
library(logging)
####
#### Logging setup ####
# Ensure log directory exists
if (!dir.exists("/Users/margotchazotte/Documents/uni/PhD/pheno_benchmark/logs")) dir.create("/Users/margotchazotte/Documents/uni/PhD/pheno_benchmark/logs", recursive = TRUE)
# Configure logging
basicConfig()
addHandler(
  writeToFile,
  file = paste("/Users/margotchazotte/Documents/uni/PhD/pheno_benchmark/logs/", Sys.Date(), "_clustering_chl2_run.log", sep = ""),
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

#### Utility: Safe CSV Write ####
safe_write_csv <- function(df, path) {
  dir <- dirname(path)
  if (!dir.exists(dir)) dir.create(dir, recursive = TRUE)
  write.csv(df, path)
}

#### Config ####
clustering <- c("phenograph_20","phenograph_30", "phenograph_40",
                "phenograph_80","flowsom","flowsom_meta_clusters")
granularity <- c("level1" = "level_1_cell_type","level2" ="level_2_cell_type","level3" ="cell_type")

#### Read data ####
chl2 <- read_csv("/Users/margotchazotte/Documents/uni/PhD/pheno_benchmark/chl2/cHL_2_MIBI_quantification.csv")

#### Main ####
for (iteration in 1:5) {
  set.seed(iteration*42)
  #### Run PhenoGraph at different k values ####
  for (k_val in c(20, 30, 40, 80)) {
    start_time <- Sys.time()
    
    pheno_chl2 <- select(chl2, CD45:`Na-K ATPase`, -`Histone H3`, -dsDNA) %>%
      Rphenograph(k = k_val)
    chl2[[paste0("phenograph_", k_val)]] <- factor(membership(pheno_chl2[[2]]))
    
    end_time <- Sys.time()
    loginfo("Rphenograph k = %d clustering (Duration: %s)",
            k_val, 
            paste0(round(as.numeric(difftime(end_time, start_time, units = "mins")), 2), " minutes"))
  }
  
  #### Run FlowSOM clustering ####
  start_time <- Sys.time()
  flowsom_chl2 <- FlowSOM(as.matrix(select(chl2, CD45:`Na-K ATPase`, -`Histone H3`, -dsDNA)), silent= FALSE, nClus = 12)
  
  end_time <- Sys.time()
  loginfo("FlowSOM clustering (Duration: %s)", 
          paste0(round(as.numeric(difftime(end_time, start_time, units = "mins")), 2), " minutes"))
  
  chl2$flowsom <- GetClusters(flowsom_chl2)
  chl2$flowsom_meta_clusters <- GetMetaclusters(flowsom_chl2)
  ####
  
  #### greedy assignment ####
  for (i in clustering) {
    for (k in granularity) {
      loginfo("Assigning %s to %s", i, k)
      results <- greedy_f1_score(chl2, k, i, tie_strategy = 'random')
      loginfo("F1 macro: %f", results$f1_macro)
      loginfo("F1 weighted: %f", results$f1_weighted)
      output <- select(chl2, CD45:cell_type, i)
      output$predicted_phenotype <- as.vector(results$mapped_predictions)
      output = rename(output, true_phenotype = k)
      safe_write_csv(output, paste("/Users/margotchazotte/Documents/uni/PhD/pheno_benchmark/chl2/replace_chl2_if_successfull/", i, 
                                   "/", names(granularity[granularity == k]),
                                   "/predictions_", iteration,".csv", 
                                   sep = ""))
    }
  }
}
