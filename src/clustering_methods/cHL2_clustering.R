#### install packages ####
library(readr)
library(tidyverse)
library(umap)
library(ggplot2)
library(Rphenograph)
library(FlowSOM) 
library(reticulate)
####
#### prep ####
# Set the path to your Python interpreter (optional but recommended)
use_condaenv('celllens-env')
# Source the Python script
source_python("greedy_f1_utils.py")
#writing csv even if folder doesnt exist
safe_write_csv <- function(df, path) {
  dir <- dirname(path)
  if (!dir.exists(dir)) dir.create(dir, recursive = TRUE)
  write.csv(df, path)
}

clustering <- c("phenograph_20","phenograph_30", "phenograph_40",
                "phenograph_80","flowsom","flowsom_meta_clusters")
granularity <- c("level1" = "level_1_cell_type","level2" ="level_2_cell_type","level3" ="cell_type")

# read data
chl2 <- read_csv("/Users/margotchazotte/Documents/uni/PhD/pheno_benchmark/chl2/cHL_2_MIBI_quantification.csv")
####
for (iteration in 1:5) {
  set.seed(iteration*42)
  #### run phenograph ####
  # run phenograph on chl2 data and overlay celltypes with clusters
  pheno_chl2 <- select(chl2, CD45:`Na-K ATPase`, -`Histone H3`, -dsDNA) %>%
    Rphenograph(k = 30)
  chl2$phenograph_30 <- factor(membership(pheno_chl2[[2]])) #results in 129 clusters
  
  pheno_chl2 <- select(chl2, CD45:`Na-K ATPase`, -`Histone H3`, -dsDNA) %>%
    Rphenograph(k = 40)
  chl2$phenograph_40 <- factor(membership(pheno_chl2[[2]])) #results in 128 clusters
  
  pheno_chl2 <- select(chl2, CD45:`Na-K ATPase`, -`Histone H3`, -dsDNA) %>%
    Rphenograph(k = 20)
  chl2$phenograph_20 <- factor(membership(pheno_chl2[[2]])) #results in 137 clusters
  
  pheno_chl2 <- select(chl2, CD45:`Na-K ATPase`, -`Histone H3`, -dsDNA) %>%
    Rphenograph(k = 80)
  chl2$phenograph_80 <- factor(membership(pheno_chl2[[2]])) #results in xx clusters

  #### run FlowSOM clustering ####
  # chl2 data 
  flowsom_chl2 <- FlowSOM(as.matrix(select(chl2, CD45:`Na-K ATPase`, -`Histone H3`, -dsDNA)), silent= FALSE, nClus = 12)
  chl2$flowsom <- GetClusters(flowsom_chl2)
  chl2$flowsom_meta_clusters <- GetMetaclusters(flowsom_chl2)
  ####
  
  #### greedy assignment ####
  # # Show results
  # cat("F1 macro:", results$f1_macro, "\n")
  # cat("f1_weighted:", results$f1_weighted, "\n")
  # cat("ARI:", results$ari, "\n")
  # cat("NMI:", results$nmi, "\n")
  # cat("mcc:", results$mcc, "\n")
  # cat("kappa:", results$kappa, "\n")
  # cat("Accuracy:", results$accuracy, "\n")
  for (i in clustering) {
    for (k in granularity) {
      cat("Iteration", iteration, ":", "assigning", i, "to", k, "\n")
      results <- greedy_f1_score(chl2, k, i, tie_strategy = 'random')
      cat("F1 macro:", results$f1_macro, "\n")
      cat("f1_weighted:", results$f1_weighted, "\n")
      output <- select(chl2, CD45:cell_type)
      output$predicted_phenotype <- as.vector(results$mapped_predictions)
      output = rename(output, true_phenotype = k)
      safe_write_csv(output, paste("/Users/margotchazotte/Documents/uni/PhD/pheno_benchmark/chl2/", i, 
                                   "/", names(granularity[granularity == k]),
                                   "/predictions_", iteration,".csv", 
                                   sep = ""))
    }
  }
}
