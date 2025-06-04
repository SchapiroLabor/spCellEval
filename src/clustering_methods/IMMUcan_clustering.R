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
immucan <- read_csv("/Users/margotchazotte/Documents/uni/PhD/pheno_benchmark/immucan/IMMUcan_quantification.csv")
####

for (iteration in 1:5) {
  set.seed(iteration*42)
  #### read and clean up data ####
  
  #unique(immucan$cell_type)
  
  # extended data
  #write_csv(immucan, "/Users/margotchazotte/Documents/uni/PhD/pheno_benchmark/immucan/immucan_clustered.csv")
  #immucan <- read_csv("/Users/margotchazotte/Documents/uni/PhD/pheno_benchmark/immucan/immucan_clustered.csv")
  ####
  
  # #### run umap ####
  # # run umap on a subset of the immucan data
  # set.seed(123)
  # sample_immu <- immucan[sample(1:nrow(immucan), 10000),]
  # 
  # umap <- select(sample_immu, MPO:cleavedPARP, -HistoneH3) %>% 
  #   umap::umap( n_neighbors = 15, min_dist = 0.1, metric = "euclidean")
  # sample_immu$umap1 <- umap$layout[, 1]
  # sample_immu$umap2 <- umap$layout[, 2]
  # 
  # plot(sample_immu$umap1, sample_immu$umap2)
  # ggplot2::ggplot(sample_immu, aes(umap1, umap2, color = cell_type)) + geom_point()
  # 
  # # run umap on full immucan data
  # umap <- select(immucan, MPO:cleavedPARP, -HistoneH3) %>% 
  #   umap(n_neighbors = 15, min_dist = 0.1, metric = "euclidean")
  # immucan$umap1 <- umap$layout[ ,1]
  # immucan$umap2 <- umap$layout[ ,2]
  # 
  # immucan[sample(1:nrow(immucan), 50000), ] %>%
  #   ggplot(aes(umap1, umap2, color = cell_type)) +
  #   geom_point(size = 0.1) +
  #   ggtitle("immucan UMAP")
  # ####
  
  #### run phenograph ####
  # run phenograph on immucan data and overlay celltypes with clusters
  pheno_immucan <- select(immucan, MPO:cleavedPARP, -HistoneH3) %>%
    Rphenograph(k = 30)
  immucan$phenograph_30 <- factor(membership(pheno_immucan[[2]])) #results in 129 clusters
  
  pheno_immucan <- select(immucan, MPO:cleavedPARP, -HistoneH3) %>%
    Rphenograph(k = 40)
  immucan$phenograph_40 <- factor(membership(pheno_immucan[[2]])) #results in 128 clusters
  
  pheno_immucan <- select(immucan, MPO:cleavedPARP, -HistoneH3) %>%
    Rphenograph(k = 20)
  immucan$phenograph_20 <- factor(membership(pheno_immucan[[2]])) #results in 137 clusters
  
  pheno_immucan <- select(immucan, MPO:cleavedPARP, -HistoneH3) %>%
    Rphenograph(k = 80)
  immucan$phenograph_80 <- factor(membership(pheno_immucan[[2]])) #results in xx clusters
  
  # #plot clusters on umap
  # immucan[sample(1:nrow(immucan), 10000),] %>%
  #   ggplot(aes(x=umap1, y=umap2, col=phenograph_cluster_k80, shape = cell_type)) +
  #   scale_shape_manual(values = c(0:17)) +
  #   geom_point(size = 1) +
  #   xlim(c(-25, 70)) +
  #   ylim(c(-20, 35))+
  #   ggtitle("immucan phenograph k80 data") +
  #   theme_bw()
  # #plot clusters v cell_type
  # immucan %>%
  #   #filter(pheno40_cell_type != "Cancer") %>%
  # ggplot(aes(pheno40_cell_type, fill = cell_type)) +
  #   geom_bar(position = "stack") +
  #   ggtitle("immucan phenograph k40 clusters") +
  #   theme_bw() +
  #   theme(axis.text.x = element_text(angle = 45,hjust = 1)) # 
  
  #### run FlowSOM clustering ####
  # immucan data 
  
  flowsom_immucan <- FlowSOM(as.matrix(select(immucan, MPO:cleavedPARP, -HistoneH3)), silent= FALSE, nClus = 12)
  immucan$flowsom <- GetClusters(flowsom_immucan)
  immucan$flowsom_meta_clusters <- GetMetaclusters(flowsom_immucan)
  # #get summary
  # FlowSOMmary(fsom = flowsom_immucan, plotFile = "FlowSOMmary_immucan_.pdf")
  # 
  # ggplot(immucan, aes(flowsom_cluster, fill = cell_type)) +
  #   geom_bar(position = "stack") +
  #   ggtitle(" immucan flowsom cluster") +
  #   theme_bw() +
  #   theme(axis.text.x = element_text(hjust = 1))
  ####
  
  # #### heatmap cluster vs mean marker expression####
  # #ground truth heatmap
  # gt_heat <- immucan %>%
  #   group_by(cell_type)%>%
  #   select(MPO:cleavedPARP, -HistoneH3) %>%
  #   summarise_each(list(mean))
  # gt_heat <- as.data.frame(gt_heat)
  # rownames(gt_heat) <- gt_heat$cell_type
  # 
  # gt_heat%>%
  #   select(-cell_type) %>%
  #   #log() %>%
  #   pheatmap::pheatmap(main = "immucan data, cell_type vs markers ", fontsize = 7, angle_col = 45 ,fontsize_col = 10 )
  # mapply('/', select(gt_heat, -cell_type), sapply(select(gt_heat, -cell_type) , max)) %>%
  #   as.data.frame(row.names = gt_heat$cell_type) %>%
  #   pheatmap::pheatmap(main = "Mean expressions of markers in cell_type, normalized to 1", fontsize = 7,fontsize_col = 10, angle_col = 45)
  # 
  # #phenograph heatmap
  # heatdata <- immucan %>%
  #   group_by(phenograph_cluster_k40)%>%
  #   filter(pheno40_cell_type != "Cancer") %>%
  #   select(MPO:cleavedPARP, -HistoneH3) %>%
  #   #select(CD14:CD3) %>%
  #   summarise_each(list(mean))
  # heatdata <- as.data.frame(heatdata)
  # rownames(heatdata) <- heatdata$phenograph_cluster_k40
  # 
  # heatdata%>%
  #   select(-phenograph_cluster_k40) %>%
  #   #log() %>%
  #   pheatmap::pheatmap(main = "immucan data, phenograph_cluster_k40 vs markers ", fontsize = 7, angle_col = 45 ,fontsize_col = 10  )
  # mapply('/', select(heatdata, -phenograph_cluster_k40), sapply(select(heatdata, -phenograph_cluster_k40) , max)) %>%
  #   as.data.frame(row.names = heatdata$phenograph_cluster_k40) %>%
  #   pheatmap::pheatmap(main = "Mean expressions of markers in clusters, normalized to 1", fontsize = 7,fontsize_col = 10, angle_col = 45)
  # 
  # 
  # 
  # 
  # #### phenograph k40 nimbus cluster assignment ####
  # immucan$pheno40_cell_type <- as.character(immucan$phenograph_cluster_k40)
  # 
  # immucan$pheno40_cell_type[immucan$pheno40_cell_type %in% c(66,69,125,112,122,50,116,104,41,82,88,101,45,52,64,
  #                                                            128,55,83,15,42,86,20,40,62,103,19,65,102,57,2,68,11,
  #                                                            39,99,117,59,96,94,105,90,81,106,77,84,76,87,107,109,
  #                                                            25,100,108,79,121,27,97,8,12,58,63,54,113,14,85,24,51,
  #                                                            53,26,80,34,44,56,16,46,118,120,95,123,119,126,111,124,
  #                                                            110,30,70,35,67,49,33,22,32,36,23,29,115,10,31,114,38,
  #                                                            47,60,75,91,93,98)] <- "Cancer" # maybe ,28,38,26
  # immucan$pheno40_cell_type[immucan$pheno40_cell_type %in% c(74,17)] <- "Plasma_cell" 
  # immucan$pheno40_cell_type[immucan$pheno40_cell_type %in% c(43,28,5,18,21,3)] <- "Stroma" # maybe 25,15, 73
  # immucan$pheno40_cell_type[immucan$pheno40_cell_type %in% c(71)] <- "Dendritic_cell"
  # immucan$pheno40_cell_type[immucan$pheno40_cell_type %in% c(127,13,43)] <- "M2_Macrophage" 
  # immucan$pheno40_cell_type[immucan$pheno40_cell_type %in% c(61)] <- "Plasmacytoid_dendritic_cell" # maybe 2,39
  # immucan$pheno40_cell_type[immucan$pheno40_cell_type %in% c(37)] <- "B_cell"
  # immucan$pheno40_cell_type[immucan$pheno40_cell_type %in% c(72)] <- "BnT"
  # immucan$pheno40_cell_type[immucan$pheno40_cell_type %in% c(4)] <- "NK_cell" #maybe 5,86, 2
  # immucan$pheno40_cell_type[immucan$pheno40_cell_type %in% c()] <- "CD4+_T_cell"
  # immucan$pheno40_cell_type[immucan$pheno40_cell_type %in% c(6)] <- "CD8+_T_cell" # maybe ,52,81
  # immucan$pheno40_cell_type[immucan$pheno40_cell_type %in% c(78,89,9,36)] <- "Neutrophil" # maybe 39
  # immucan$pheno40_cell_type[immucan$pheno40_cell_type %in% c(7)] <- "Treg" #maybe 5, 15, maybe 26,28,38 unsure about 19,16,96
  # immucan$pheno40_cell_type[immucan$pheno40_cell_type %in% c(1,48,73,74,92)] <- "undefined"
  # 
  # #maybe_cancer <- c(92,98,43,91,48,89,71,75,127,78,74)
  # 
  # table(immucan$pheno40_cell_type)
  # table(immucan$cell_type)
  # 
  
  #### greedy assignment ####
  
  # #greedy assignment
  # results <- greedy_f1_score(immucan, 'cell_type', "phenograph_cluster_k30", tie_strategy = 'random')
  # 
  # # map assigned clusters to new column 
  # immucan$predicted_phenotype <- results$mapped_predictions
  # 
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
      results <- greedy_f1_score(immucan, k, i, tie_strategy = 'random')
      cat("F1 macro:", results$f1_macro, "\n")
      cat("f1_weighted:", results$f1_weighted, "\n")
      output <- select(immucan, MPO:cell_type)
      output$predicted_phenotype <- as.vector(results$mapped_predictions)
      output = rename(output, true_phenotype = k)
      safe_write_csv(output, paste("/Users/margotchazotte/Documents/uni/PhD/pheno_benchmark/immucan/", i, 
                                   "/", names(granularity[granularity == k]),
                                   "/predictions_", iteration,".csv", 
                                   sep = ""))
    }
  }
}
