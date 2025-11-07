#install.packages("funkyheatmap")
library(funkyheatmap)
library(dplyr, warn.conflicts = FALSE)
library(tibble, warn.conflicts = FALSE)
library(readr)
library(RColorBrewer)
library(ggplot2)
library(colorspace)

results <- read_delim("final_results_averaged.csv", delim = ";", escape_double = FALSE, trim_ws = TRUE)
#old <- read_csv("Documents/uni/PhD/pheno_benchmark/results/results_lvl3.csv")
results <- results[ , c("method", "Weighted F1","Hierarchical F1", "Macro F1",  "MCC", "ARI", "JSD Scaled", "Overall Performance" ,"Stability", "Scalability")]
results$`Overall Performance` <- round(results$`Overall Performance`, 3)
colnames(results)[colnames(results) == "method"] <- "id"
results$Scalability[results$id == 'Deepcelltypes_adapted'] <- results$Scalability[results$id == 'Deepcelltypes']
results$Scalability[results$id == 'RIBCA_adapted'] <- results$Scalability[results$id == 'RIBCA']

rcinfo <- tibble(
  id = colnames(results),
  group = c(NA, rep("Classification Performance", 4), rep("Celltype Composition", 2), "Overall Performance", rep("Computation", 2)),
  options = lapply(seq(10), function(x) lst()),
  name = c("", "Weighted F1", "Hierarchical F1", "Macro F1", "MCC", "ARI", "JSD Scaled", "Overall Performance", "Stability", "Scalability"),
  palette = c(NA, rep("CP_palette", 4), rep("Composition_palette", 2), "overall_palette",  rep("Computation_palette", 2)), 
  geom = c("text", "circle","circle", "circle", "circle", "funkyrect", "funkyrect", "bar", "rect", "rect"),
  width = c(7, 1, 1, 1, 1,1,1,4,1,1),
  hjust = c(0,NA,NA,NA,NA,NA,NA,NA,NA,NA)
)

rcinfo <- rcinfo %>%
  add_row(id = "Overall Performance", group = "Overall Performance", name = "", geom = "text", options = lst(lst(overlay = TRUE)), palette = NA, .before = 9, width = 1, hjust = 0.95)

palettes <- list(overall_palette = rev(sequential_hcl(8, "BuGn")), Composition_palette = brewer.pal(9, "PuBu")[-1], CP_palette = brewer.pal(9, "RdPu")[-1], Computation_palette = rev(sequential_hcl(8, "Peach")))
palettes$funky_palette_grey <- RColorBrewer::brewer.pal(9, "Greys")[-1]

column_groups <- tibble(
  Category = c("Overall Performance", "Classification Performance", "Celltype Composition", "Computation"),
  group = c("Overall Performance", "Classification Performance", "Celltype Composition", "Computation"),
  palette = c("overall_palette", "CP_palette", "Composition_palette", "Computation_palette")
)

#rcinfo$id_size <- c(NA,"weighted_score",NA,"f1_weighted_mean","hierarchical_f1_mean", "macro_f1_mean","g_mean_mean","r2_mean","ari_mean","jsd_scaled_mean","stability","scalability")

legends <- list(
  list(
    palette = "overall_palette",
    geom = "bar",
    title = "Overall",
    labels = c(0, rep("", 7), 1)
  ),
  list(
    palette = "CP_palette",
    geom = "circle",
    title = "Classification Performance",
    labels = c("",0, rep("", 7), 1)
  ),
  list(
    palette = "Composition_palette",
    geom = "funkyrect",
    title = "Celltype Composition",
    labels = c("",0, rep("", 7), 1) ,
    enabled = TRUE
  ),
  list(
    palette = "Computation_palette",
    geom = "rect",
    title = "Computation",
    size = c(rep(1,5)),
    labels = c(0, rep("", 3), 1),
    enabled = TRUE
    
  ),
  list(
    palette = "funky_palette_grey",
    geom = "funkyrect",
    title = "Overall",
    enabled = FALSE,
    labels = c("0", "", "0.2", "", "0.4", "", "0.6", "", "0.8", "", "1")
  )
)

row_info <- tibble(id = results$id, group = c(rep("Supervised",5), "Unsupervised", "Prior-Knowledge Driven", rep("Unsupervised",3), "Prior-Knowledge Driven",rep("Unsupervised",2),"Prior-Knowledge Driven","Unsupervised", "Pre-Trained Models", "Prior-Knowledge Driven", "Pre-Trained Models", "Unsupervised", rep("Pre-Trained Models",2), "Baseline", "Pre-Trained Models","Baseline"))
row_info$group <- factor(row_info$group, levels = c( "Supervised", "Unsupervised", "Prior-Knowledge Driven", "Pre-Trained Models", "Baseline"))
results <- results[order(row_info$group), ]
row_info <- row_info[order(row_info$group), ]
row_groups <- tibble(level1 = c("Supervised", "Unsupervised", "Prior-Knowledge Driven", "Pre-Trained Models", "Baseline"), group = c("Supervised", "Unsupervised", "Prior-Knowledge Driven", "Pre-Trained Models", "Baseline"))


p = funky_heatmap(results, 
              column_info = rcinfo, 
              column_groups = column_groups, 
              palettes = palettes, 
              legends = legends, 
              scale_column = FALSE, 
              row_info = row_info,
              row_groups = row_groups)
ggsave(
  filename = "figure_plots/funky_heatmap.svg",
  plot = p,
  width = 15,       # width in inches
  height = 10,      # height in inches
  dpi = 300,        # resolution in dots per inch     # specify a background color
)

