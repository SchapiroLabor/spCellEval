library(tidyverse)
library(ggplot2)

datasets <- tibble(
  platform = c(rep("MIBI", 2), rep("CODEX", 1), rep("IMC", 1), rep("Lunaphore", 1), "CODEX", "MIBI", rep("IMC", 2)),
  cells = c(1669853, 230895, 145161,100000,100000, 1200000, 500000, 553121, 457117), 
  tissue = c(rep("Lymph node", 3), "Bone marrow","Heart", "Bone marrow", "Decidua", "Breast",  "Multiple"),
  stage1 = c("remaining", "pilot", rep("remaining", 6), "pilot")
)
datasets$platform <- factor(datasets$platform, levels = c("MIBI", "CODEX", "IMC", "Lunaphore"))
datasets$tissue <- factor(datasets$tissue, levels = c("Lymph node", "Bone marrow", "Breast", "Decidua", "Multiple","Heart"))
datasets$stage1 <- factor(datasets$stage1, levels = c("remaining", "pilot"))

ggplot(datasets, aes(platform, cells, fill = stage1)) +
  geom_col(position = "stack", width = 0.8, color = "#3E424B") +
  scale_fill_manual(values = c( "grey", "#7AA2C4")) + #46a2da #006884
  theme_classic() +
  theme(panel.grid.major.y = element_line(size = 0.5, colour = "lightgrey"),
        text = element_text(size = 20),
        axis.text = element_text(colour = "black"), 
        legend.title = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(colour = "black", size = 20))+ #363636
  #guides(fill = "none") +
  labs(x = "Platform", y = "cells", title = "Cells per Platform")


ggplot(datasets, aes(tissue, cells, fill = stage1)) +
  geom_col(position = "stack", width = 0.8, color = "#3E424B") +
  scale_fill_manual(values = c( "grey", "#7AA2C4")) +
  theme_classic() +
  theme(panel.grid.major.y = element_line(size = 0.5, colour = "lightgrey"),
        text = element_text(size = 20),
        axis.text = element_text(colour = "black"), 
        axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(colour = "black", size = 20))+ #363636
  guides(fill = "none") +
  labs(x = "Tissue", y = "cells", title = "Cells per Tissue")


