library(tidyverse)
library(ggplot2)
install.packages("Cairo")
library(Cairo)

library(webr)

#dataprep
datasets <- tibble(
  platform = c(rep("MIBI", 2), rep("CODEX", 1), rep("IMC", 1), rep("Lunaphore", 1), "CODEX", "MIBI", rep("IMC", 2)),
  cells = c(1669853, 230895, 145161,100000,100000, 1200000, 500000, 553121, 457117), 
  tissue = c(rep("Lymph node", 3), "Bone marrow","Heart", "Bone marrow", "Decidua", "Breast",  "Multiple"),
  stage1 = c("full", "pilot", rep("full", 6), "pilot")
)
datasets$platform <- factor(datasets$platform, levels = c("MIBI", "IMC","CODEX",  "Lunaphore"))
datasets$tissue <- factor(datasets$tissue, levels = c("Lymph node", "Bone marrow","Multiple", "Breast",  "Decidua", "Heart"))
datasets$stage1 <- factor(datasets$stage1, levels = c( "pilot", "full"))

#### barplots ####
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

#### radial plot ####
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
  labs(x = "Tissue", y = "cells", title = "Cells per Tissue") +
  coord_radial(theta = "x")

#### pie charts ####
ggplot(datasets, aes(x = "", y = cells, fill = tissue, color = stage1)) +
  geom_col(width = 1) +
  scale_fill_brewer(palette = "YlGnBu", name = "Tissue")+
  coord_polar("y") +
  labs(title = "Cells per Tissue") +
  scale_color_manual(values = c("grey", "black"), name = "") +
  theme_minimal() +
  guides(color = "none") +
  theme(panel.grid = element_blank(),
        #panel.grid.major.y = element_line(size = 0.5, colour = "lightgrey"),
        text = element_blank(),
        #axis.text.x = element_blank(),
        legend.title = element_text(size = 15),
        plot.title = element_text(colour = "black", size = 20, hjust = 0.5))


ggplot(datasets, aes(x = "", y = cells, fill = platform, color = stage1)) +
  geom_col(width = 1) +
  scale_fill_brewer(palette = "YlGnBu", name = "Platform")+
  coord_polar("y") +
  labs(title = "Cells per Platform") +
  scale_color_manual(values = c("grey", "black"), name = "") +
  theme_minimal() +
  guides(color = "none") +
  theme(panel.grid = element_blank(),
        #panel.grid.major.y = element_line(size = 0.5, colour = "lightgrey"),
        text = element_blank(),
        #axis.text.x = element_blank(),
        legend.title = element_text(size = 15),
        plot.title = element_text(colour = "black", size = 20, hjust = 0.5))

#### PieDonuts ####

#platform 
platform <- datasets %>% group_by(platform, stage1) %>% summarise(n = sum(cells))
platform$stage1 <- as.character(platform$stage1)

PieDonut(platform, aes(platform, stage1, count=n), title = "Pilot: Cells per Platform", 
         #ratioByGroup = FALSE, 
         showRatioDonut = FALSE, showRatioPie = FALSE, 
         showPieName = TRUE, 
         explodeDonut = TRUE,
         donutLabelSize = 4,
         pieLabelSize = 4,
         selected = c(2,4))

# PieDonut(platform, aes(stage1, platform, count=n), title = "Pilot: Cells per Platform", 
#          ratioByGroup = FALSE, 
#          showRatioDonut = FALSE, showRatioPie = FALSE, showPieName = FALSE, labelposition = 1,
#          explode = 2)

#tissue
organ <- datasets %>% group_by(tissue, stage1) %>% summarise(n = sum(cells))
organ$stage1 <- as.character(organ$stage1)

PieDonut(organ, aes(tissue, stage1, count=n), title = "Pilot: Cells per Tissue", 
         #ratioByGroup = FALSE, 
         showRatioDonut = FALSE, showRatioPie = FALSE, 
         showPieName = TRUE, 
         explodeDonut = TRUE,
         donutLabelSize = 4,
         pieLabelSize = 4,
         selected = c(2,6,9))


# PieDonut(organ, aes(stage1, tissue, count=n), title = "Pilot: Cells per Tissue", 
#          ratioByGroup = FALSE, 
#          showRatioDonut = FALSE, showRatioPie = FALSE, showPieName = FALSE, labelposition = 1,
#          explode = 2)
