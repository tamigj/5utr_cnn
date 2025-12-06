rm(list=ls())

# Install packages if needed
required_packages = c("stringr", "ggplot2", "dplyr", "gridExtra")
new_packages = required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

library(stringr)
library(ggplot2)
library(dplyr)
library(gridExtra)

#------------#
# VARIABLES  #
#------------#
dir_data = '/oak/stanford/groups/pritch/users/tami/5utr_cnn/data/'
dir_out_ref = '/oak/stanford/groups/pritch/users/tami/5utr_cnn/output/reference_inception_model/best_model_evaluation/'
dir_out_analysis = '/oak/stanford/groups/pritch/users/tami/5utr_cnn/output/reference_inception_model/data_analysis/'

# Create output directory if it doesn't exist
dir.create(dir_out_analysis, showWarnings = FALSE, recursive = TRUE)

#------------#
# LOAD DATA  #
#------------#
df_raw = read.delim2(str_interp("${dir_out_ref}/predictions_with_ref_data.tsv"))

#---------------#
# PROCESS DATA  #
#---------------#
df = df_raw %>% 
  mutate(seq_len = nchar(ref_sequence)) %>%
  mutate(seq_len_category = case_when(
    seq_len <= 60 ~ "25-60bp",
    seq_len <= 120 ~ "61-120bp",
    seq_len <= 160 ~ "121-160bp",
    TRUE ~ "161-180bp"
  )) %>%
  mutate(seq_len_category = factor(seq_len_category,
                                   levels = c("25-60bp", '61-120bp', '121-160bp', '161-180bp')))

# Convert to numeric
df$true_translation = as.numeric(df$true_translation)
df$predicted_translation = as.numeric(df$predicted_translation)

#------------------------#
# PANEL A: Correlation plot
#------------------------#
# Calculate overall R²
r2_overall = cor(df$true_translation, df$predicted_translation)^2
n_overall = nrow(df)
r2_label = paste0("R² = ", round(r2_overall, 3), "\nN = ", format(n_overall, big.mark = ","))

pA = ggplot(df, 
            aes(x = true_translation, 
                y = predicted_translation)) + 
  geom_point(alpha = 0.5, color = "black") +
  geom_abline(slope = 1, intercept = 0, color = 'red', linetype = 'dashed', linewidth = 1) +
  geom_text(aes(x = Inf, y = -Inf, label = r2_label),
            hjust = 1.1, vjust = -0.2,
            inherit.aes = FALSE, size = 5) +
  xlab("True translation") +
  ylab("Predicted translation") +
  theme_light() +
  theme(axis.text = element_text(size = 12),
        axis.title = element_text(size = 13))

#------------------------#
# PANEL B: R² BY SEQUENCE LENGTH (2x2 faceted)
#------------------------#
r2_df <- df %>% 
  group_by(seq_len_category) %>%
  summarise(r2 = cor(as.numeric(true_translation), 
                     as.numeric(predicted_translation))^2,
            n = n(),
            .groups = 'drop') %>%
  mutate(label = paste0("R² = ", round(r2, 3), "\nN = ", format(n, big.mark = ",")))

pB = ggplot(df, 
       aes(x = as.numeric(true_translation), 
           y = as.numeric(predicted_translation), 
           color = seq_len_category)) + 
  geom_point() +
  facet_wrap(~seq_len_category) +
  geom_text(data = r2_df, 
            aes(x = Inf, y = -Inf, label = label),
            hjust = 1.1, vjust = -0.2, 
            inherit.aes = FALSE, size = 5) +
  xlab("True translation") +
  ylab("Predicted translation") +
  geom_abline(slope = 1, color = 'grey', linetype = 'dashed') +
  theme_light() +
  theme(axis.text = element_text(size = 13),
        axis.title = element_text(size = 13),
        strip.text = element_text(size = 14, color = "white"),
        strip.background = element_rect(fill = "black"),
        legend.position = 'none')

#------------------------#
# PANEL C: Barplot with SE error bars
#------------------------#
# Data from compute_mean_and_se_r2.py results
retrain_data = data.frame(
  condition = c("100", 
                "125", 
                "150", 
                "any"),
  mean_r2 = c(-0.030680, 0.013882, 0.115551, 0.185688),
  se = c(0.046064, 0.024434, 0.018818, 0.029863),
  n = c(30, 30, 30, 30)
)

# Factor to maintain order (100, 125, 150, any from top to bottom)
retrain_data$condition = factor(retrain_data$condition, 
                                levels = c("100", 
                                          "125", 
                                          "150", 
                                          "any"))

pC = ggplot(retrain_data, 
            aes(x = mean_r2, y = condition)) +
  geom_col(fill = "steelblue", color = "black", width = 0.7) +
  geom_errorbarh(aes(xmin = mean_r2 - se, xmax = mean_r2 + se), 
                 height = 0.3, linewidth = 0.8) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.5) +
  xlab("Mean R²") +
  ylab("Train seq < __bp") +
  coord_flip() +
  theme_light() +
  theme(axis.text = element_text(size = 12),
        axis.title = element_text(size = 13),
        panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank())

#------------------------#
# COMBINE PANELS
#------------------------#
# Arrange in a grid: 
# - Two columns with ratio 1:2
# - First column: pA (top) and pC (bottom)
# - Second column: pB (full height)
combined = grid.arrange(
  pA, pC, pB,
  ncol = 2,
  nrow = 2,
  layout_matrix = rbind(c(1, 3), c(2, 3)),
  widths = c(1, 2),
  heights = c(1, 1)
)

# Save
ggsave(paste0(dir_out_analysis, "/three_panel_analysis.png"), 
       combined, width = 12, height = 7, dpi = 300)

ggsave(paste0(dir_out_analysis, "/three_panel_analysis.pdf"), 
       combined, width = 12, height = 7)

print("Plots saved successfully!")
