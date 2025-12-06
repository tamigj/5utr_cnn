rm(list=ls())

# Install packages if needed
required_packages = c("ggplot2", "dplyr", "tidyr", "gridExtra")
new_packages = required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)

#------------#
# VARIABLES  #
#------------#
# Use absolute path to workspace root
workspace_root = '/mnt/oak/users/tami/5utr_cnn'
dir_out_analysis = file.path(workspace_root, 'output/variation_inception_model/data_analysis')

print(paste("Workspace root:", workspace_root))
print(paste("Output directory:", dir_out_analysis))

# Create output directory if it doesn't exist
dir.create(dir_out_analysis, showWarnings = FALSE, recursive = TRUE)

#------------#
# LOAD DATA  #
#------------#
# Panel A: Hardcoded data from 20p (20% data) log files
df = data.frame(
  combo = c("0.05_0.05_0.9", "0.05_0.15_0.8", "0.1_0.1_0.8", "0.15_0.05_0.8", 
            "0.1_0.4_0.5", "0.25_0.25_0.5", "0.4_0.1_0.5", "0.4_0.4_0.2"),
  loss_weight_ref = c(0.05, 0.05, 0.1, 0.15, 0.1, 0.25, 0.4, 0.4),
  loss_weight_alt = c(0.05, 0.15, 0.1, 0.05, 0.4, 0.25, 0.1, 0.4),
  loss_weight_delta = c(0.9, 0.8, 0.8, 0.8, 0.5, 0.5, 0.5, 0.2),
  r2_test = c(0.385536, 0.376661, 0.379411, 0.373155, 0.368635, 0.365523, 0.356451, 0.350261),
  r2_seen = c(0.401399, 0.391788, 0.39151, 0.383423, 0.374828, 0.372944, 0.36385, 0.35548),
  r2_unseen = c(0.370271, 0.362103, 0.367758, 0.363258, 0.362648, 0.358358, 0.349306, 0.345207),
  stringsAsFactors = FALSE
)

print(paste("Loaded", nrow(df), "combinations for Panel A (20% data)"))

# Panel C: Hardcoded data from full data with REF, ALT, and Variant Effect R²
df_c = data.frame(
  combo = c("0.1_0.1_0.8", "0.1_0.1_0.8", "0.1_0.1_0.8",
            "0.05_0.05_0.9", "0.05_0.05_0.9", "0.05_0.05_0.9"),
  prediction_type = c("Variant Effect", "REF Translation", "ALT Translation",
                      "Variant Effect", "REF Translation", "ALT Translation"),
  r2_test = c(0.369143, 0.487873, 0.402831,
              0.392186, 0.477921, 0.391317),
  r2_seen = c(0.388290, 0.652797, 0.484697,
              0.439772, 0.584080, 0.462478),
  r2_unseen = c(0.350625, 0.321627, 0.331752,
                0.346687, 0.370779, 0.329371),
  stringsAsFactors = FALSE
)

print(paste("Loaded", nrow(df_c), "rows for Panel C (full data)"))

# Sort by loss weights (delta descending, then ref, then alt)
df = df[order(-df$loss_weight_delta, df$loss_weight_ref, df$loss_weight_alt), ]

# Create factor for combo to maintain order (reversed for x-axis)
df$combo = factor(df$combo, levels = rev(unique(df$combo)))

# Add category for equal vs unequal weights
df$weight_category = ifelse(df$loss_weight_ref == df$loss_weight_alt,
                            "Equal weight on REF and ALT",
                            "Unequal weight on REF and ALT")
df$weight_category = factor(df$weight_category, 
                            levels = c("Equal weight on REF and ALT", 
                                       "Unequal weight on REF and ALT"))

# Reshape to long format for plotting
df_long = df %>%
  select(combo, weight_category, r2_test, r2_seen, r2_unseen) %>%
  pivot_longer(cols = c(r2_test, r2_seen, r2_unseen),
               names_to = "metric",
               values_to = "r2") %>%
  mutate(metric = case_when(
    metric == "r2_test" ~ "Overall",
    metric == "r2_seen" ~ "Seen genes",
    metric == "r2_unseen" ~ "Unseen genes"
  )) %>%
  mutate(metric = factor(metric, levels = c("Overall", "Seen genes", "Unseen genes")))

#------------#
# PANEL A   #
#------------#
# Sort by loss weights (delta descending, then ref, then alt)
df = df[order(-df$loss_weight_delta, df$loss_weight_ref, df$loss_weight_alt), ]

# Create factor for combo to maintain order (reversed for x-axis)
df$combo = factor(df$combo, levels = rev(unique(df$combo)))

# Add category for equal vs unequal weights
df$weight_category = ifelse(df$loss_weight_ref == df$loss_weight_alt,
                            "Equal weight on REF and ALT",
                            "Unequal weight on REF and ALT")
df$weight_category = factor(df$weight_category, 
                            levels = c("Equal weight on REF and ALT", 
                                       "Unequal weight on REF and ALT"))

# Reshape to long format for plotting
df_long = df %>%
  select(combo, weight_category, r2_test, r2_seen, r2_unseen) %>%
  pivot_longer(cols = c(r2_test, r2_seen, r2_unseen),
               names_to = "metric",
               values_to = "r2") %>%
  mutate(metric = case_when(
    metric == "r2_test" ~ "Overall",
    metric == "r2_seen" ~ "Seen genes",
    metric == "r2_unseen" ~ "Unseen genes"
  )) %>%
  mutate(metric = factor(metric, levels = c("Overall", "Seen genes", "Unseen genes")))

pA = ggplot(df_long, aes(x = combo, y = r2, fill = metric)) +
  geom_col(position = "dodge", color = "black", width = 0.7) +
  scale_fill_manual(values = c("Overall" = "#F8766D", 
                               "Seen genes" = "#7CAE00", 
                               "Unseen genes" = "#00BFC4"),
                   name = "") +
  facet_wrap(~weight_category, ncol = 2, scales = "free_x") +
  xlab("Weights for loss function (REF_ALT_variant effect)") +
  ylab("R²") +
  theme_light() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
        axis.text.y = element_text(size = 12),
        axis.title = element_text(size = 13),
        legend.position = "none",
        panel.grid.major.x = element_blank(),
        panel.grid.minor = element_blank(),
        strip.text = element_text(size = 14, color = "white"),
        strip.background = element_rect(fill = "black"))

#------------#
# PANEL C    #
#------------#
# Process Panel C data
df_c$combo = factor(df_c$combo, levels = c("0.1_0.1_0.8", "0.05_0.05_0.9"))
df_c$prediction_type = factor(df_c$prediction_type, 
                              levels = c("Variant Effect", "REF Translation", "ALT Translation"))

df_c_long = df_c %>%
  select(combo, prediction_type, r2_test, r2_seen, r2_unseen) %>%
  pivot_longer(cols = c(r2_test, r2_seen, r2_unseen),
               names_to = "metric",
               values_to = "r2") %>%
  mutate(metric = case_when(
    metric == "r2_test" ~ "Overall",
    metric == "r2_seen" ~ "Seen genes",
    metric == "r2_unseen" ~ "Unseen genes"
  )) %>%
  mutate(metric = factor(metric, levels = c("Overall", "Seen genes", "Unseen genes")))

# Calculate y-axis limits for Panel C only
c_r2_values = c(df_c$r2_test, df_c$r2_seen, df_c$r2_unseen)
y_min_c = 0
y_max_c = max(c_r2_values, na.rm = TRUE)
# Add a bit of padding at the top
y_max_c = y_max_c + 0.05 * y_max_c

pC = ggplot(df_c_long, aes(x = prediction_type, y = r2, fill = metric)) +
  geom_col(position = "dodge", color = "black", width = 0.7) +
  scale_fill_manual(values = c("Overall" = "#F8766D", 
                               "Seen genes" = "#7CAE00", 
                               "Unseen genes" = "#00BFC4"),
                   name = "") +
  facet_wrap(~combo, ncol = 2) +
  scale_y_continuous(limits = c(y_min_c, y_max_c)) +
  xlab("") +
  ylab("R²") +
  theme_light() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
        axis.text.y = element_text(size = 12),
        axis.title = element_text(size = 13),
        legend.position = "right",
        legend.text = element_text(size = 11),
        panel.grid.major.x = element_blank(),
        panel.grid.minor = element_blank(),
        strip.text = element_text(size = 14, color = "white"),
        strip.background = element_rect(fill = "black"))

#------------#
# COMBINE    #
#------------#
combined = grid.arrange(pA, pC, ncol = 1, heights = c(1, 1))

# Save
ggsave(paste0(dir_out_analysis, "/r2_by_loss_weights.png"), 
       combined, width = 12, height = 10, dpi = 300)

ggsave(paste0(dir_out_analysis, "/r2_by_loss_weights.pdf"), 
       combined, width = 12, height = 10)

print("Plot saved successfully!")
print("\nSummary Panel A:")
print(df)
print("\nSummary Panel C:")
print(df_c)
