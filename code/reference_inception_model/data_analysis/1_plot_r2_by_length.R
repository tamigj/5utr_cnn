rm(list=ls())

library(stringr)
library(ggplot2)
library(dplyr)
library(reticulate)


#------------#
# VARIABLES  #
#------------#
dir_data = '/oak/stanford/groups/pritch/users/tami/5utr_cnn/data/'
dir_out_ref = '/oak/stanford/groups/pritch/users/tami/5utr_cnn/output/reference_inception_model/best_model_evaluation/'


#------------#
# LOAD DATA  #
#------------#
df_training = read.delim2(str_interp("${dir_data}/ref_data.tsv"))

df_raw = read.delim2(str_interp(
  "${dir_out_ref}/predictions_with_ref_data.tsv"))


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


#------------------------#
# R² BY SEQUENCE LENGTH  #
#------------------------#
r2_df <- df %>% 
  group_by(seq_len_category) %>%
  summarise(r2 = cor(as.numeric(true_translation), 
                     as.numeric(predicted_translation))^2,
            n = n(),
            .groups = 'drop') %>%
  mutate(label = paste0("R² = ", round(r2, 3), "\nN = ", format(n, big.mark = ",")))

p1 = ggplot(df, 
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

ggsave(paste0(dir_out_ref, "/r2_by_seq_len_category.png"), 
       width = 10, height = 6)

ggsave(paste0(dir_out_ref, "/r2_by_seq_len_category.pdf"), 
       width = 10, height = 6)


#----------------------------------------------#
# DISTRIBUTION OF TRAINING SIZE IN TRAIN DATA  #
#----------------------------------------------#
set.seed(123)

df_sample <- df_training %>%
  slice_sample(prop = 0.7) %>%
  mutate(seq_len = nchar(REF_sequence)) %>%
  mutate(seq_len_category = case_when(
    seq_len <= 60 ~ "<60",
    seq_len <= 100 ~ "60-100",
    seq_len <= 150 ~ "100-150",
    TRUE ~ ">150"
  )) %>%
  mutate(seq_len_category = factor(seq_len_category,
                                   levels = c('<60', '60-100', '100-150', '>150')))

# p2: bar chart of counts by category
p2 <- df_sample %>%
  count(seq_len_category) %>%
  ggplot(aes(x = seq_len_category, y = n, fill = seq_len_category)) +
  geom_col(color = "black") +
  geom_text(aes(label = format(n, big.mark = ",")), vjust = -0.5, size = 4) +
  xlab("Sequence length category") +
  ylab("N") +
  theme_light() +
  theme(axis.text = element_text(size = 13),
        axis.title = element_text(size = 13),
        legend.position = 'none')

# p3: faceted histograms with N in top left
n_df <- df_sample %>%
  count(seq_len_category) %>%
  mutate(label = paste0("N = ", format(n, big.mark = ",")),
         x_pos = ifelse(seq_len_category == ">150", -Inf, Inf),
         hjust_val = ifelse(seq_len_category == ">150", -0.1, 1.1))

p3 <- df_sample %>%
  ggplot(aes(x = seq_len, fill = seq_len_category)) +
  geom_histogram(color = "black", bins = 30) +
  facet_wrap(~seq_len_category, scales = "free") +
  geom_text(data = n_df,
            aes(x = x_pos, y = Inf, label = label, hjust = hjust_val),
            vjust = 1.5,
            inherit.aes = FALSE, size = 4) +
  xlab("Sequence length") +
  ylab("Count") +
  theme_light() +
  theme(axis.text = element_text(size = 13),
        axis.title = element_text(size = 13),
        strip.text = element_text(size = 14, color = "white"),
        strip.background = element_rect(fill = "black"),
        legend.position = 'none')

ggsave(paste0(dir_out_ref, "/n_by_seq_len_category_training.png"), p2, width = 6, height = 4)
ggsave(paste0(dir_out_ref, "/n_by_seq_len_category_training.pdf"), p2, width = 6, height = 4)

ggsave(paste0(dir_out_ref, "/seq_len_distribution_by_category.png"), p3, width = 6, height = 4)
ggsave(paste0(dir_out_ref, "/seq_len_distribution_by_category.pdf"), p3, width = 6, height = 4)
