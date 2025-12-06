rm(list=ls())

library(data.table)
library(dplyr)

#------------#
# LOAD DATA  #
#------------#
df_1m = data.frame(fread(
  "/oak/stanford/groups/pritch/users/tami/5utr_ukb/data/1m_panel/naptrap_1m_panel_annotated.txt",
  select = c('gene_name', 'CHR', 'POS', 'REF', 'ALT', 'REF_sequence', 'ALT_sequence'))) %>% 
  
  rename(gene = 'gene_name') %>%
  mutate(humvar = paste(CHR, POS, REF, ALT, sep='-')) %>%
  select(-CHR, -POS, -REF, -ALT) %>%
  group_by(humvar) %>%
  filter(n() == 1) %>%
  ungroup()

df_naptrap_raw = data.frame(fread(
  "/oak/stanford/groups/pritch/users/tami/5utr_ukb/data/naptrap/processed/humvar_5utr_ntrap_processed_all_for_cs230.csv",
  select = c('gene', 'humvar',
             'position_within_reporter_seq', 
             'predicted_category', 
             'posterior_hs', 'hs_percentile',
             'pw_mean_translation',
             'pw_mean_log2_delta_t', 
             'pw_mean_log2_diploid_fc'))) %>% 
  group_by(across(-position_within_reporter_seq)) %>%
  slice_max(position_within_reporter_seq, n = 1, with_ties = FALSE) %>%
  ungroup()

df_naptrap_refs = df_naptrap_raw %>%
  filter(humvar == 'ref') %>%
  rename(REF_pw_mean_translation = 'pw_mean_translation') %>%
  select(gene, REF_pw_mean_translation)

df_naptrap_alt = df_naptrap_raw %>%
  filter(humvar != 'ref') %>%
  rename(ALT_pw_mean_translation = 'pw_mean_translation') %>%
  group_by(humvar) %>%
  filter(n() == 1) %>%
  ungroup()

df_naptrap = merge(df_naptrap_alt, df_naptrap_refs, by='gene',
                   all.x=TRUE)

df_naptrap_w_sequences = merge(df_naptrap, df_1m, 
                               by=c('gene', 'humvar'))


#----------------#
# CLEAN UP DATA  #
#----------------#
# Add cols to make it reproducible with earlier code 
df_naptrap_w_sequences = df_naptrap_w_sequences %>%
  mutate(pw_mean_translation = ALT_pw_mean_translation) %>%
  select(gene, humvar, REF_sequence, ALT_sequence, 
         REF_pw_mean_translation, ALT_pw_mean_translation,
         pw_mean_translation,
         pw_mean_log2_delta_t, pw_mean_log2_diploid_fc,
         position_within_reporter_seq:hs_percentile) %>%
  mutate(ref_seq_len = nchar(REF_sequence),
         alt_seq_len = nchar(ALT_sequence))

df_naptrap_refs = merge(df_naptrap_refs, 
                        df_naptrap_w_sequences %>% select(gene, REF_sequence),
                        by='gene') %>% 
  distinct() %>%
  rename(pw_mean_translation = 'REF_pw_mean_translation')


#------------#
# SAVE DATA  #
#------------#
write.table(df_naptrap_w_sequences, 
            "/oak/stanford/groups/pritch/users/tami/5utr_cnn/data/naptrap_full_data.tsv",
            row.names=FALSE, quote=FALSE, sep='\t')

write.table(df_naptrap_refs, 
            "/oak/stanford/groups/pritch/users/tami/5utr_cnn/data/ref_data.tsv",
            row.names=FALSE, quote=FALSE, sep="\t")
