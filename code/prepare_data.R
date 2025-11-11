rm(list=ls())

library(data.table)
library(dplyr)
library(ggplot2)


#------------#
# LOAD DATA  #
#------------#
df_1m = data.frame(fread(
  "/oak/stanford/groups/pritch/users/tami/5utr_ukb/data/1m_panel/naptrap_1m_panel_annotated.txt"
))

df_naptrap_raw = data.frame(fread(
  "/oak/stanford/groups/pritch/users/tami/5utr_ukb/data/naptrap/processed/humvar_5utr_ntrap_ash_all.csv"
))

df_naptrap_w_refs = data.frame(fread(
  "/oak/stanford/groups/pritch/users/tami/5utr_ukb/data/naptrap/processed/humvar_5utr_ntrap_processed_all_includes_refs.csv"
))

df_naptrap_features = data.frame(fread(
  "/oak/stanford/groups/pritch/users/tami/5utr_ukb/data/naptrap/processed/humvar_5utr_ntrap_ash_all_variant_and_gene_features.tsv",
  select = c('gene', 'humvar', 'UTR_length')
))

df_naptrap_mut = data.frame(fread(
  "/oak/stanford/groups/pritch/users/tami/5utr_ukb/data/naptrap/processed/humvar_5utr_ntrap_ash_all_with_ppoly.csv",
  select = c('humvar', 'mu_gnomad', 'p_poly_gnomad_binned')
))


#-------------#
# MERGE DATA  #
#-------------#
df_naptrap = merge(df_naptrap_raw %>% select(-pw_mean_translation,
                                             -pw_se_translation), 
                   df_naptrap_w_refs,
                   by=c('gene', 'humvar'), all.x=TRUE, all.y=TRUE)

df_naptrap_extra = merge(df_naptrap_features %>% select(-gene), df_naptrap_mut, by='humvar')

df_all = merge(
  merge(df_1m %>% 
          mutate(humvar = paste(CHR, POS, REF, ALT, sep='-')) %>%
          select(humvar, REF_sequence, ALT_sequence), 
        
        df_naptrap, by='humvar', all.y=TRUE), 
  df_naptrap_extra, by='humvar', all.x=TRUE)

df_nn = df_all %>%
  select(humvar, gene, 
         REF_sequence, ALT_sequence, 
         library, duplicate,
         position_within_reporter_seq, UTR_length,
         
         AC, AN, AF, 
         posterior_hs,
         mu_gnomad, p_poly_gnomad_binned,
         
         pw_mean_translation, pw_se_translation,
         ash_mean_translation,
         
         pw_mean_log2_delta_t, pw_se_log2_delta_t,
         ash_mean_log2_delta_t, ash_se_log2_delta_t,
         
         pw_mean_log2_diploid_fc, pw_se_log2_diploid_fc,
         ash_mean_log2_diploid_fc, ash_se_log2_diploid_fc) %>%
  
  rename(mutation_rate = 'mu_gnomad',
         prob_polymorphic = 'p_poly_gnomad_binned',
         shet = 'posterior_hs')

write.table(df_nn, 
            "/oak/stanford/groups/pritch/users/tami/5utr_cnn/data/naptrap_full_data.tsv",
            row.names=FALSE, quote=FALSE, sep='\t')


#------------------#
# DATA FOR STEP 1  #
#------------------#
df_refs = df_nn %>% 
  filter(humvar == 'ref') %>%
  mutate(library = ifelse(duplicate == 'yes', NA, library)) %>%
  select(gene, library, duplicate,
         pw_mean_translation, pw_se_translation) %>%
  distinct()

df_ref_sequences = df_1m %>% 
  select(gene_name, REF_sequence) %>% 
  distinct() %>%
  rename(gene = 'gene_name')

df_refs = merge(df_refs, df_ref_sequences, by='gene', all.x=TRUE)
df_refs = merge(df_refs,
                df_naptrap_features %>% select(gene, UTR_length) %>% distinct,
                by='gene', all.x=TRUE)

# Clean-up
df_refs = df_refs %>%
  rename(is_duplicate = "duplicate") %>%
  mutate(is_duplicate = ifelse(is_duplicate == 'yes', TRUE, FALSE)) %>%
  rename(ref_sequence = 'REF_sequence') %>%
  rename(gene_utr_length = 'UTR_length') %>%
  rename(naptrap_library = 'library') %>%
  select(gene, ref_sequence, pw_mean_translation, pw_se_translation,
         naptrap_library, is_duplicate, gene_utr_length)

write.table(df_refs, 
            "/oak/stanford/groups/pritch/users/tami/5utr_cnn/data/ref_data.tsv",
            row.names=FALSE, quote=FALSE, sep="\t")
