rm(list=ls())

library(dplyr)
library(tidyr)
library(ggplot2)
library(data.table)
library(stringr)

library(plotrix)
library(devtools)
library(nnls)
library(viridis)
library(DescTools)
library(gridExtra)

setwd('/oak/stanford/groups/pritch/users/tami/5utr_ukb/')


#-----------------#
# SPECIFICATIONS  #
#-----------------#
normalization_spec = 'rpm'
reference_spec = 'reference'
remove_input_0_spec = TRUE
remove_median_0_spec = TRUE
n_bins = 100

libraries = c('25_60', '61_120', '121_160',
              '160_180_lib4', '160_180_lib5',
              '160_180_lib6', '160_180_lib7',
              '160_180_lib8', '160_180_lib9',
              '160_180_lib10')

libraries_in_750k = c('25_60', '61_120', '121_160',
                      '160_180_lib4', '160_180_lib5',
                      '160_180_lib6', '160_180_lib7')

dir_data = './data/naptrap/processed/'
dir_out_all = './output/naptrap/qc_plots_all/'

colors = c('#11324d', '#19588a', '#6eabde', 'lightgrey', '#ca181a')
predicted_categories = c('Total loss', 'Strongly negative', 'Negative', 'No ORF effect', 'Positive')


#------------#
# FUNCTIONS  #
#------------#

# Recording filtering effects
add_sumstat = function(df, sumstats_df, filter_txt){
  
  n_variants = nrow(df)
  n_genes = length(unique(df$gene))
  
  sumstats_df[nrow(sumstats_df)+1, ] = c(filter_txt, n_variants, n_genes)
  
  return(sumstats_df)
  
}

add_lost_columns_to_sumstats = function(sumstats_df) {
  
  sumstats_df$Variants = as.numeric(sumstats_df$Variants)
  sumstats_df$Genes = as.numeric(sumstats_df$Genes)
  
  sumstats_df$Variants_lost <- c(0, -diff(sumstats_df$Variants))
  sumstats_df$Genes_lost <- c(0, -diff(sumstats_df$Genes))
  
  return(sumstats_df)
}

# Data cleaning
keep_relevant_cols_from_df_categories = function(df_categories_raw){
  
  df_categories = df_categories_raw %>%
    rename(predicted_category = 'predicted_category') %>%
    select(gene_name, CHR, POS, REF, ALT,
           AC, AN, AF,
           position_within_reporter_seq, start_codon,
           orf_ann_ref, orf_ann_alt,
           variant_type,
           REF_sequence, insert_seq, predicted_category,
           posterior_hs:hs_percentile) %>%
    rename(ref_seq = 'REF_sequence') %>%
    
    # Make humvar
    rowwise() %>%
    mutate(humvar = ifelse(grepl("_ref", gene_name), 'ref',
                           paste0(CHR, "-", POS, "-", REF, "-", ALT))) %>%
    ungroup()
  
  return(df_categories)
}


# Data processing
remove_reporters_with_any_zeros = function(df){
  
  df = df %>%
    mutate(to_remove = ifelse(input_12h_B1 == 0 | input_12h_B2 == 0 |
                                input_12h_B3 == 0 | input_12h_B4 == 0, 1, 0)) %>%
    filter(to_remove == 0) %>%
    select(-to_remove)
  
  return(df)
  
}

remove_genes_with_0_median = function(df){
  
  # Calculate translation
  df = df %>%
    mutate(translation1 = pulldown_12h_B1/input_12h_B1,
           translation2 = pulldown_12h_B2/input_12h_B2,
           translation3 = pulldown_12h_B3/input_12h_B3,
           translation4 = pulldown_12h_B4/input_12h_B4)
  
  # Calculate median
  df = df %>%
    group_by(gene) %>%
    
    # Compute median translation
    mutate(median1 = median(translation1),
           median2 = median(translation2),
           median3 = median(translation3),
           median4 = median(translation4)) %>%
    
    ungroup() %>%
    
    filter(median1 != 0,
           median2 != 0,
           median3 != 0,
           median4 != 0) %>%
    
    select(-starts_with("translation"),
           -starts_with("median"))
  
  return(df)
  
}

normalize_replicates = function(df_raw, normalization){
  
  if (normalization == 'rpm'){
    
    df = df_raw %>%
      
      # Calculate RPM as ((reporter read count/Total Reads)*1,000,000)
      mutate(input_12h_B1_norm = (input_12h_B1/sum(input_12h_B1))*1000000,
             input_12h_B2_norm = (input_12h_B2/sum(input_12h_B2))*1000000,
             input_12h_B3_norm = (input_12h_B3/sum(input_12h_B3))*1000000,
             input_12h_B4_norm = (input_12h_B4/sum(input_12h_B4))*1000000,
             
             pulldown_12h_B1_norm = (pulldown_12h_B1/sum(pulldown_12h_B1))*1000000,
             pulldown_12h_B2_norm = (pulldown_12h_B2/sum(pulldown_12h_B2))*1000000,
             pulldown_12h_B3_norm = (pulldown_12h_B3/sum(pulldown_12h_B3))*1000000,
             pulldown_12h_B4_norm = (pulldown_12h_B4/sum(pulldown_12h_B4))*1000000)
    
  }
  
  return(df)
  
}


# Calculate constructs
calculate_translation = function(df){
  
  df = df %>%
    
    ungroup() %>%
    
    mutate(translation1 = pulldown_12h_B1_norm/input_12h_B1_norm,
           translation2 = pulldown_12h_B2_norm/input_12h_B2_norm,
           translation3 = pulldown_12h_B3_norm/input_12h_B3_norm,
           translation4 = pulldown_12h_B4_norm/input_12h_B4_norm)
  
  return(df)
  
}

calculate_delta_translation = function(df, reference_txt){
  
  # Calculate median for all
  df = df %>%
    group_by(gene) %>%
    
    # Compute median translation
    mutate(median1 = median(translation1),
           median2 = median(translation2),
           median3 = median(translation3),
           median4 = median(translation4)) %>%
    
    ungroup()
  
  if (reference_txt == 'median'){
    
    # Calculate delta translation
    df = df %>%
      mutate(delta_t1 = (translation1/median1),
             delta_t2 = (translation2/median2),
             delta_t3 = (translation3/median3),
             delta_t4 = (translation4/median4)) %>%
      filter(humvar != 'ref')
    
  } else if (reference_txt == 'reference'){
    
    df_refs = df %>%
      filter(humvar == 'ref') %>%
      select(gene, translation1:translation4) %>%
      rename(ref1 = 'translation1',
             ref2 = 'translation2',
             ref3 = 'translation3',
             ref4 = 'translation4')
    
    # df = df %>%
    #   filter(humvar != 'ref')
    
    genes_without_refs = df$gene[!(df$gene %in% df_refs$gene)]
    n_vars_affected = length(genes_without_refs)
    
    print(str_interp("${n_vars_affected} variants removed due to missing reference."))
    
    df = merge(df, df_refs, by='gene', all.x=TRUE, all.y=FALSE)
    
    df = df %>%
      
      filter(!(gene %in% genes_without_refs)) %>%
      
      # Calculate delta translation
      mutate(delta_t1 = (translation1/ref1),
             delta_t2 = (translation2/ref2),
             delta_t3 = (translation3/ref3),
             delta_t4 = (translation4/ref4)) %>%
      
      select(-starts_with("ref"))
    
  }
  
  return(df)
  
}

calculate_diploid_fold_changes = function(df){
  
  df = df %>%
    
    # Haploid FC
    mutate(haploid_fc1 = delta_t1-1,
           haploid_fc2 = delta_t2-1,
           haploid_fc3 = delta_t3-1,
           haploid_fc4 = delta_t4-1,
           
           # Diploid FC
           diploid_fc1 = 1+0.5*haploid_fc1,
           diploid_fc2 = 1+0.5*haploid_fc2,
           diploid_fc3 = 1+0.5*haploid_fc3,
           diploid_fc4 = 1+0.5*haploid_fc4,
           
           # log2(Diploid FC)
           log2_diploid_fc1 = log2(diploid_fc1),
           log2_diploid_fc2 = log2(diploid_fc2),
           log2_diploid_fc3 = log2(diploid_fc3),
           log2_diploid_fc4 = log2(diploid_fc4)) %>%
    
    select(-starts_with("haploid"), -starts_with("diploid"))
  
  return(df)
  
}

calculate_log2_delta_t = function(df){
  
  log_min_delta_t1 = log2(min(df$delta_t1[df$delta_t1 > 0]))
  log_min_delta_t2 = log2(min(df$delta_t2[df$delta_t2 > 0]))
  log_min_delta_t3 = log2(min(df$delta_t3[df$delta_t3 > 0]))
  log_min_delta_t4 = log2(min(df$delta_t4[df$delta_t4 > 0]))
  
  df = df %>%
    mutate(log2_delta_t1 = ifelse(delta_t1 != 0, log2(delta_t1), log_min_delta_t1),
           log2_delta_t2 = ifelse(delta_t2 != 0, log2(delta_t2), log_min_delta_t2),
           log2_delta_t3 = ifelse(delta_t3 != 0, log2(delta_t3), log_min_delta_t3),
           log2_delta_t4 = ifelse(delta_t4 != 0, log2(delta_t4), log_min_delta_t4))
  
  return(df)
  
}

calculate_expected_pulldown = function(df){
  
  # Bin variants based on mean expected pulldown
  df = df %>%
    
    # Calculate expected pulldown
    mutate(expected_pulldown_1 = input_12h_B1_norm*median1,
           expected_pulldown_2 = input_12h_B2_norm*median2,
           expected_pulldown_3 = input_12h_B3_norm*median3,
           expected_pulldown_4 = input_12h_B4_norm*median4) %>%
    
    ungroup()
  
  return(df)
}


# Calculate SEs
get_design_matrix = function(x, num_bins) {
  n <- nrow(x)
  p <- ncol(x)
  
  # Compute the bin edges based on quantiles
  x_bin <- quantile(array(unlist(x)),
                    probs = seq(0, 1,
                                length.out = num_bins + 1
                    )
  )[-(num_bins + 1)]
  
  # Digitize x based on the computed bin edges
  bins <- matrix(findInterval(array(unlist(x)), x_bin, left.open = TRUE), n, p)
  
  # Initialize the one-hot encoding vector, v
  v <- array(0, dim = c(n, p, num_bins))
  for (i in 1:n) {
    for (j in 1:p) {
      v[i, j, 1:bins[i, j]] <- 1
    }
  }
  
  # Compute the coefficients for the diagonal and off-diagonal elements
  diag_coef <- ((p - 1) / p)^2
  offdiag_coef <- (1 / p)^2
  
  # Calculate u: made up of one hot encoding and coefficent matrix
  u <- array(0, dim = c(n, p, num_bins))
  for (i in 1:n) {
    for (k in 1:p) {
      u[i, k, ] <- (diag_coef - offdiag_coef) *
        v[i, k, ] + offdiag_coef * colSums(v[i, , ])
    }
  }
  
  return(list(u = u, v = v))
}

estimate_standard_error = function(y, x, num_bins) {
  x <- 1 / x
  n <- nrow(x)
  p <- ncol(x)
  
  # Get design matrix
  design_matrices <- get_design_matrix(x, num_bins)
  A <- design_matrices$u
  V <- design_matrices$v
  
  residuals <- y - matrix(
    rep(apply(y, 1, mean), ncol(y)),
    nrow = nrow(y), ncol = ncol(y)
  )
  
  A_transposed <- aperm(A, c(3, 2, 1))
  dims <- dim(A)
  nrow <- prod(dims[1:2]) # Number of rows to reshape to
  ncol <- dims[3]
  A_flat_row_major <- as.vector(A_transposed)
  A <- matrix(A_flat_row_major, nrow = nrow, ncol = ncol, byrow = TRUE)
  V_transposed <- aperm(V, c(3, 2, 1))
  V_flat_row_major <- as.vector(V_transposed)
  V <- matrix(V_flat_row_major, nrow = nrow, ncol = ncol, byrow = TRUE)
  residuals <- as.vector(t(residuals))
  
  # Solve for theta using NNLS
  nnls_result <- nnls(A, residuals^2)
  theta <- coef(nnls_result)
  
  return(matrix(V %*% theta, nrow = n, ncol = p, byrow = TRUE))
}

calculate_se = function(df, beta, num_bins) {
  
  x <- as.matrix(
    df[c(
      "expected_pulldown_1", "expected_pulldown_2",
      "expected_pulldown_3", "expected_pulldown_4"
    )]
  )
  
  y <- as.matrix(
    df[c(
      paste(beta, 1:4, sep='')
    )]
  )
  
  estimated_error <- sqrt(estimate_standard_error(y, x, num_bins))
  stopifnot(all(estimated_error >= 0))
  df_combined <- cbind(df, estimated_error)
  
  colnames(df_combined)[
    (ncol(df) + 1):ncol(df_combined)
  ] <- c(
    paste("se_", beta, 1:4, sep='')
  )
  
  df <- df_combined
  
  return(df)
}

calculate_se_translation = function(df, n_bins){
  
  # Assign translation categories
  df = df %>%
    mutate(translation_category = ifelse(orf_ann_alt == '{}' |
                                           !(grepl('main ORF', orf_ann_alt)),
                                         'Total loss of translation',
                                         'Some translation'),
           translation_category = factor(translation_category,
                                         levels = c('Total loss of translation',
                                                    'Some translation')))
  
  # Calculate SE in both categories
  df1 = calculate_se(df %>%
                       filter(translation_category == 'Total loss of translation'),
                     'translation', n_bins/5)
  df2 = calculate_se(df %>% filter(translation_category == 'Some translation'),
                     'translation', n_bins)
  
  # Merge dataframes
  df = rbind(df1, df2)
  
  return(df)
}


# SE QC plots
add_theme_textsize = function(p){
  
  p = p +
    theme(axis.title = element_text(size=16),
          axis.text = element_text(size=14),
          legend.title = element_text(size=14),
          legend.text = element_text(size=14))
  
  return(p)
  
}

plot_expected_pulldown_vs_SE = function(df, beta_variable, dir_out_lib){
  
  if (beta_variable == 'diploid'){
    
    pA = ggplot(df, aes(x=expected_pulldown_1, y=se_log2_diploid_fc1)) +
      geom_point() + theme_light() +
      xlab("Expected pulldown 1") +
      ylab(expression(paste("SE of log"[2]*"(diploid FC 1)")))
    
    pB = ggplot(df, aes(x=expected_pulldown_2, y=se_log2_diploid_fc2)) +
      geom_point() + theme_light() +
      xlab("Expected pulldown 2") +
      ylab(expression(paste("SE of log"[2]*"(diploid FC 2)")))
    
    pC = ggplot(df, aes(x=expected_pulldown_3, y=se_log2_diploid_fc3)) +
      geom_point() + theme_light() +
      xlab("Expected pulldown 3") +
      ylab(expression(paste("SE of log"[2]*"(diploid FC 3)")))
    
    pD = ggplot(df, aes(x=expected_pulldown_4, y=se_log2_diploid_fc4)) +
      geom_point() + theme_light() +
      xlab("Expected pulldown 4") +
      ylab(expression(paste("SE of log"[2]*"(diploid FC 4)")))
    
  } else if (beta_variable == 'translation'){
    
    pA = ggplot(df, aes(x=expected_pulldown_1, y=se_translation1,
                        color=factor(translation_category))) +
      geom_point(alpha=0.6) + theme_light() +
      xlab("Expected pulldown 1") +
      ylab(expression(paste("SE of translation 1"))) +
      theme(legend.position='none') +
      scale_color_manual(values=c('black', 'grey'))
    
    pB = ggplot(df, aes(x=expected_pulldown_2, y=se_translation2,
                        color=factor(translation_category))) +
      geom_point(alpha=0.6) + theme_light() +
      xlab("Expected pulldown 2") +
      ylab(expression(paste("SE of translation 2"))) +
      theme(legend.position='none') +
      scale_color_manual(values=c('black', 'grey'))
    
    pC = ggplot(df, aes(x=expected_pulldown_3, y=se_translation3,
                        color=factor(translation_category))) +
      geom_point(alpha=0.6) + theme_light() +
      xlab("Expected pulldown 3") +
      ylab(expression(paste("SE of translation 3"))) +
      theme(legend.position='none') +
      scale_color_manual(values=c('black', 'grey'))
    
    pD = ggplot(df, aes(x=expected_pulldown_4, y=se_translation4,
                        color=factor(translation_category))) +
      geom_point(alpha=0.6) + theme_light() +
      xlab("Expected pulldown 4") +
      ylab(expression(paste("SE of translation 4"))) +
      theme(legend.position='bottom') +
      scale_color_manual(values=c('black', 'grey'), name='')
    
  } else if (beta_variable == 'log2_delta_t'){
    
    pA = ggplot(df, aes(x=expected_pulldown_1, y=se_log2_delta_t1)) +
      geom_point(alpha=0.6) + theme_light() +
      xlab("Expected pulldown 1") +
      ylab(expression(paste("SE of log2 ", Delta, " T1"))) +
      theme(legend.position='none') +
      scale_color_manual(values=c('black', 'grey'))
    
    pB = ggplot(df, aes(x=expected_pulldown_2, y=se_log2_delta_t2)) +
      geom_point(alpha=0.6) + theme_light() +
      xlab("Expected pulldown 2") +
      ylab(expression(paste("SE of log2 ", Delta, " T2"))) +
      theme(legend.position='none') +
      scale_color_manual(values=c('black', 'grey'))
    
    pC = ggplot(df, aes(x=expected_pulldown_3, y=se_log2_delta_t3)) +
      geom_point(alpha=0.6) + theme_light() +
      xlab("Expected pulldown 3") +
      ylab(expression(paste("SE of log2 ", Delta, " T3"))) +
      theme(legend.position='none') +
      scale_color_manual(values=c('black', 'grey'))
    
    pD = ggplot(df, aes(x=expected_pulldown_4, y=se_log2_delta_t4)) +
      geom_point(alpha=0.6) + theme_light() +
      xlab("Expected pulldown 4") +
      ylab(expression(paste("SE of log2 ", Delta, " T4"))) +
      theme(legend.position='none') +
      scale_color_manual(values=c('black', 'grey'))
    
  }
  
  pA = add_theme_textsize(pA)
  pB = add_theme_textsize(pB)
  pC = add_theme_textsize(pC)
  pD = add_theme_textsize(pD)
  
  print(gridExtra::grid.arrange(pA, pB, pC, pD, nrow=2))
  
  # Save file
  png(str_interp('${dir_out_lib}/4_expected_pd_vs_se_${beta_variable}.png'),
      width=800, height=500)
  plot(gridExtra::grid.arrange(pA, pB, pC, pD, nrow=2))
  dev.off()
  
}

plot_expected_pulldown_vs_pulldown = function(df, beta_variable, dir_out_lib){
  
  range1 <- range(log10(df$pulldown_12h_B1_norm[df$pulldown_12h_B1_norm > 0]),
                  log10(df$expected_pulldown_1[df$expected_pulldown_1 > 0]))
  
  range2 <- range(log10(df$pulldown_12h_B2_norm[df$pulldown_12h_B2_norm > 0]),
                  log10(df$expected_pulldown_2[df$expected_pulldown_2 > 0]))
  
  range3 <- range(log10(df$pulldown_12h_B3_norm[df$pulldown_12h_B3_norm > 0]),
                  log10(df$expected_pulldown_3[df$expected_pulldown_3 > 0]))
  
  range4 <- range(log10(df$pulldown_12h_B4_norm[df$pulldown_12h_B4_norm > 0]),
                  log10(df$expected_pulldown_4[df$expected_pulldown_4 > 0]))
  
  if (beta_variable == 'diploid'){
    
    pA = ggplot(df %>%
                  filter(pulldown_12h_B1_norm != 0,
                         expected_pulldown_1 != 0),
                aes(x=log10(pulldown_12h_B1_norm),
                    y=log10(expected_pulldown_1),
                    color=se_log2_diploid_fc1)) +
      geom_point() + theme_light() +
      xlab(expression(log[10]("Observed pulldown 1"))) +
      ylab(expression(log[10]("Expected pulldown 1"))) +
      scale_color_viridis() +
      labs(color = "SE (diploid FC 1)") +
      xlim(range1) + ylim(range1)
    
    pB = ggplot(df %>%
                  filter(pulldown_12h_B2_norm != 0,
                         expected_pulldown_2 != 0),
                aes(x=log10(pulldown_12h_B2_norm),
                    y=log10(expected_pulldown_2),
                    color=se_log2_diploid_fc2)) +
      geom_point() + theme_light() +
      xlab(expression(log[10]("Observed pulldown 2"))) +
      ylab(expression(log[10]("Expected pulldown 2"))) +
      scale_color_viridis() +
      labs(color = "SE (diploid FC 2)") +
      xlim(range2) + ylim(range2)
    
    pC = ggplot(df %>%
                  filter(pulldown_12h_B3_norm != 0,
                         expected_pulldown_3 != 0),
                aes(x=log10(pulldown_12h_B3_norm),
                    y=log10(expected_pulldown_3),
                    color=se_log2_diploid_fc3)) +
      geom_point() + theme_light() +
      xlab(expression(log[10]("Observed pulldown 3"))) +
      ylab(expression(log[10]("Expected pulldown 3"))) +
      scale_color_viridis() +
      labs(color = "SE (diploid FC 3)") +
      xlim(range3) + ylim(range3)
    
    pD = ggplot(df %>%
                  filter(pulldown_12h_B4_norm != 0,
                         expected_pulldown_4 != 0),
                aes(x=log10(pulldown_12h_B4_norm),
                    y=log10(expected_pulldown_4),
                    color=se_log2_diploid_fc4)) +
      geom_point() + theme_light() +
      xlab(expression(log[10]("Observed pulldown 4"))) +
      ylab(expression(log[10]("Expected pulldown 4"))) +
      scale_color_viridis() +
      labs(color = "SE (diploid FC 4)") +
      xlim(range4) + ylim(range4)
    
  } else if (beta_variable == 'translation'){
    
    pA = ggplot(df %>%
                  filter(pulldown_12h_B1_norm != 0,
                         expected_pulldown_1 != 0),
                aes(x=log10(pulldown_12h_B1_norm),
                    y=log10(expected_pulldown_1),
                    color=se_translation1)) +
      geom_point() + theme_light() +
      xlab(expression(log[10]("Observed pulldown 1"))) +
      ylab(expression(log[10]("Expected pulldown 1"))) +
      scale_color_viridis() +
      labs(color = "SE (translation 1)") +
      xlim(range1) + ylim(range1)
    
    pB = ggplot(df %>%
                  filter(pulldown_12h_B2_norm != 0,
                         expected_pulldown_2 != 0),
                aes(x=log10(pulldown_12h_B2_norm),
                    y=log10(expected_pulldown_2),
                    color=se_translation2)) +
      geom_point() + theme_light() +
      xlab(expression(log[10]("Observed pulldown 2"))) +
      ylab(expression(log[10]("Expected pulldown 2"))) +
      scale_color_viridis() +
      labs(color = "SE (translation 2)") +
      xlim(range2) + ylim(range2)
    
    pC = ggplot(df %>%
                  filter(pulldown_12h_B3_norm != 0,
                         expected_pulldown_3 != 0),
                aes(x=log10(pulldown_12h_B3_norm),
                    y=log10(expected_pulldown_3),
                    color=se_translation3)) +
      geom_point() + theme_light() +
      xlab(expression(log[10]("Observed pulldown 3"))) +
      ylab(expression(log[10]("Expected pulldown 3"))) +
      scale_color_viridis() +
      labs(color = "SE (translation 3)") +
      xlim(range3) + ylim(range3) +
      theme(legend.title = element_blank())
    
    pD = ggplot(df %>%
                  filter(pulldown_12h_B4_norm != 0,
                         expected_pulldown_4 != 0),
                aes(x=log10(pulldown_12h_B4_norm),
                    y=log10(expected_pulldown_4),
                    color=se_translation4)) +
      geom_point() + theme_light() +
      xlab(expression(log[10]("Observed pulldown 4"))) +
      ylab(expression(log[10]("Expected pulldown 4"))) +
      scale_color_viridis() +
      labs(color = "SE (translation 4)") +
      xlim(range4) + ylim(range4)
    
  } else if (beta_variable == 'log2_delta_t'){
    
    pA = ggplot(df %>%
                  filter(pulldown_12h_B1_norm != 0,
                         expected_pulldown_1 != 0),
                aes(x=log10(pulldown_12h_B1_norm),
                    y=log10(expected_pulldown_1),
                    color=se_log2_delta_t1)) +
      geom_point() + theme_light() +
      xlab(expression(log[10]("Observed pulldown 1"))) +
      ylab(expression(log[10]("Expected pulldown 1"))) +
      scale_color_viridis() +
      labs(color = expression(paste("SE (log2 ", Delta, " T1)"))) +
      xlim(range1) + ylim(range1)
    
    pB = ggplot(df %>%
                  filter(pulldown_12h_B2_norm != 0,
                         expected_pulldown_2 != 0),
                aes(x=log10(pulldown_12h_B2_norm),
                    y=log10(expected_pulldown_2),
                    color=se_log2_delta_t2)) +
      geom_point() + theme_light() +
      xlab(expression(log[10]("Observed pulldown 2"))) +
      ylab(expression(log[10]("Expected pulldown 2"))) +
      scale_color_viridis() +
      labs(color = expression(paste("SE (log2 ", Delta, " T2)"))) +
      xlim(range2) + ylim(range2)
    
    pC = ggplot(df %>%
                  filter(pulldown_12h_B3_norm != 0,
                         expected_pulldown_3 != 0),
                aes(x=log10(pulldown_12h_B3_norm),
                    y=log10(expected_pulldown_3),
                    color=se_log2_delta_t3)) +
      geom_point() + theme_light() +
      xlab(expression(log[10]("Observed pulldown 3"))) +
      ylab(expression(log[10]("Expected pulldown 3"))) +
      scale_color_viridis() +
      labs(color = expression(paste("SE (log2 ", Delta, " T3)"))) +
      xlim(range3) + ylim(range3)
    
    pD = ggplot(df %>%
                  filter(pulldown_12h_B4_norm != 0,
                         expected_pulldown_4 != 0),
                aes(x=log10(pulldown_12h_B4_norm),
                    y=log10(expected_pulldown_4),
                    color=se_log2_delta_t4)) +
      geom_point() + theme_light() +
      xlab(expression(log[10]("Observed pulldown 4"))) +
      ylab(expression(log[10]("Expected pulldown 4"))) +
      scale_color_viridis() +
      labs(color = expression(paste("SE (log2 ", Delta, " T4)"))) +
      xlim(range4) + ylim(range4)
    
  }
  
  pA = add_theme_textsize(pA)
  pB = add_theme_textsize(pB)
  pC = add_theme_textsize(pC)
  pD = add_theme_textsize(pD)
  
  print(gridExtra::grid.arrange(pA, pB, pC, pD, nrow=2))
  
  # Save file
  png(str_interp('${dir_out_lib}/5_observed_vs_expected_pd_${beta_variable}.png'),
      width=800, height=500)
  plot(gridExtra::grid.arrange(pA, pB, pC, pD, nrow=2))
  dev.off()
  
}


# SE calibration plots, across libraries
compute_95ci_plot = function(df, var_txt, dir_out_plots){
  
  betas = paste(var_txt, c(1:4), sep='')
  ses = paste("se_", var_txt, c(1:4), sep='')
  
  df = df %>%
    rename(beta1 = all_of(betas[1]),
           beta2 = all_of(betas[2]),
           beta3 = all_of(betas[3]),
           beta4 = all_of(betas[4]),
           
           se1 = all_of(ses[1]),
           se2 = all_of(ses[2]),
           se3 = all_of(ses[3]),
           se4 = all_of(ses[4]))
  
  
  df = df %>%
    rowwise() %>%
    mutate(delta_4 = 1/3*sum(c(beta1, beta2, beta3)) - beta4,
           delta_3 = 1/3*sum(c(beta1, beta2, beta4)) - beta3,
           delta_2 = 1/3*sum(c(beta1, beta3, beta4)) - beta2,
           delta_1 = 1/3*sum(c(beta2, beta3, beta4)) - beta1) %>%
    
    mutate(var_4 = 1/9*sum(c(se1^2 + se2^2 + se3^2)) + se4^2,
           var_3 = 1/9*sum(c(se1^2 + se2^2 + se4^2)) + se3^2,
           var_2 = 1/9*sum(c(se1^2 + se3^2 + se4^2)) + se2^2,
           var_1 = 1/9*sum(c(se2^2 + se3^2 + se4^2)) + se1^2) %>%
    
    mutate(ci_lower_1 = delta_1 - 1.96*sqrt(var_1),
           ci_lower_2 = delta_2 - 1.96*sqrt(var_2),
           ci_lower_3 = delta_3 - 1.96*sqrt(var_3),
           ci_lower_4 = delta_4 - 1.96*sqrt(var_4)) %>%
    
    mutate(ci_upper_1 = delta_1 + 1.96*sqrt(var_1),
           ci_upper_2 = delta_2 + 1.96*sqrt(var_2),
           ci_upper_3 = delta_3 + 1.96*sqrt(var_3),
           ci_upper_4 = delta_4 + 1.96*sqrt(var_4))
  
  df = df %>%
    
    rowwise() %>%
    
    mutate(zero_in_ci_1 = ifelse(ci_lower_1 < 0 & ci_upper_1 > 0, TRUE, FALSE),
           zero_in_ci_2 = ifelse(ci_lower_2 < 0 & ci_upper_2 > 0, TRUE, FALSE),
           zero_in_ci_3 = ifelse(ci_lower_3 < 0 & ci_upper_3 > 0, TRUE, FALSE),
           zero_in_ci_4 = ifelse(ci_lower_4 < 0 & ci_upper_4 > 0, TRUE, FALSE))
  
  summary_df = df %>%
    group_by(library) %>%
    summarize(perc_1 = sum(zero_in_ci_1)/n(),
              perc_2 = sum(zero_in_ci_2)/n(),
              perc_3 = sum(zero_in_ci_3)/n(),
              perc_4 = sum(zero_in_ci_4)/n())
  
  summary_df_long <- summary_df %>%
    pivot_longer(cols = starts_with("perc_"),
                 names_to = "replicate",
                 values_to = "percentage") %>%
    mutate(replicate = str_replace(replicate, "perc_", ""))
  
  p = ggplot(summary_df_long, aes(x = library, y = percentage, color=replicate)) +
    geom_point(size = 3, alpha = 0.7,
               position = position_jitter(width = 0.1)) +
    stat_summary(fun = mean, geom = "crossbar",
                 width = 0.5, color = "black", linewidth = 0.8) +
    labs(x = "Library",
         y = "Percentage",
         color = "Replicate") +
    theme_light() +
    geom_hline(yintercept=0.95, color='grey') +
    ggtitle(var_txt) +
    theme(plot.title = element_text(size=14, hjust=0.5, face='bold'),
          axis.text = element_text(size=13),
          axis.title = element_text(size=13),
          legend.text = element_text(size=13),
          legend.title = element_text(size=13))
  
  # Save file
  png(str_interp('${dir_out_plots}/11_se_calibration_${var_txt}.png'),
      width=1200, height=400)
  plot(p)
  dev.off()
}


# Calculate precision-weighted mean and SE
calculate_prec_weighted_mean_and_se <- function(df, value_prefix, se_prefix) {
  
  value_cols <- grep(paste0("^", value_prefix), names(df), value = TRUE)
  se_cols <- grep(paste0("^", se_prefix), names(df), value = TRUE)
  
  pw_mean_col = paste("pw_mean", value_prefix, sep='_')
  pw_se_col = paste("pw", se_prefix, sep='_')
  
  # Calculate precision-weighted means across duplicates
  pw_df = df %>%
    
    rename(mean1 = value_cols[1],
           mean2 = value_cols[2],
           mean3 = value_cols[3],
           mean4 = value_cols[4],
           
           se1 = se_cols[1],
           se2 = se_cols[2],
           se3 = se_cols[3],
           se4 = se_cols[4]) %>%
    
    mutate(weighted_mean_1 = mean1/se1^2,
           weighted_mean_2 = mean2/se2^2,
           weighted_mean_3 = mean3/se3^2,
           weighted_mean_4 = mean4/se4^2) %>%
    
    mutate(weighted_se_1 = 1/se1^2,
           weighted_se_2 = 1/se2^2,
           weighted_se_3 = 1/se3^2,
           weighted_se_4 = 1/se4^2) %>%
    
    rowwise() %>%
    mutate(sum_weighted_means = sum(c(weighted_mean_1, weighted_mean_2,
                                      weighted_mean_3, weighted_mean_4)),
           sum_weighted_se = sum(c(weighted_se_1, weighted_se_2,
                                   weighted_se_3, weighted_se_4))) %>%
    ungroup() %>%
    
    group_by(gene, humvar) %>%
    summarize(pw_mean = sum(sum_weighted_means)/sum(sum_weighted_se),
              pw_se = sqrt(1/sum(sum_weighted_se)))
  
  names(pw_df) = c('gene', 'humvar',
                   pw_mean_col, pw_se_col)
  
  df = merge(df, pw_df, by=c('gene', 'humvar'), all.x=TRUE, all.y=TRUE)
  
  return(df)
  
}


#----------------------------#
# LOAD PREDICTED CATEGORIES  #
#----------------------------#
df_categories_raw = data.frame(fread(
  './data/1m_panel/naptrap_1m_panel_annotated.txt'))

df_categories_raw = df_categories_raw %>%
  rename(insert_seq = 'ALT_sequence')

df_categories = keep_relevant_cols_from_df_categories(df_categories_raw)


#---------------------------------#
# FAULTY REPORTERS TO BE REMOVED  #
#---------------------------------#
df_faulty = data.frame(fread("./data/processed/faulty_reporters_in_750k.csv"))


#--------------------------------#
# PROCESS DATA WITHIN LIBRARIES  #
#--------------------------------#
naptrap_list = list()

for (library in libraries){
  
  #--------------------#
  # LOAD & CLEAN DATA  #
  #--------------------#
  filename = str_interp("./data/naptrap/raw/humvar1m_5utr_ntrap_${library}_v1_pm_reads_counts.csv")
  
  df_raw = read.csv(filename)
  
  df_naptrap = df_raw %>%
    separate(X, into=c('gene', 'humvar'), "_") %>%
    mutate(humvar = str_replace(humvar, "-x", "-*")) %>%
    mutate(library = library)
  
  # Merge NaP-TRAP and categories
  df = merge(df_naptrap, df_categories,
             by.x=c('gene','humvar'),
             by.y=c('gene_name', 'humvar'), all.x=TRUE, all.y=FALSE)
  
  
  #---------------------------#
  # DATA PROCESSING PIPELINE  #
  #---------------------------#
  
  # (A) FILTERING AND NORMALIZATION ----
  
  # Remove spike-ins
  df = df %>%
    filter(!(grepl("spk", humvar)))
  
  # Remove faulty reporters
  if (library %in% libraries_in_750k){
    df = df %>%
      filter(!(humvar %in% df_faulty$humvar))
  }
  
  # Remove genes with missing shet
  df = df %>%
    filter(humvar == 'ref' | !is.na(posterior_hs))
  
  # Remove missing predicted categories (should be 0)
  df = df %>%
    filter(humvar == 'ref' | !is.na(predicted_category))
  
  
  # Remove reporters with 0 input in any of the replicates
  if (remove_input_0_spec){
    df = remove_reporters_with_any_zeros(df)
  }
  
  # Remove genes with median = 0
  if (remove_median_0_spec){
    df = remove_genes_with_0_median(df)
  }
  
  # Normalize (RPM)
  df = normalize_replicates(df, normalization_spec)
  
  
  # (B) COMPUTE TRANSLATION, DELTA T and log2 DIPLOID FC ----
  
  # Calculate translation values as normalized pulldown/input
  df = calculate_translation(df)
  
  # Calculate delta translation (based on median or reference)
  df = calculate_delta_translation(df, reference_spec)
  
  df = calculate_log2_delta_t(df)
  
  # Remove wacky log2(delta t)
  df = df %>% 
    filter(!(log2_delta_t1 %in% c(NA, Inf, -Inf)), 
           !(log2_delta_t2 %in% c(NA, Inf, -Inf)), 
           !(log2_delta_t3 %in% c(NA, Inf, -Inf)), 
           !(log2_delta_t4 %in% c(NA, Inf, -Inf)))
  
  # Calculate fold changes
  df = calculate_diploid_fold_changes(df)
  
  # (C) CALCULATE STANDARD ERRORS ----
  
  # Calculate expected pulldown
  df = calculate_expected_pulldown(df)
  
  if (library == '25_60'){
    n_bins = 30
  } else {
    n_bins = n_bins
  }
  
  df = calculate_se(df, 'log2_diploid_fc', n_bins)
  df = calculate_se(df, 'log2_delta_t', n_bins)
  df = calculate_se_translation(df, n_bins)
  
  # (D) CLEAN UP DATA ----
  df = df %>%
    select(-starts_with("input"),
           -starts_with("pulldown"),
           -starts_with("expected_"),
           -starts_with("median"),
           -translation_category,
           -insert_seq)
  
  # Append list
  naptrap_list[[library]] = df
  
}

df = rbindlist(naptrap_list)
df$library = factor(df$library, levels=libraries)


#---------------------------------------------#
# CALCULATE PRECISION-WEIGHTED MEANS and SEs  #
#---------------------------------------------#

# Calculate precision-weighted mean and SE
duplicates = df$humvar[duplicated(df$humvar)]
df = df %>% mutate(duplicate = ifelse(humvar %in% duplicates, 'yes', 'no'))

df = calculate_prec_weighted_mean_and_se(df, 'log2_diploid_fc', 'se_log2_diploid_fc')
df = calculate_prec_weighted_mean_and_se(df, 'log2_delta_t', 'se_log2_delta_t')
df = calculate_prec_weighted_mean_and_se(df, 'translation', 'se_translation')

# Print number checks
n_genes = length(unique(df$gene))
n_variants = length(unique(df$humvar))
n_rows = nrow(df)

print(str_interp(
  "The full data has ${n_rows} rows, ${n_variants} unique humvar, and ${n_genes} genes."))

print(str_interp(
  "There are ${sum(df$duplicate == 'yes')} duplicates."
))

#------------#
# SAVE DATA  #
#------------#
write.csv(df, str_interp('${dir_data}/humvar_5utr_ntrap_processed_all_for_cs230.csv'),
          row.names=FALSE)

