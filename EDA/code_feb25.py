import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Place expression data folder in same directory as this script
# Rename subfolder name to "Combined"
data = pd.read_csv('./Expression_data/Combined/TPM_expression_counts_from_25_maize_lines_from_NAM_population.txt', delimiter='\t')
data.head()

print('Number of runs, i.e. samples:', len(data))
print('Number of genes (appr.):', len(data.columns))

# Describe gene expression levels

# removing null values to avoid errors  
data.dropna(inplace = True) 

# percentile list 
perc =[.20, .40, .60, .80] 

# list of dtypes to include 
include =['object', 'float', 'int'] 
  
# calling describe method 
desc = data.describe(percentiles = perc, include = include) 
  
# display 
desc 

data = data.set_index('Run')
data = data.drop(columns=['growth_condition', 'Cultivar', 'Developmental_stage', 'organism_part', 'Age'])

max_gene_level = data.max(axis=0)
print('Max gene expression level of all genes:', max_gene_level.max())
print('Max gene expression levels by quantile:', max_gene_level.quantile([.1, .15, .2, .25, .3, .5, .75, .8, .9, 1]))

print('Number of genes whose max expression level is 0:', max_gene_level.isin([0]).sum())
print('Number of genes whose max expression level is less than 1 TPM:', max_gene_level[max_gene_level < 1.].count())
print('Number of genes whose max expression level is less than 5 TPM:', max_gene_level[max_gene_level < 5.].count())
print('Number of genes whose max expression level is less than 10 TPM:', max_gene_level[max_gene_level < 10.].count())
