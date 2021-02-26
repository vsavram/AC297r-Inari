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

df_genes = data.drop(columns=['Run', 'growth_condition', 'Cultivar', 'Developmental_stage', 'organism_part', 'Age'])
