# Construct gene coexpression networks and analyze topological features

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

os.chdir("/Users/VICTOR/Desktop/Harvard/AC297r/EDA")


# Import the TPM data
tpm_data = pd.read_table("../Data/TPM_Combined/TPM_expression_counts_from_25_maize_lines_from_NAM_population.txt")
# Import the test TPM data
B73_data = pd.read_table("../Data/B73_Only_Data/TPM_expression_counts_from_B73.txt")


# Create a dataframe to exclusively hold the metadata
metadata = tpm_data[['Run', 'growth_condition', 'Cultivar', 
                     'Developmental_stage', 'organism_part', 'Age']]
# Create a dataframe that only holds the expression data
tpm_data = tpm_data.drop(['Run', 'growth_condition', 'Cultivar', 
                          'Developmental_stage', 'organism_part', 'Age'], axis=1)


#---------------------------------------------------------------------------------------------------
# Remove lowly expressed gene and log transform the data
#---------------------------------------------------------------------------------------------------

# Determine lowly expressed genes (not expressed in over 80% of samples and max expression level is less than 1)
lowly_expressed = []
cutoff = 0.8*tpm_data.shape[0]
for gene in tpm_data.columns:
    data = tpm_data[gene]
    # Determine the number of samples that do not express the given gene
    num_no_expression = np.sum(data == 0)
    lowly_expressed.append((num_no_expression >= cutoff) & (np.max(data) < 2))
    
# Remove lowly expressed genes
tpm = tpm_data.loc[:,~np.array(lowly_expressed)]
# Log transform the tpm data
log_tpm = np.log2(tpm + 1)


#---------------------------------------------------------------------------------------------------
# Look at the gene expression distributions
#---------------------------------------------------------------------------------------------------

# Calculate the mean expression level for every gene
mean_expression = log_tpm.mean(axis=0)
len(mean_expression)
# Calculated the mean expression level for every gene per individual
log_tpm['Cultivar'] = metadata['Cultivar']
mean_exp_individual = log_tpm.groupby(by=['Cultivar']).mean()

fig, axes = plt.subplots(1,2,figsize=(14,6))
# Plot the distribution for the mean expression level across all samples across genes
ax = axes.flatten()
sns.distplot(mean_expression, kde=True, color = 'darkblue', 
             kde_kws={'linewidth': 4}, ax = ax[0])
ax[0].set_xlabel('Average Expression per Gene', size = 18)
ax[0].set_ylabel('Density', size = 18)
ax[0].set_title('Average Expression', size = 20)
ax[0].tick_params(axis='both', labelsize=14)
# Plot the distribution for the mean expression level per individual across genes
for individual in mean_exp_individual.index:
    sns.kdeplot(mean_exp_individual.loc[individual], label = individual, ax = ax[1])
ax[1].set_xlabel('Average Expression per Gene', size = 18)
ax[1].set_ylabel('Density', size = 18)
ax[1].set_title('Average Expression (Individual)', size = 20)
ax[1].tick_params(axis='both', labelsize=14)
ax[1].legend(fontsize=14, ncol=2)
plt.tight_layout();
plt.savefig("./Data_EDA/gene_exp_distributions.png")

# Drop the Cultivar column
log_tpm = log_tpm.drop(['Cultivar'], axis=1)


#---------------------------------------------------------------------------------------------------
# Perform dimensionality reduction
#---------------------------------------------------------------------------------------------------

import scprep
from sklearn.manifold import TSNE

# Perform PCA on the data
pca, singular_values = scprep.reduce.pca(log_tpm, n_components = 50, return_singular_values = True)
percent_variance = (singular_values**2/sum(singular_values**2))*100  # Calculate percent variance explained

# Perform TSNE on the data
tsne_operator = TSNE(n_components = 2, perplexity = 30, random_state = 10)  # Perplexity usually set 5-50 (loosely number of neighbors)
tsne = pd.DataFrame(tsne_operator.fit_transform(pca.iloc[:,0:50]))  # Perform tSNE on first 50 PC's

# Create PCA and TSNE plots colored by individual
fig, axes = plt.subplots(1,2,figsize = (18, 6))
ax = axes.flatten()
for individual in np.unique(metadata['Cultivar']):
    indices_to_keep = np.where(metadata['Cultivar'] == individual)[0]
    # Create the PCA plot
    ax[0].scatter(x = pca.iloc[indices_to_keep,0], y = pca.iloc[indices_to_keep,1], 
                    s = 10, alpha = 0.7, label = individual)
    # Create the TSNE plot
    ax[1].scatter(x = tsne.iloc[indices_to_keep,0], y = tsne.iloc[indices_to_keep,1], 
                    s = 10, alpha = 0.7, label = individual)
ax[0].set_xlabel("PC-1", fontsize = 20)
ax[0].set_ylabel("PC-2", fontsize = 20)
ax[0].set_title("PCA", fontsize = 24)
ax[0].tick_params(axis='both', labelsize=14)
ax[1].set_xlabel("TSNE-1", fontsize = 20)
ax[1].set_ylabel("TSNE-2", fontsize = 20)
ax[1].set_title("TSNE", fontsize = 24)
ax[1].tick_params(axis='both', labelsize=14)
ax[1].legend(bbox_to_anchor=(1.05, 1), title = 'individual ID', title_fontsize = 16, fontsize = 14, ncol=2)
plt.tight_layout();
plt.savefig("./Data_EDA/PCA_TSNE_individual.png")

# Create TSNE plots colored by tissue
fig, axes = plt.subplots(1,2,figsize = (16, 6))
ax = axes.flatten()
for tissue in np.unique(metadata['organism_part']):
    indices_to_keep = np.where(metadata['organism_part'] == tissue)[0]
    # Create the PCA plot
    ax[0].scatter(x = pca.iloc[indices_to_keep,0], y = pca.iloc[indices_to_keep,1], 
                    s = 10, alpha = 0.7, label = tissue)
    # Create the TSNE plot
    ax[1].scatter(x = tsne.iloc[indices_to_keep,0], y = tsne.iloc[indices_to_keep,1], 
                    s = 10, alpha = 0.7, label = tissue)
ax[0].set_xlabel("PC-1", fontsize = 20)
ax[0].set_ylabel("PC-2", fontsize = 20)
ax[0].set_title("PCA", fontsize = 24)
ax[0].tick_params(axis='both', labelsize=14)
ax[1].set_xlabel("TSNE-1", fontsize = 20)
ax[1].set_ylabel("TSNE-2", fontsize = 20)
ax[1].set_title("TSNE", fontsize = 24)
ax[1].tick_params(axis='both', labelsize=14)
ax[1].legend(bbox_to_anchor=(1.05, 1), title = 'organism part', title_fontsize = 16, fontsize = 14)
plt.tight_layout();
plt.savefig("./Data_EDA/PCA_TSNE_tissue.png")

