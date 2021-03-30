# Perform imputation on the expression data

import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from autoencodersbetaVAE import VariationalAutoencoder
import pandas as pd
import random
import tensorflow as tf
import sys
import argparse
import json

os.chdir("/Users/VICTOR/Desktop/Harvard/AC297r")

# Import the preprocessed TPM data
train_full = pd.read_csv("./Data/TPM_Combined/preprocessed_log_TPM.csv", index_col=0)
# Import the test TPM data
test_full = pd.read_table("./Data/B73_Only_Data/TPM_expression_counts_from_B73.txt")

# Determine lowly expressed genes (not expressed in over 80% of samples and max expression level is less than 1)
test_full = test_full[test_full.columns.intersection(train_full.columns)]
# Log transform the test data
test_full = np.log2(test_full + 1)

# Determine the number of genes
num_genes = train_full.shape[1]
# Determine the number of entries in the train and test sets
train_n = train_full.shape[0]*train_full.shape[1]
test_n = test_full.shape[0]*test_full.shape[1]


#---------------------------------------------------------------------------------------------------
# Randomly generate missing entries
# Randomly generate missing genes
#---------------------------------------------------------------------------------------------------

# Determine the coordinates for each entry in the expression data
train_coordinates = [[i,j] for i in range(train_full.shape[0]) for j in range(train_full.shape[1])]
test_coordinates = [[i,j] for i in range(test_full.shape[0]) for j in range(test_full.shape[1])]
train_coordinates = np.array(train_coordinates)
test_coordinates = np.array(test_coordinates)

# Randomly generate missing entries (5%, 10%, and 50% of data missing)
train_5perc_missing, test_5perc_missing = [],[]
train_10perc_missing, test_10perc_missing = [],[]
train_50perc_missing, test_50perc_missing = [],[]
for i in range(3):
    
    # Randomly generate missing entries (5% of data missing)
    missing_indices = np.random.choice(range(train_n), size=int(0.05*train_n), replace=False)
    train_5perc_missing.append(train_coordinates[missing_indices])
    missing_indices = np.random.choice(range(test_n), size=int(0.05*test_n), replace=False)
    test_5perc_missing.append(test_coordinates[missing_indices])
    
    # Randomly generate missing entries (10% of data missing)
    missing_indices = np.random.choice(range(train_n), size=int(0.1*train_n), replace=False)
    train_10perc_missing.append(train_coordinates[missing_indices])
    missing_indices = np.random.choice(range(test_n), size=int(0.1*test_n), replace=False)
    test_10perc_missing.append(test_coordinates[missing_indices])
    
    # Randomly generate missing entries (50% of data missing)
    missing_indices = np.random.choice(range(train_n), size=int(0.5*train_n), replace=False)
    train_50perc_missing.append(train_coordinates[missing_indices])
    missing_indices = np.random.choice(range(test_n), size=int(0.5*test_n), replace=False)
    test_50perc_missing.append(test_coordinates[missing_indices])

# Randomly generate missing entries by picking a subset of genes (5%, 10%, and 50% of the genes have no data)
train_5perc_genes, test_5perc_genes = [],[]
train_10perc_genes, test_10perc_genes = [],[]
train_50perc_genes, test_50perc_genes = [],[]
for i in range(3):
    
    # Randomly generate missing entries (5% of the genes will have no data)
    missing_indices = np.random.choice(range(num_genes), size=int(0.05*num_genes), replace=False)
    train_5perc_genes.append(missing_indices)
    missing_indices = np.random.choice(range(num_genes), size=int(0.05*num_genes), replace=False)
    test_5perc_genes.append(missing_indices)
    
    # Randomly generate missing entries (10% of the genes will have no data)
    missing_indices = np.random.choice(range(num_genes), size=int(0.1*num_genes), replace=False)
    train_10perc_genes.append(missing_indices)
    missing_indices = np.random.choice(range(num_genes), size=int(0.1*num_genes), replace=False)
    test_10perc_genes.append(missing_indices) 
    
    # Randomly generate missing entries (50% of the genes will have no data)
    missing_indices = np.random.choice(range(num_genes), size=int(0.5*num_genes), replace=False)
    train_50perc_genes.append(missing_indices)
    missing_indices = np.random.choice(range(num_genes), size=int(0.5*num_genes), replace=False)
    test_50perc_genes.append(missing_indices)
    

#---------------------------------------------------------------------------------------------------
### TRAIN THE VAE ###
#---------------------------------------------------------------------------------------------------

# Turn off eager execution (eager execution is on by default in Tensorflow 2)
tf.compat.v1.disable_eager_execution()

# Set the default parameters
training_epochs = 250 
batch_size = 250
learning_rate = 0.0005
latent_size = 200 
hidden_size_1 = 6000
hidden_size_2 = 2000
beta = 1  

n_row = train_full.shape[1] # dimensionality of data space

# Set the number of imputer iterations
ImputeIter = 3

# Define the VAE network size
Decoder_hidden1 = hidden_size_1 
Decoder_hidden2 = hidden_size_2 
Encoder_hidden1 = hidden_size_2 
Encoder_hidden2 = hidden_size_1 
        
        
# define dict for network structure:
network_architecture = dict(n_hidden_recog_1=Encoder_hidden1, # 1st layer encoder neurons
                            n_hidden_recog_2=Encoder_hidden2, # 2nd layer encoder neurons
                            n_hidden_gener_1=Decoder_hidden1, # 1st layer decoder neurons
                            n_hidden_gener_2=Decoder_hidden2, # 2nd layer decoder neurons
                            n_input=n_row, # data input size
                            n_z=latent_size)  # dimensionality of latent space

# Initialize the VAE
vae = VariationalAutoencoder(network_architecture,
                             learning_rate=learning_rate, 
                             batch_size=batch_size,istrain=True,restore_path=None,
                             beta=beta)

# Fit the standard scaler to the full training set
sc = StandardScaler()
sc.fit(train_full)
# Transform the full training set
train_full_scaled = sc.transform(train_full)

# Train the VAE on the full training set
vae = vae.train(data=train_full_scaled, training_epochs=training_epochs)


#---------------------------------------------------------------------------------------------------
# Perform VAE imputation - missing entries
#---------------------------------------------------------------------------------------------------

# Define a function that performs VAE imputation on a training set and test set
def VAE_impute(train_full, test_full, train_missing_list, test_missing_list):
    # Iterate over the different sets of missing values 
    train_r2,test_r2 = [],[]
    for train_missing_indices,test_missing_indices in zip(train_missing_list,test_missing_list): 
        
        ### TRAINING SET IMPUTATION ###
        # Set the missing values in the training set
        train_missing = train_full.copy()
        for indices in train_missing_indices:
            train_missing.iloc[indices[0],indices[1]] = np.nan
           
        # Determine where the missing values occur for the training set
        na_ind = np.where(np.isnan(train_missing))
        na_count = len(na_ind[0])
        # Scale the training set with missing values
        train_missing[na_ind] = 0
        train_missing = sc.transform(train_missing)
        train_missing[na_ind] = np.nan
        # Impute missing values for the training set
        train_impute = vae.impute(data_corrupt = train_missing, max_iter = ImputeIter)
        
        # Inverse transform the imputed data (remove scaling)
        train_impute = sc.inverse_transform(train_impute)
        
        # Determine the R2 score for the TRAINING SET
        y_true,y_pred = [],[]
        for row_index,col_index in zip(na_ind[0],na_ind[1]):
            y_true.append(train_full[row_index,col_index]/2)
            y_pred.append(train_impute[row_index,col_index])
        # Calculate the R2 score
        r2 = r2_score(y_true, y_pred)
        train_r2.append(r2)
        
        ### TEST SET IMPUTATION ###
        # Set the missing values in the test set
        test_missing = test_full.copy()
        for indices in test_missing_indices:
            test_missing.iloc[indices[0],indices[1]] = np.nan
            
        # Determine where the missing values occur for the training set
        na_ind = np.where(np.isnan(test_missing))
        na_count = len(na_ind[0])
        # Scale the training set with missing values
        test_missing[na_ind] = 0
        test_missing = sc.transform(test_missing)
        test_missing[na_ind] = np.nan
        # Impute missing values for the training set
        test_impute = vae.impute(data_corrupt = test_missing, max_iter = ImputeIter)
        
        # Inverse transform the imputed data (remove scaling)
        test_impute = sc.inverse_transform(test_impute)
        
        # Determine the R2 score for the TEST SET
        y_true,y_pred = [],[]
        for row_index,col_index in zip(na_ind[0],na_ind[1]):
            y_true.append(test_full[row_index,col_index]/2)
            y_pred.append(test_impute[row_index,col_index])
        # Calculate the R2 score
        r2 = r2_score(y_true, y_pred)
        test_r2.append(r2)

    # Compute the average R2 score for the training and test sets
    mean_train_r2 = np.mean(train_r2)
    mean_test_r2 = np.mean(test_r2)
    
    return mean_train_r2,mean_test_r2


# Perform imputation for 5% missing entries
train_perc5_r2,test_perc5_r2 = VAE_impute(train_full, test_full, train_5perc_missing, test_5perc_missing)

# Perform imputation for 10% missing entries
train_perc10_r2,test_perc10_r2 = VAE_impute(train_full, test_full, train_10perc_missing, test_10perc_missing)

# Perform imputation for 50% missing entries
train_perc50_r2,test_perc50_r2 = VAE_impute(train_full, test_full, train_50perc_missing, test_50perc_missing)

# Create a dataframe to store the R2 scores
r2_df = pd.DataFrame({'training R2': [train_perc5_r2, train_perc10_r2, train_perc50_r2],
                      'test R2': [test_perc5_r2, test_perc10_r2, test_perc50_r2]}, 
                     index = ['5% missing', '10% missing', '50% missing'])


### Imputation on randomly selected missing genes - some genes have no data ###

# Define a function that performs KNN imputation on a training set and test set
def VAE_impute_2(train_full, test_full, train_missing_list, test_missing_list):
    # Iterate over the different sets of missing values 
    train_r2,test_r2 = [],[]
    for train_missing_indices,test_missing_indices in zip(train_missing_list,test_missing_list): 
        
        ### TRAINING SET IMPUTATION ###
        # Set the missing values in the training set
        train_missing = train_full.copy()
        train_missing.iloc[:,train_missing_indices] = np.nan
        
        train_missing = train_missing.to_numpy()
        # Determine where the missing values occur for the training set
        na_ind = np.where(np.isnan(train_missing))
        na_count = len(na_ind[0])
        # Scale the training set with missing values
        train_missing[na_ind] = 0
        train_missing = sc.transform(train_missing)
        train_missing[na_ind] = np.nan
        # Impute missing values for the training set
        train_impute = vae.impute(data_corrupt = train_missing, max_iter = ImputeIter)
        
        # Inverse transform the imputed data (remove scaling)
        train_impute = sc.inverse_transform(train_impute)
        
        # Determine the R2 score for the TRAINING SET
        y_true,y_pred = [],[]
        for row_index,col_index in zip(na_ind[0],na_ind[1]):
            y_true.append(train_full[row_index,col_index]/2)
            y_pred.append(train_impute[row_index,col_index])
        # Calculate the R2 score
        r2 = r2_score(y_true, y_pred)
        train_r2.append(r2)
    
        ### TEST SET IMPUTATION ###
        # Set the missing values in the test set
        test_missing = test_full.copy()
        test_missing.iloc[:,test_missing_indices] = np.nan
        
        test_missing = test_missing.to_numpy()
        # Determine where the missing values occur for the training set
        na_ind = np.where(np.isnan(test_missing))
        na_count = len(na_ind[0])
        # Scale the training set with missing values
        test_missing[na_ind] = 0
        test_missing = sc.transform(test_missing)
        test_missing[na_ind] = np.nan
        # Impute missing values for the training set
        test_impute = vae.impute(data_corrupt = test_missing, max_iter = ImputeIter)
        
        # Inverse transform the imputed data (remove scaling)
        test_impute = sc.inverse_transform(test_impute)
        
        # Determine the R2 score for the TEST SET
        y_true,y_pred = [],[]
        for row_index,col_index in zip(na_ind[0],na_ind[1]):
            y_true.append(test_full[row_index,col_index]/2)
            y_pred.append(test_impute[row_index,col_index])
        # Calculate the R2 score
        r2 = r2_score(y_true, y_pred)
        test_r2.append(r2)

    # Compute the average R2 score for the training and test sets
    mean_train_r2 = np.mean(train_r2)
    mean_test_r2 = np.mean(test_r2)
        
    return mean_train_r2,mean_test_r2


# Perform imputation for 5% missing genes
train_perc5_r2,test_perc5_r2 = VAE_impute_2(train_full, test_full, train_5perc_genes, test_5perc_genes)

# Perform imputation for 10% missing genes
train_perc10_r2,test_perc10_r2 = VAE_impute_2(train_full, test_full, train_10perc_genes, test_10perc_genes)

# Perform imputation for 50% missing genes
train_perc50_r2,test_perc50_r2 = VAE_impute_2(train_full, test_full, train_50perc_genes, test_50perc_genes)

# Create a dataframe to store the R2 scores
r2_df = pd.DataFrame({'training R2': [train_perc5_r2, train_perc10_r2, train_perc50_r2],
                      'test R2': [test_perc5_r2, test_perc10_r2, test_perc50_r2]}, 
                     index = ['5% missing', '10% missing', '50% missing'])


# Iterate over the different sets of missing values 
train_r2,test_r2 = [],[]
for train_missing,test_missing in zip(train_5perc_missing,test_5perc_missing): 
    
    ### TRAINING SET IMPUTATION ###
    # Set the missing values in the training set
    train_missing = tpm_data.copy()
    for indices in train_missing:
        train_missing.iloc[indices[0],indices[1]] = np.nan
       
    # Determine where the missing values occur for the training set
    na_ind = np.where(np.isnan(train_missing))
    na_count = len(na_ind[0])
    # Scale the training set with missing values
    train_missing[na_ind] = 0
    train_missing = sc.transform(train_missing)
    train_missing[na_ind] = np.nan
    # Impute missing values for the training set
    train_impute = vae.impute(data_corrupt = train_missing, max_iter = ImputeIter)
    
    # Inverse transform the imputed data (remove scaling)
    train_impute = sc.inverse_transform(train_impute)
    
    # Determine the R2 score for the TRAINING SET
    y_true,y_pred = [],[]
    for row_index,col_index in zip(na_ind[0],na_ind[1]):
        y_true.append(train_full[row_index,col_index]/2)
        y_pred.append(train_impute[row_index,col_index])
    # Calculate the R2 score
    r2 = r2_score(y_true, y_pred)
    train_r2.append(r2)
    
    ### TEST SET IMPUTATION ###
    # Set the missing values in the test set
    test_missing = test_data.copy()
    for indices in test_missing:
        test_missing.iloc[indices[0],indices[1]] = np.nan
        
    # Determine where the missing values occur for the training set
    na_ind = np.where(np.isnan(test_missing))
    na_count = len(na_ind[0])
    # Scale the training set with missing values
    test_missing[na_ind] = 0
    test_missing = sc.transform(test_missing)
    test_missing[na_ind] = np.nan
    # Impute missing values for the training set
    test_impute = vae.impute(data_corrupt = test_missing, max_iter = ImputeIter)
    
    # Inverse transform the imputed data (remove scaling)
    test_impute = sc.inverse_transform(test_impute)
    
    # Determine the R2 score for the TRAINING SET
    y_true,y_pred = [],[]
    for row_index,col_index in zip(na_ind[0],na_ind[1]):
        y_true.append(train_full[row_index,col_index]/2)
        y_pred.append(train_impute[row_index,col_index])
    # Calculate the R2 score
    r2 = r2_score(y_true, y_pred)
    test_r2.append(r2)

# Compute the average R2 score for the training and test sets
mean_train_r2_5perc = np.mean(train_r2)
mean_test_r2_5perc = np.mean(test_r2)

#---------------------------------------------------------------------------------------------------
# Perform VAE imputation - missing genes
#---------------------------------------------------------------------------------------------------


train_missing = train_5perc_missing[0]
test_missing = test_5perc_missing[0]

    
# Set the missing values in the training set
train_set = tpm_data.copy()
for indices in train_missing:
    train_set.iloc[indices[0],indices[1]] = np.nan
# Set the missing values in the test set
test_set = test_data.copy()
for indices in test_missing:
    test_set.iloc[indices[0],indices[1]] = np.nan
        
train_full = tpm_data.iloc[:,0:7000]
train_mising = train_set.iloc[:,0:7000]
test_full = test_data.iloc[:,0:7000]
test_missing = test_set.iloc[:,0:7000]

train_full.to_csv("train_full.csv")
train_mising.to_csv("train_mising.csv")
test_full.to_csv("test_full.csv")
test_missing.to_csv("test_missing.csv")

