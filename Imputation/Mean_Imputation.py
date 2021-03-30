# Perform imputation on the expression data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score

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
# Determine the mean for every gene in the training set
#---------------------------------------------------------------------------------------------------

mean_exp = np.mean(train_full, axis=0)


def mean_imputation(train_full, test_full, train_missing_list, test_missing_list, mean_exp):
    # Iterate over the different sets of missing values 
    train_r2,test_r2 = [],[]
    for train_missing_indices,test_missing_indices in zip(train_missing_list,test_missing_list): 
        
        ### TRAINING SET IMPUTATION ###
        # Set the missing values in the training set
        imputed_train = train_full.copy()
        for indices in train_missing_indices:
            imputed_train.iloc[indices[0],indices[1]] = mean_exp[indices[1]]
            
        # Impute the missing values on the training set
        y_true = [train_full.iloc[indices[0],indices[1]] for indices in train_missing_indices]
        y_pred = [imputed_train.iloc[indices[0],indices[1]] for indices in train_missing_indices]
        train_r2.append(r2_score(y_true,y_pred))
        
        ### TRAINING SET IMPUTATION ###
        # Set the missing values in the training set
        imputed_test = test_full.copy()
        for indices in test_missing_indices:
            imputed_test.iloc[indices[0],indices[1]] = mean_exp[indices[1]]
            
        # Impute the missing values on the training set
        y_true = [test_full.iloc[indices[0],indices[1]] for indices in test_missing_indices]
        y_pred = [imputed_test.iloc[indices[0],indices[1]] for indices in test_missing_indices]
        test_r2.append(r2_score(y_true,y_pred))
        
    # Compute the average R2 score for the training and test sets
    mean_train_r2 = np.mean(train_r2)
    mean_test_r2 = np.mean(test_r2)

    return mean_train_r2,mean_test_r2


# Perform imputation for 5% missing entries
train_perc5_r2,test_perc5_r2 = mean_imputation(train_full, test_full, train_5perc_missing, test_5perc_missing, mean_exp)

# Perform imputation for 10% missing entries
train_perc10_r2,test_perc10_r2 = mean_imputation(train_full, test_full, train_10perc_missing, test_10perc_missing, mean_exp)

# Perform imputation for 50% missing entries
train_perc50_r2,test_perc50_r2 = mean_imputation(train_full, test_full, train_50perc_missing, test_50perc_missing, mean_exp)

# Create a dataframe to store the R2 scores
r2_df = pd.DataFrame({'training R2': [train_perc5_r2, train_perc10_r2, train_perc50_r2],
                      'test R2': [test_perc5_r2, test_perc10_r2, test_perc50_r2]}, 
                     index = ['5% missing', '10% missing', '50% missing'])
r2_df.to_csv("mean_missing_entries_results.csv")



def mean_imputation_2(train_full, test_full, train_missing_list, test_missing_list, mean_exp):
    # Iterate over the different sets of missing values 
    train_r2,test_r2 = [],[]
    for train_missing_indices,test_missing_indices in zip(train_missing_list,test_missing_list): 
        
        ### TRAINING SET IMPUTATION ###
        # Set the missing values in the training set
        imputed_train = train_full.copy()
        for col_index in train_missing_indices:
            imputed_train.iloc[:,col_index] = mean_exp[col_index]
            
        # Impute the missing values on the training set
        y_true = train_full.iloc[:,train_missing_indices].to_numpy().ravel()
        y_pred = imputed_train.iloc[:,train_missing_indices].to_numpy().ravel()
        train_r2.append(r2_score(y_true,y_pred))
        
        ### TRAINING SET IMPUTATION ###
        # Set the missing values in the training set
        imputed_test = test_full.copy()
        for col_index in test_missing_indices:
            imputed_test.iloc[:,col_index] = mean_exp[col_index]
            
        # Impute the missing values on the training set
        y_true = test_full.iloc[:,test_missing_indices].to_numpy().ravel()
        y_pred = imputed_test.iloc[:,test_missing_indices].to_numpy().ravel()
        test_r2.append(r2_score(y_true,y_pred))
        
    # Compute the average R2 score for the training and test sets
    mean_train_r2 = np.mean(train_r2)
    mean_test_r2 = np.mean(test_r2)

    return mean_train_r2,mean_test_r2


# Perform imputation for 5% missing genes
train_perc5_r2,test_perc5_r2 = mean_imputation_2(train_full, test_full, train_5perc_genes, test_5perc_genes, mean_exp)

# Perform imputation for 10% missing genes
train_perc10_r2,test_perc10_r2 = mean_imputation_2(train_full, test_full, train_10perc_genes, test_10perc_genes, mean_exp)

# Perform imputation for 50% missing genes
train_perc50_r2,test_perc50_r2 = mean_imputation_2(train_full, test_full, train_50perc_genes, test_50perc_genes, mean_exp)

# Create a dataframe to store the R2 scores
r2_df = pd.DataFrame({'training R2': [train_perc5_r2, train_perc10_r2, train_perc50_r2],
                      'test R2': [test_perc5_r2, test_perc10_r2, test_perc50_r2]}, 
                     index = ['5% missing', '10% missing', '50% missing'])
r2_df.to_csv("mean_missing_genes_results.csv")

