# Perform imputation on the expression data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score

os.chdir("/Users/VICTOR/Desktop/Harvard/AC297r")

# Import the preprocessed TPM data
tpm_data = pd.read_csv("./Data/TPM_Combined/preprocessed_log_TPM.csv", index_col=0)
# Import the test TPM data
test_data = pd.read_table("./Data/B73_Only_Data/TPM_expression_counts_from_B73.txt")

# Determine lowly expressed genes (not expressed in over 80% of samples and max expression level is less than 1)
test_data = test_data[test_data.columns.intersection(tpm_data.columns)]
# Log transform the test data
test_data = np.log2(test_data + 1)

# Determine the number of genes
num_genes = tpm_data.shape[1]
# Determine the number of entries in the train and test sets
train_n = tpm_data.shape[0]*tpm_data.shape[1]
test_n = test_data.shape[0]*test_data.shape[1]


#---------------------------------------------------------------------------------------------------
# Randomly generate missing entries
# Randomly generate missing genes
#---------------------------------------------------------------------------------------------------

# Determine the coordinates for each entry in the expression data
train_coordinates = [[i,j] for i in range(tpm_data.shape[0]) for j in range(tpm_data.shape[1])]
test_coordinates = [[i,j] for i in range(test_data.shape[0]) for j in range(test_data.shape[1])]
train_coordinates = np.array(train_coordinates)
test_coordinates = np.array(test_coordinates)

# Randomly generate missing entries (5% and 10% of data missing)
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

# Randomly generate missing entries by picking a subset of genes (5% and 10% of the genes have no data)
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
# Perform KNN imputation
#---------------------------------------------------------------------------------------------------

from sklearn.impute import KNNImputer


# Initialize the KNN imputer
imputer = KNNImputer(n_neighbors=10)
# Fit the imputer on the training data
imputer.fit(tpm_data)


### Imputation on randomly selected missing entries ###

# Define a function that performs KNN imputation on a training set and test set
def KNN_impute(train_full, test_full, train_missing_list, test_missing_list):
    # Iterate over the different sets of missing values 
    train_r2,test_r2 = [],[]
    for train_missing,test_missing in zip(train_missing_list,test_missing_list): 
        # Set the missing values in the training set
        train_set = train_full.copy()
        for indices in train_missing:
            train_set.iloc[indices[0],indices[1]] = np.nan
        # Set the missing values in the test set
        test_set = test_full.copy()
        for indices in test_missing:
            test_set.iloc[indices[0],indices[1]] = np.nan
        
        # Impute the missing values on the training set
        imputed_train = imputer.transform(train_set)
        y_true = [train_full.iloc[indices[0],indices[1]] for indices in train_missing]
        y_pred = [imputed_train[indices[0],indices[1]] for indices in train_missing]
        train_r2.append(r2_score(y_true,y_pred))
    
        # Impute the missing values on the test set
        imputed_test = imputer.transform(test_set)
        y_true = [test_full.iloc[indices[0],indices[1]] for indices in test_missing]
        y_pred = [imputed_test[indices[0],indices[1]] for indices in test_missing]
        test_r2.append(r2_score(y_true,y_pred))
    
    # Compute the average R2 score for the training and test sets
    mean_train_r2 = np.mean(train_r2)
    mean_test_r2 = np.mean(test_r2)

    return mean_train_r2,mean_test_r2


# Perform imputation for 5% missing entries
train_perc5_r2,test_perc5_r2 = KNN_impute(tpm_data, test_data, train_5perc_missing, test_5perc_missing)

# Perform imputation for 10% missing entries
train_perc10_r2,test_perc10_r2 = KNN_impute(tpm_data, test_data, train_10perc_missing, test_10perc_missing)

# Perform imputation for 50% missing entries
train_perc50_r2,test_perc50_r2 = KNN_impute(tpm_data, test_data, train_50perc_missing, test_50perc_missing)

# Create a dataframe to store the R2 scores
r2_df = pd.DataFrame({'training R2': [train_perc5_r2, train_perc10_r2, train_perc50_r2],
                      'test R2': [test_perc5_r2, test_perc10_r2, test_perc50_r2]}, 
                     index = ['5% missing', '10% missing', '50% missing'])
r2_df.to_csv("KNN_missing_entries_results.csv")


### Imputation on randomly selected missing genes - some genes have no data ###

# Define a function that performs KNN imputation on a training set and test set
def KNN_impute_2(train_full, test_full, train_missing_list, test_missing_list):
    # Iterate over the different sets of missing values 
    train_r2,test_r2 = [],[]
    for train_missing,test_missing in zip(train_missing_list,test_missing_list): 
        # Set the missing values in the training set
        train_set = train_full.copy()
        train_set.iloc[:,train_missing] = np.nan
        # Set the missing values in the test set
        test_set = test_full.copy()
        test_set.iloc[:,test_missing] = np.nan
        
        # Impute the missing values on the training set
        imputed_train = imputer.transform(train_set)
        y_true = train_full.iloc[:,train_missing].to_numpy().ravel()
        y_pred = imputed_train[:,train_missing].ravel()
        train_r2.append(r2_score(y_true,y_pred))
    
        # Impute the missing values on the test set
        imputed_test = imputer.transform(test_set)
        y_true = test_full.iloc[:,test_missing].to_numpy().ravel()
        y_pred = imputed_test[:,test_missing].ravel()
        test_r2.append(r2_score(y_true,y_pred))
    
    # Compute the average R2 score for the training and test sets
    mean_train_r2 = np.mean(train_r2)
    mean_test_r2 = np.mean(test_r2)

    return mean_train_r2,mean_test_r2


# Perform imputation for 5% missing genes
train_perc5_r2,test_perc5_r2 = KNN_impute_2(tpm_data, test_data, train_5perc_genes, test_5perc_genes)

# Perform imputation for 10% missing genes
train_perc10_r2,test_perc10_r2 = KNN_impute_2(tpm_data, test_data, train_10perc_genes, test_10perc_genes)

# Perform imputation for 50% missing genes
train_perc50_r2,test_perc50_r2 = KNN_impute_2(tpm_data, test_data, train_50perc_genes, test_50perc_genes)

# Create a dataframe to store the R2 scores
r2_df = pd.DataFrame({'training R2': [train_perc5_r2, train_perc10_r2, train_perc50_r2],
                      'test R2': [test_perc5_r2, test_perc10_r2, test_perc50_r2]}, 
                     index = ['5% missing genes', '10% missing genes', '50% missing genes'])
r2_df.to_csv("KNN_missing_genes_results.csv")

### Imputation on randomly selected missing entries - 10% ###

# Iterate over the different sets of missing values 
train_r2,test_r2 = [],[]
for train_missing,test_missing in zip(train_10perc_missing,test_10perc_missing): 
    # Set the missing values in the training set
    train_set = tpm_data.copy()
    for indices in train_missing:
        train_set.iloc[indices[0],indices[1]] = np.nan
    # Set the missing values in the test set
    test_set = test_data.copy()
    for indices in test_missing:
        test_set.iloc[indices[0],indices[1]] = np.nan
    
    # Impute the missing values on the training set
    imputed_train = imputer.transform(train_set)
    y_true = [tpm_data.iloc[indices[0],indices[1]] for indices in train_missing]
    y_pred = [imputed_train[indices[0],indices[1]] for indices in train_missing]
    train_r2.append(r2_score(y_true,y_pred))

    # Impute the missing values
    imputed_test = imputer.transform(test_set)
    y_true = [test_data.iloc[indices[0],indices[1]] for indices in test_missing]
    y_pred = [imputed_test[indices[0],indices[1]] for indices in test_missing]
    test_r2.append(r2_score(y_true,y_pred))

# Compute the average R2 score for the training and test sets
mean_train_r2_10perc = np.mean(train_r2)
mean_test_r2_10perc = np.mean(test_r2)


### Imputation on randomly selected missing genes - 5% of genes have no data ###

# Iterate over the different sets of missing values 
train_r2,test_r2 = [],[]
for train_missing,test_missing in zip(train_5perc_genes,test_5perc_genes): 
    # Set the missing values in the training set
    train_set = tpm_data.copy()
    train_set.iloc[:,train_missing] = np.nan
    # Set the missing values in the test set
    test_set = test_data.copy()
    test_set.iloc[:,test_missing] = np.nan
    
    # Impute the missing values on the training set
    imputed_train = imputer.transform(train_set)
    y_true = tpm_data.iloc[:,train_missing].to_numpy().ravel()
    y_pred = imputed_train[:,train_missing].ravel()
    train_r2.append(r2_score(y_true,y_pred))

    # Impute the missing values
    imputed_test = imputer.transform(test_set)
    y_true = test_data.iloc[:,test_missing].to_numpy().ravel()
    y_pred = imputed_test[:,test_missing].ravel()
    test_r2.append(r2_score(y_true,y_pred))

# Compute the average R2 score for the training and test sets
mean_train_r2_5perc_genes = np.mean(train_r2)
mean_test_r2_5perc_genes = np.mean(test_r2)


### Imputation on randomly selected missing genes - 10% of genes have no data ###

# Iterate over the different sets of missing values 
train_r2,test_r2 = [],[]
for train_missing,test_missing in zip(train_10perc_genes,test_10perc_genes): 
    # Set the missing values in the training set
    train_set = tpm_data.copy()
    train_set.iloc[:,train_missing] = np.nan
    # Set the missing values in the test set
    test_set = test_data.copy()
    test_set.iloc[:,test_missing] = np.nan
    
    # Impute the missing values on the training set
    imputed_train = imputer.transform(train_set)
    y_true = tpm_data.iloc[:,train_missing].to_numpy().ravel()
    y_pred = imputed_train[:,train_missing].ravel()
    train_r2.append(r2_score(y_true,y_pred))

    # Impute the missing values
    imputed_test = imputer.transform(test_set)
    y_true = test_data.iloc[:,test_missing].to_numpy().ravel()
    y_pred = imputed_test[:,test_missing].ravel()
    test_r2.append(r2_score(y_true,y_pred))

# Compute the average R2 score for the training and test sets
mean_train_r2_10perc_genes = np.mean(train_r2)
mean_test_r2_10perc_genes = np.mean(test_r2)