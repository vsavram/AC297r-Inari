# Use Variational Auto-Encoders to perform imputation

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

#parser = argparse.ArgumentParser()
#parser.add_argument('--config', type=str, default='config.json', help='configuration json file')

    
#args = parser.parse_args()
#with open(args.config) as f:
#    config = json.load(f)

# Set the default parameters
training_epochs = 250 
batch_size = 250
learning_rate = 0.0005
latent_size = 200 
hidden_size_1 = 6000
hidden_size_2 = 2000
beta = 1   
#data_path = "./data_complete.csv"
#corrupt_data_path = "./LGGGBM_missing_10perc_trial_1.csv"
#save_root = "./output/"


#---------------------------------------------------------------------------------------------------
# Perform VAE imputation
#---------------------------------------------------------------------------------------------------

# Import the training set without missing values and with missing values
train_full = pd.read_csv("train_full", index_col=0).values
train_missing = pd.read_csv("train_missing", index_col=0).values

n_row = train_full.shape[1] # dimensionality of data space
#non_missing_row_ind= np.where(np.isfinite(np.sum(data_missing,axis=1)))
na_ind = np.where(np.isnan(train_missing))

sc = StandardScaler()
#data_missing_complete = np.copy(data_missing[non_missing_row_ind[0],:])
sc.fit(train_full)
train_missing[na_ind] = 0
train_missing = sc.transform(train_missing)
train_missing[na_ind] = np.nan
#del data_missing_complete
train_full = sc.transform(train_full)

   
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

# Train the VAE on the full training set
vae = vae.train(data=train_full, training_epochs=training_epochs)
        
#saver = tf.train.Saver()
#save_path = saver.save(vae.sess, save_root+"ep"+str(training_epochs)+"_bs"+str(batch_size)+"_lr"+str(learning_rate)+"_bn"+str(latent_size)+"_opADAM"+"_beta"+str(beta)+"_betaVAE"+".ckpt")

