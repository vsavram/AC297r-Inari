'''
Master plan for world domination

- Import data
    - For pytorch geometric, this is a subclass of InMemoryDataset
- Choose:
    - Train test split; masking
    - K-fold
    - Simple p/1-p split
- Choose:
    - Imputation
    - Prediction
- Train
    - How does training happen?
    - Neural network (separate class)
    - Save model
- Evaluate
    - Depending on prediction/imputation
- Output predicted values
'''


import torch
from torch_geometric.data import Data
import numpy as np
import argparse
from models import FAE_GraphConv, FAE_GCN, FAE_SAGEConv, FAE_FeatGraphConv, AE_MLP, Embedding_ExpGAE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from predict import predict
from impute import impute


def get_options(args=None):
    parser = argparse.ArgumentParser(description="Graph Feature Auto-Encoder")

    # Define the arguments for specifying the data location
    parser.add_argument('network_path' ,'--network_path', type=str, default='../data/maize_ppi.npy', help='Path to network')
    parser.add_argument('exp_path' ,'--exp_path', type=str, default='../data/maize_expressions.npy', help='Path to expression data')

    # Define an argument that specifies the amount of masking
    parser.add_argument('percent_masking' ,'--percent_masking', type=float, default=0.3, help='Amount of masking')
    
    parser.add_argument('--model', type=str, default='FeatGraphConv', help="Values in ['GraphConv', 'GCN', 'SAGEConv',"
                                                                           "'FeatGraphConv','MLP' ,"
                                                                           " 'Magic', 'LR', 'RF']")
    parser.add_argument('--embedding', action='store_true', help='Whether to make predictions on the graph embedding '
                                                                 '(only in prediction problem)')
    parser.add_argument

    opts = parser.parse_args(args)
    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda

    return opts


def import_expressions(path):
    """Import data representing expression levels of genes.
    
    Parameters
    ----------
    path : str
        path to expression data in .npy format. The columns are the
        samples; the rows are the genes. 
    
    Returns
    -------
    ndarray of float
    """
    exp_data = np.array(np.load(path, allow_pickle=True), dtype=np.float)

    return exp_data

def import_graph(path):
    """Import data representing network relations between genes, such as
    protein-protein interactions, with proteins replaced with corresponding
    genes.
    
    Parameters
    ----------
    path : str 
        path to graph data in .npy format. Each columns contains two genes
        whose proteins have been computationally predicted to have interactions.
        The genes are represented by integers that correspond to the indices of the
        rows in the expression data where the same genes occupy. i.e. If one column
        of the PPI file is [3, 10]， it indicates that genes in the 3rd and the 10th
        row of the expression interact with each other.
    
    Returns
    -------
    ndarray of int
    """
    graph_data = np.array(np.load(path, allow_pickle=True), dtype=np.int)

    return graph_data
        

def create_Torch_Data(exp, net):
    """
    Create a Torch Data object compatible with pytorch_geometric functions
    
    Parameters
    ----------
    exp : ndarray of float 
        the expression data (2D)
    ppi : nd array of int
        the network data (2D)
    
    Returns
    -------
    torch.Data
        torch Data object containing the expression data and network data
    """

    # Convert the PPI and the expression data to tensors
    net_edges = torch.tensor(net, dtype=torch.long)
    exp_data = torch.tensor(exp, dtype=torch.float)

    # Create a torch geometric data object
    data = Data(x=exp_data, edge_index=net_edges)

    return data  
    
def load_model(opts):
    if not opts.embedding:
        model = {'GraphConv': FAE_GraphConv,
              'GCN': FAE_GCN,
              'SAGEConv': FAE_SAGEConv ,
              'FeatGraphConv': FAE_FeatGraphConv,
              'MLP': AE_MLP,
              'Magic': MAGIC,
                 'LR': LinearRegression,
                 'RF': RandomForestRegressor
        }.get(opts.model, None)
    else:
        model = Embedding_ExpGAE

    assert model is not None, "Currently unsupported model: {}!".format(opts.model)
    return model  


if __name__ == "__main__":

    # Pull the arguments
    opts = get_options()
    
    # Import the expression and network data
    exp_data = import_expressions(opts.exp_path)
    net_data = import_graph(opts.network_path)

    # Create a Torch Data object
    data = create_Torch_Data(exp_data, net_data)  

    # Determine model
    model_class = load_model(opts)

    if opts.problem == 'Prediction':
        if not opts.embedding:
            supervised_prediction_eval(model_class, data, opts)
        else:
            embedding_prediction_eval(model_class, data, opts)
    
    elif opts.problem == 'Imputation_eval':
        imputation_eval(model_class, data, opts)
    elif opts.problem == 'Imputation':
        imputed = impute(model_class, data, opts)
        # need to save
        np.save(opts.model + opts.network + '_imputed.npy', imputed.cpu().detach().numpy())