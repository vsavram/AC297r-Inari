import torch
from torch_geometric.data import Data


def create_TorchData(exp, ppi):

    # Convert the PPI and the expression data to tensors
    ppi_edges = torch.tensor(ppi, dtype=torch.long)
    exp_data = torch.tensor(exp, dtype=torch.float)

    # Create a torch geometric data object
    data = Data(x=exp_data, edge_index=ppi_edges)

    return data


#-------------------------



from torch_geometric.data import InMemoryDataset,Data
from torch_geometric.utils import dense_to_sparse, to_undirected, remove_self_loops
import torch
import numpy as np
from sklearn.model_selection import train_test_split



class RNA_Data(InMemoryDataset):
    # expression_file = 'maize_expressions.npy'
    # expression_file = 'mouse_rnaSeq.npy'


    def __init__(self, root, network='MousePPI', transform=None, pre_transform=None):
        self.network = network
        super(InMemoryDataset, self).__init__(root, transform, pre_transform)

        if self.network == 'MousePPI':
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['MousePPI_processed_rnaSeq_data.pt']

    def download(self):
        pass

    def index_to_mask(self, indices, index, shape):
        mask = torch.zeros(shape)
        mask[indices[0, index], indices[1, index]] = 1
        return mask

    def process(self):
        edge_index = torch.tensor(np.array(
            np.load(self.root + '/' + self.network + '.npy', allow_pickle=True), dtype=np.int))
        gene_names = None
        if self.network == 'MousePPI':
            x = torch.tensor(np.load(self.root + '/' + 'mouse_rnaSeq.npy', allow_pickle=True), dtype=torch.float)
        else:
            x = np.load(self.root + '/' + 'rnaSeq.npy', allow_pickle=True)
            gene_names = x[:, 0]
            x = torch.tensor(np.array(x[:, 1:], dtype=np.float), dtype=torch.float)
        print(x.size(0))
        matrix_mask = torch.zeros([x.size(0), x.size(1)])
        matrix_mask[x.nonzero(as_tuple=True)] = 1
        indices = np.array(x.data.numpy().nonzero())
        ix_train, ix_test = train_test_split(np.arange(len(indices[0])), test_size=0.25, random_state=42)

        data = Data(x=x, edge_index=edge_index, y=x, nonzeromask=matrix_mask)
        data.gene_names = gene_names if gene_names is not None else None
        if self.network == 'MousePPI':
            torch.save(self.collate([data]), self.processed_paths[0])