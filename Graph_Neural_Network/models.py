from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch.nn import Linear
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

## {{TODO: Comments, revisions}}

class FeatGraphConv(MessagePassing):
    def __init__(self, in_channels,hidden , out_channels, aggr='mean', bias = True,
                 **kwargs):
        super(FeatGraphConv, self).__init__(aggr=aggr, **kwargs)
        self.lin1 = Linear(2*hidden, out_channels, bias=bias)
        self.lin2 = Linear(in_channels, hidden, bias=bias)
    def forward(self, x, edge_index, edge_weight=None, size=None):
        edge_index,edge_weight = add_remaining_self_loops(edge_index=edge_index,edge_weight = edge_weight)
        h = self.lin2(x)
        return self.propagate(edge_index, size=size, x=x, h=h,
                              edge_weight=edge_weight)

    def message(self, h_j, edge_weight):
        return h_j if edge_weight is None else edge_weight.view(-1, 1) * h_j

    def update(self, aggr_out, h):
        return self.lin1(torch.cat((h, aggr_out), 1))

class FAE_FeatGraphConv(nn.Module):
    def __init__(self, in_channels, opts):
        super(FAE_FeatGraphConv, self).__init__()
        self.opts = opts
        if self.opts.problem == 'Prediction':
            self.conv1 = FeatGraphConv(in_channels, 64, 64, aggr='mean')
            self.conv2 = FeatGraphConv(64, 32, 32, aggr='mean')
            self.lin = Linear(32, 1)
        else:
            self.conv1 = FeatGraphConv(in_channels, 16, 32, aggr='mean')
            self.lin = Linear(32, in_channels)
    def forward(self, data):
        if self.opts.problem == 'Prediction':
            x, edge_index = data.x, data.edge_index
            x = torch.relu(self.conv1(x, edge_index))
            x = torch.relu(self.conv2(x, edge_index))
            return self.lin(x)
        else:
            x, edge_index = data.x, data.edge_index
            x = torch.relu(self.conv1(x, edge_index))
            x = self.lin(x)
            return x 

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 64, cached=True)
        self.conv2 = GCNConv(64, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


class Embedding_ExpGAE(GAE):
    def __init__(self, in_channels, out_channels):
        encoder = Encoder(in_channels, out_channels)
        super(Embedding_ExpGAE, self).__init__(encoder=encoder)
        self.predictor_lr = LinearRegression()
        self.predictor_rf = RandomForestRegressor(n_estimators=20, max_depth=2)
    def fit_predictor(self, z, y):
        self.predictor_lr.fit(z, y)
        self.predictor_rf.fit(z, y)

    def predict(self, x, edge_index):
        z = self.encode(x, edge_index)
        return self.predictor_lr.predict(z.cpu().data.numpy()), self.predictor_rf.predict(z.cpu().data.numpy())
