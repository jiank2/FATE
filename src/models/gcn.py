import torch.nn as nn
import torch.nn.functional as F

from layers.graph_convolution import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.linear = nn.Linear(nhid, 1)
        self.dropout = dropout

    def forward(self, adj, x, with_nonlinearity=True):
        x = self.gc1(x, adj)
        if with_nonlinearity:
            x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = self.linear(x)
        return x
