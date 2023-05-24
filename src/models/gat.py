import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.graph_attention import GraphAttentionLayer, SpGraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [
            GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
            for _ in range(nheads)
        ]
        for i, attention in enumerate(self.attentions):
            self.add_module("attention_{}".format(i), attention)

        self.out_att = GraphAttentionLayer(
            nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False
        )

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, dropout_attn=0.0):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [
            # SpGraphAttentionLayer(
            #     nfeat, nhid, dropout=dropout, alpha=alpha, concat=True
            # )
            SpGraphAttentionLayer(
                nfeat, nhid, dropout=dropout_attn, alpha=alpha, concat=True
            )
            for _ in range(nheads)
        ]
        for i, attention in enumerate(self.attentions):
            self.add_module("attention_{}".format(i), attention)

        # self.out_att = SpGraphAttentionLayer(
        #     nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False
        # )
        self.out_att = SpGraphAttentionLayer(
            nhid * nheads, nclass, dropout=dropout_attn, alpha=alpha, concat=False
        )

    def forward(self, adj, x, with_nonlinearity=True):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x
