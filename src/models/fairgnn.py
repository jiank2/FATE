import torch
import torch.nn as nn
import torch.nn.functional as F


from layers.graph_convolution import GraphConvolution


class BackboneGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(BackboneGCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, adj, x):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class SensitiveEstimator(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(SensitiveEstimator, self).__init__()
        self.gcn = BackboneGCN(nfeat, nhid, nhid, dropout)
        self.fc = nn.Linear(nhid, nclass)

    def forward(self, adj, x):
        x = self.gcn(adj, x)
        x = self.fc(x)
        return x


class FairGNN(nn.Module):
    """FairGNN, only works for binary classification and binary sensitive attribute."""

    def __init__(self, nfeat, nhid, dropout, lr, weight_decay):
        super(FairGNN, self).__init__()
        # modules
        self.backbone = BackboneGCN(nfeat, nhid, nhid, dropout)
        self.estimator = SensitiveEstimator(nfeat, 128, 1, dropout)
        self.classifier = nn.Linear(nhid, 1)
        self.adversary = nn.Linear(nhid, 1)

        self.lr = lr
        self.weight_decay = weight_decay

        # loss
        self.criterion = nn.BCEWithLogitsLoss()
        self.loss_classifiers = 0
        self.loss_adversary = 0
    
    def init_optimizers(self):
        # parameters
        params_classifiers = (
            list(self.backbone.parameters())
            + list(self.classifier.parameters())
            + list(self.estimator.parameters())
        )

        # optimizers
        self.optimizer_classifers = torch.optim.Adam(
            params_classifiers, lr=self.lr, weight_decay=self.weight_decay
        )
        self.optimizer_adversary = torch.optim.Adam(
            self.adversary.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def forward(self, adj, x):
        sensitive_output = self.estimator(adj, x)
        label_output = self.backbone(adj, x)
        label_output = self.classifier(label_output)
        return label_output#, sensitive_output

    def optimize(
        self,
        adj,
        x,
        labels,
        sensitive_labels,
        idx_train,
        alpha,
        beta,
        retain_graph=False,
        enable_update=True,
        num_train_sensitive_labels=200,
    ):
        self.train()

        self.adversary.requires_grad_(False)
        self.optimizer_classifers.zero_grad()

        # predict class label
        node_embedding = self.backbone(adj, x)
        label_output = self.classifier(node_embedding)
        label_logit = torch.sigmoid(label_output)
        adversary_output = self.adversary(node_embedding)
        self.cls_loss = self.criterion(
            label_output[idx_train], labels[idx_train].unsqueeze(1).float()
        )

        # predict sensitive attribute
        sensitive_logit = self.estimator(adj, x)
        sensitive_logit = torch.sigmoid(sensitive_logit.detach())
        # s_score = (s_score > 0.5).float()
        if num_train_sensitive_labels is not None:
            sensitive_logit[idx_train[:num_train_sensitive_labels]] = sensitive_labels[idx_train[:num_train_sensitive_labels]].unsqueeze(1).float()
        else:
            sensitive_logit[idx_train] = sensitive_labels[idx_train].unsqueeze(1).float()
        self.adv_loss = self.criterion(adversary_output, sensitive_logit)

        # calculate covariance
        self.covariance = torch.abs(
            torch.mean(
                (sensitive_logit - torch.mean(sensitive_logit))
                * (label_logit - torch.mean(label_logit))
            )
        )

        # calculate total loss
        self.loss_classifiers = (
            self.cls_loss + alpha * self.covariance - beta * self.adv_loss
        )
        if enable_update:
            self.loss_classifiers.backward(retain_graph=retain_graph)
            self.optimizer_classifers.step()

        # update adversary
        self.adversary.requires_grad_(True)
        self.optimizer_adversary.zero_grad()
        adversary_output = self.adversary(node_embedding.detach())
        self.loss_adversary = self.criterion(adversary_output, sensitive_logit)
        if enable_update:
            self.loss_adversary.backward(retain_graph=retain_graph)
            self.optimizer_adversary.step()
        return self.loss_classifiers, self.loss_adversary
