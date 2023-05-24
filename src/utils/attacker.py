import copy
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.csgraph import laplacian

from models.gcn import GCN
from utils.evaluator import Evaluator
from utils.fairness_criteria import GroupFairnessKDE, INFORMForAttacker
from utils.helper_functions import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Attacker:
    def __init__(
        self,
        attack_configs,
        data,
        no_cuda,
        device,
        random_seed=None,
    ):
        # get cuda-related info
        self.no_cuda = no_cuda
        self.device = device

        # get configs
        self.attack_configs = attack_configs

        self.with_nonlinearity = (
            True if self.attack_configs["model"] in ("gcn",) else False
        )

        # get dataset stats
        self.num_nodes = data.num_nodes
        self.num_edges = data.num_edges
        self.num_node_features = data.num_node_features
        self.num_classes = data.num_classes

        # get dataset
        self.original_graph = data.graph
        self.features = torch.nan_to_num(
            normalize_feature_min_max(data.features)
        )  # replace nan values to 0
        self.labels = data.labels
        self.sensitive_labels = data.sensitive_labels
        self.train_idx = data.train_idx
        self.val_idx = data.val_idx
        self.test_idx = data.test_idx

        # move to corresponding device
        if not self.no_cuda:
            self.features = self.features.to(self.device)
            self.labels = self.labels.to(self.device)
            self.sensitive_labels = self.sensitive_labels.to(self.device)
            self.train_idx = self.train_idx.to(self.device)
            self.val_idx = self.val_idx.to(self.device)
            self.test_idx = self.test_idx.to(self.device)

        # init loss functions
        self.utility_criterion = nn.BCEWithLogitsLoss()
        if self.attack_configs["fairness_definition"] in ("statistical_parity"):
            self.fairness_criterion = GroupFairnessKDE(
                delta=self.attack_configs["delta"],
                device=self.device,
            )
        elif self.attack_configs["fairness_definition"] in ("individual_fairness"):
            self.fairness_criterion = INFORMForAttacker(
                similarity_laplacian=laplacian(
                    filter_similarity_matrix(
                        get_similarity_matrix(
                            mat=self.original_graph.tocoo(),
                            similarity_measure=self.attack_configs[
                                "inform_similarity_measure"
                            ],
                        ),
                        sigma=0.75,
                    )
                ),
                no_cuda=self.no_cuda,
                device=self.device,
            )

        # init evaluator
        self.evaluator = Evaluator()

        # set random seed
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            if not self.no_cuda:
                torch.cuda.manual_seed(random_seed)

    def attack(self):
        _, path = self._file_exists()

        # init
        perturbed_adj = copy.deepcopy(self.original_graph).todense()

        ones_graph = torch.ones(self.num_nodes, self.num_nodes)
        if not self.no_cuda:
            ones_graph = ones_graph.to(self.device)

        for epoch in range(self.attack_configs["attack_steps"]):
            # perturbed graph to tensor
            perturbed_graph = torch.FloatTensor(perturbed_adj).to(self.device)

            # get idx for unlabeled nodes
            unlabeled_idx = torch.LongTensor(
                self.train_idx.tolist() + self.val_idx.tolist() + self.test_idx.tolist()
            )

            # prepare graph
            original_graph_normalized = sparse_matrix_to_sparse_tensor(
                symmetric_normalize(
                    self.original_graph + sp.eye(self.original_graph.shape[0])
                )
            ).to_dense()

            # put into cuda
            if not self.no_cuda:
                unlabeled_idx = unlabeled_idx.to(self.device)
                original_graph_normalized = original_graph_normalized.to(self.device)

            # get hypergradient
            graph_delta = self._hypergradient_computation(
                perturbed_graph=perturbed_graph,
                train_idx=self.train_idx,
                val_idx=unlabeled_idx,
            )

            if self.attack_configs["perturbation_mode"] == "flip":
                pass
            elif self.attack_configs["perturbation_mode"] == "delete":
                graph_delta[graph_delta < 0] = 0  # only keep the positive grad terms
            elif self.attack_configs["perturbation_mode"] == "add":
                graph_delta[graph_delta > 0] = 0  # only keep the negative grad terms

            topology_budget = (
                1
                if epoch == 0
                else int(
                    (int(self.attack_configs["perturbation_rate"] * self.num_edges) - 1)
                    // 2
                    * 2
                    // (self.attack_configs["attack_steps"] - 1)
                )
            )

            s_adj = -graph_delta * (ones_graph - 2 * perturbed_graph.data)
            _, idx = torch.topk(s_adj.flatten(), topology_budget)
            idx_row, idx_column = np.unravel_index(
                idx.cpu().numpy(), perturbed_adj.shape
            )
            for i in range(len(idx_row)):
                perturbed_adj[idx_row[i], idx_column[i]] = (
                    1 - perturbed_adj[idx_row[i], idx_column[i]]
                )

        # save attacked graph
        self._save_perturbed_dataset(perturbed_adj, path)

    def _hypergradient_computation(
        self,
        perturbed_graph,
        train_idx,
        val_idx,
    ):
        # initialize params
        graph_diff = torch.zeros_like(perturbed_graph)

        perturbed_graph_with_grad = perturbed_graph.detach()
        perturbed_graph_with_grad.requires_grad_(True)
        perturbed_graph_with_grad_normalized = symmetric_normalize_tensor(
            perturbed_graph_with_grad
            + torch.eye(perturbed_graph_with_grad.shape[0]).to(self.device)
        )

        perturbed_graph_normalized = symmetric_normalize_tensor(
            perturbed_graph + torch.eye(perturbed_graph.shape[0]).to(self.device)
        )

        loss = torch.tensor(0.0).to(self.device)

        ### pre-train model ###
        backbone = GCN(
            nfeat=self.num_node_features,
            nhid=self.attack_configs["hidden_dimension"],
            nclass=1,
            dropout=self.attack_configs["pre_train_dropout"],
        )
        if not self.no_cuda:
            backbone.to(self.device)
        opt = torch.optim.Adam(
            backbone.parameters(),
            lr=self.attack_configs["pre_train_lr"],
            weight_decay=self.attack_configs["pre_train_weight_decay"],
        )
        backbone = self._gcn_pre_train(
            model=backbone,
            opt=opt,
            graph=perturbed_graph_normalized,
            num_epochs=self.attack_configs["pre_train_num_epochs"],
            train_idx=train_idx,
        )

        ### calculate loss ###
        # for high-order hypergradients, change it to some number larger than 1
        for _ in range(1):
            backbone.train()
            opt.zero_grad()
            output = backbone(
                perturbed_graph_normalized,
                self.features,
                with_nonlinearity=self.with_nonlinearity,
            )
            loss_train = self.utility_criterion(
                output[train_idx],
                self.labels[train_idx].unsqueeze(1).float(),
            )
            loss_train.backward(retain_graph=False)
            opt.step()

            # obtain bias grad on validation set
            output = backbone(
                perturbed_graph_with_grad_normalized,
                self.features,
                with_nonlinearity=self.with_nonlinearity,
            )
            if self.attack_configs["fairness_definition"] == "individual_fairness":
                loss = -self.fairness_criterion(output)
            elif self.attack_configs["fairness_definition"] == "statistical_parity":
                output = torch.sigmoid(output)
                loss = -self.fairness_criterion(
                    output=output[val_idx],
                    labels=self.labels[val_idx],
                    sensitive_labels=self.sensitive_labels[val_idx],
                    bandwidth=self.attack_configs["bandwidth"],
                    tau=self.attack_configs["tau"],
                    is_statistical_parity=True,
                )
        grad_graph = torch.autograd.grad(
            loss, perturbed_graph_with_grad, create_graph=True
        )
        grad_graph = grad_graph[0].data
        graph_diff = (
            grad_graph
            + grad_graph.permute(1, 0)
            - torch.diag(torch.diagonal(grad_graph, 0))
        )

        return graph_diff

    def _gcn_pre_train(
        self,
        model,
        opt,
        graph,
        num_epochs,
        train_idx,
    ):
        """Pre-train GCN for attacking."""
        for _ in range(num_epochs):
            model.train()
            opt.zero_grad()

            # train
            output = model(
                graph, self.features, with_nonlinearity=self.with_nonlinearity
            )
            loss_train = self.utility_criterion(
                output[train_idx], self.labels[train_idx].unsqueeze(1).float()
            )
            loss_train.backward()
            opt.step()

        return model

    def _save_perturbed_dataset(self, perturbed_adj, path):
        adj = sp.coo_matrix(perturbed_adj)
        torch.save(
            {
                "num_nodes": self.num_nodes,
                "num_edges": int((adj.nnz + adj.diagonal().sum()) // 2),
                "num_node_features": self.num_node_features,
                "num_classes": self.num_classes,
                "adjacency_matrix": adj,
                "node_features": self.features.cpu().numpy(),
                "labels": self.labels.cpu().numpy(),
                "sensitive_labels": self.sensitive_labels.cpu().numpy(),
                "train_idx": self.train_idx.cpu().numpy(),
                "val_idx": self.val_idx.cpu().numpy(),
                "test_idx": self.test_idx.cpu().numpy(),
            },
            path,
        )

        del adj

    def _file_exists(self):
        attack_setting = "rate={rate}_mode={mode}_steps={steps}_lr={lr}_nepochs={nepochs}_seed={seed}".format(
            rate=self.attack_configs["perturbation_rate"],
            mode=self.attack_configs["perturbation_mode"],
            steps=self.attack_configs["attack_steps"],
            lr=self.attack_configs["pre_train_lr"],
            nepochs=self.attack_configs["pre_train_num_epochs"],
            seed=self.random_seed,
        )
        folder_path = os.path.join(
            "..",
            "data",
            "fate",
            self.attack_configs["dataset"],
            self.attack_configs["fairness_definition"],
        )

        try:
            os.makedirs(folder_path)
        except:
            pass

        file_path = os.path.join(folder_path, f"{attack_setting}.pt")

        return os.path.exists(file_path), file_path
