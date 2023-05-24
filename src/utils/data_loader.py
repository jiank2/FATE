import os

import scipy.sparse as sp
import torch

from utils.helper_functions import *


class GraphDataset:
    def __init__(self, configs):
        prefix_sensitive_attr = "no" if configs["no_sensitive_attribute"] else "with"

        # get file path
        file_path = "../data/clean/{name}/{name}_{prefix_sensitive_attr}_sensitive_attr.pt".format(
            name=configs["name"], prefix_sensitive_attr=prefix_sensitive_attr
        )

        # check if dataset exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError("Dataset does not exist!")

        # load data
        data = torch.load(file_path)

        # read stats
        self.num_nodes = data["num_nodes"]
        self.num_edges = data["num_edges"]
        self.num_node_features = data["num_node_features"]
        self.num_classes = data["num_classes"]

        # load graph and feature as dense tensor to enable autograd
        self.graph = data["adjacency_matrix"]
        try:
            self.features = torch.FloatTensor(data["node_features"].todense())
        except:
            self.features = torch.FloatTensor(data["node_features"].astype("int64"))
        self.labels = torch.LongTensor(data["labels"])
        self.sensitive_labels = torch.LongTensor(data["sensitive_labels"])

        self.is_ratio = configs["is_ratio"]
        self.split_by_class = configs["split_by_class"]
        self.num_train = configs["num_train"]
        self.num_val = configs["num_val"]
        self.num_test = self.num_nodes - self.num_train - self.num_val
        self.ratio_train = configs["ratio_train"]
        self.ratio_val = configs["ratio_val"]

        del data

    def set_random_split(self, splits):
        self.train_idx = torch.LongTensor(splits["train_idx"])
        self.val_idx = torch.LongTensor(splits["val_idx"])
        self.test_idx = torch.LongTensor(splits["test_idx"])

    def get_split_by_class(self, num_train_per_class):
        res = None
        for c in range(self.num_classes):
            idx = (self.labels == c).nonzero(as_tuple=False).view(-1)
            idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
            res = torch.cat((res, idx)) if res is not None else idx
        return res

    def preprocess_matrix(self, type="laplacian"):
        if type == "laplacian":
            self.graph = sparse_matrix_to_sparse_tensor(
                symmetric_normalize(self.raw_graph + sp.eye(self.raw_graph.shape[0]))
            )
        elif type == "row":
            self.graph = sparse_matrix_to_sparse_tensor(
                row_normalize(self.raw_graph + sp.eye(self.raw_graph.shape[0]))
            )
        else:
            raise ValueError("type should be laplacian or row")

    def preprocess(self, type="laplacian"):
        if type == "laplacian":
            self.graph = symmetric_normalize_tensor(
                self.raw_graph + torch.eye(self.raw_graph.shape[0])
            )
        elif type == "row":
            self.graph = row_normalize_tensor(
                self.raw_graph + torch.eye(self.raw_graph.shape[0])
            )
        else:
            raise ValueError("type should be laplacian or row")
