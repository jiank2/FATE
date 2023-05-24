import logging
import os

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.io import loadmat
from scipy.sparse import find

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_regular_dataset(
    dataset,
    root_path,
    sensitive_attr,
    label,
    unused_features,
    with_sensitive_attr=True,
):
    logger.info(f"Start preprocessing {dataset} dataset")

    # load features
    idx_features_labels = pd.read_csv(os.path.join(root_path, f"{dataset}.csv"))
    header = list(idx_features_labels.columns)

    # if necessary, remove sensitive attribute column in input features
    folder_prefix = "with"
    if not with_sensitive_attr:
        header.remove(sensitive_attr)
        folder_prefix = "no"

    # remove label column in input features
    header.remove(label)

    # remove other unnecessary columns in input features
    for unused_feat in unused_features:
        header.remove(unused_feat)

    # create labels
    labels = idx_features_labels[label].values.astype(int)
    sensitive_labels = idx_features_labels[sensitive_attr].values.astype(int)

    # create feature matrix
    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)

    # load edge list
    edgelist_path = os.path.join(root_path, f"{dataset}_edges.txt")
    edges_unordered = np.genfromtxt(edgelist_path).astype(int)

    # create indices
    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}

    # create graph
    edges = np.array(
        list(map(idx_map.get, edges_unordered.flatten())), dtype=int
    ).reshape(edges_unordered.shape)
    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(labels.shape[0], labels.shape[0]),
        dtype=np.float32,
    )

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # build dataset and save
    result = {
        "num_nodes": features.shape[0],
        "num_edges": int((adj.nnz + adj.diagonal().sum()) // 2),
        "num_node_features": features.shape[1],
        "num_classes": len(set(labels)),
        "adjacency_matrix": adj,
        "node_features": features,
        "labels": labels,
        "sensitive_labels": sensitive_labels,
    }
    print(dataset)
    print(result)

    save_path = os.path.join(
        "..",
        "data",
        "clean",
        f"{dataset}",
    )
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    torch.save(
        result, os.path.join(save_path, f"{dataset}_{folder_prefix}_sensitive_attr.pt")
    )


def build_pokec_dataset(
    dataset,
    root_path,
    sensitive_attr,
    label,
    unused_features,
    with_sensitive_attr=True,
):
    logger.info(f"Start preprocessing {dataset} dataset")

    # load features
    idx_features_labels = pd.read_csv(os.path.join(root_path, f"{dataset}.csv"))
    header = list(idx_features_labels.columns)

    # remove id column
    header.remove("user_id")

    # if necessary, remove sensitive attribute column in input features
    folder_prefix = "with"
    if not with_sensitive_attr:
        header.remove(sensitive_attr)
        folder_prefix = "no"

    # remove label column in input features
    header.remove(label)

    # remove other unnecessary columns in input features
    for unused_feat in unused_features:
        header.remove(unused_feat)

    # create labels
    labels = idx_features_labels[label].values.astype(int)

    # create sensitive labels
    sensitive_labels = idx_features_labels[sensitive_attr].values.astype(int)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)

    # get idx whose label and sensitive attributes are (un)available
    known_sensitive_labels_idx = set(np.where(sensitive_labels >= 0)[0])
    known_label_idx = np.where(labels >= 0)[0]
    idx_used = np.asarray(list(known_sensitive_labels_idx & set(known_label_idx)))
    idx_nonused = np.asarray(
        list(set(np.arange(len(labels))).difference(set(idx_used)))
    )

    # filter features, labels and sensitive labels
    features = features[idx_used, :]
    labels = labels[idx_used]
    sensitive_labels = sensitive_labels[idx_used]

    # read id column
    user_id = np.array(idx_features_labels["user_id"], dtype=int)

    # load edgelist
    edges_unordered = np.genfromtxt(
        os.path.join(root_path, f"{dataset}_relationship.txt"), dtype=int
    )

    # filter edges that contain users with unknown label or sensitive label
    user_id_nonused = user_id[idx_nonused]
    user_id_used = user_id[idx_used]
    used_ind1 = [
        i for i, elem in enumerate(edges_unordered[:, 0]) if elem not in user_id_nonused
    ]
    used_ind2 = [
        i for i, elem in enumerate(edges_unordered[:, 1]) if elem not in user_id_nonused
    ]
    intersect_ind = list(set(used_ind1) & set(used_ind2))
    edges_unordered = edges_unordered[intersect_ind, :]

    idx_map = {j: i for i, j in enumerate(user_id_used)}
    edges_un = np.array(
        list(map(idx_map.get, edges_unordered.flatten())), dtype=int
    ).reshape(edges_unordered.shape)

    adj = sp.coo_matrix(
        (np.ones(edges_un.shape[0]), (edges_un[:, 0], edges_un[:, 1])),
        shape=(labels.shape[0], labels.shape[0]),
        dtype=np.float32,
    )

    # get largest connected component
    graph = nx.from_scipy_sparse_matrix(adj)
    graph_ccs = (graph.subgraph(cc).copy() for cc in nx.connected_components(graph))
    graph_max_cc = max(graph_ccs, key=len)

    # re-map node id
    node_ids = list(graph_max_cc.nodes())
    idx_s = node_ids

    # filter features with nodes in largest connected component
    features = features[idx_s, :]
    features = features[
        :, np.where(np.std(np.array(features.todense()), axis=0) != 0)[0]
    ]
    features = np.array(features.todense())

    # filter labels and sensitive labels with nodes in largest connected component
    labels = np.array(labels[idx_s], dtype=int)
    sensitive_labels = np.array(sensitive_labels[idx_s], dtype=int)

    # binarize label and sensitive label
    labels[labels > 1] = 1
    sensitive_labels[sensitive_labels > 0] = 1
    idx_map_n = {j: int(i) for i, j in enumerate(idx_s)}

    # filter edges with both endpoints in largest connected component
    idx_nonused2 = np.asarray(
        list(set(np.arange(len(list(graph.nodes())))).difference(set(idx_s)))
    )
    used_ind1 = [i for i, elem in enumerate(edges_un[:, 0]) if elem not in idx_nonused2]
    used_ind2 = [i for i, elem in enumerate(edges_un[:, 1]) if elem not in idx_nonused2]
    intersect_ind = list(set(used_ind1) & set(used_ind2))
    edges_un = edges_un[intersect_ind, :]
    edges = np.array(list(map(idx_map_n.get, edges_un.flatten())), dtype=int).reshape(
        edges_un.shape
    )
    edges = np.unique(edges, axis=0)
    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(labels.shape[0], labels.shape[0]),
        dtype=np.float32,
    )

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # build dataset and save
    result = {
        "num_nodes": features.shape[0],
        "num_edges": int((adj.nnz + adj.diagonal().sum()) // 2),
        "num_node_features": features.shape[1],
        "num_classes": len(set(labels)),
        "adjacency_matrix": adj,
        "node_features": features,
        "labels": labels,
        "sensitive_labels": sensitive_labels,
    }
    print(dataset)
    print(result)

    save_path = os.path.join(
        "..",
        "data",
        "clean",
        f"{dataset}",
    )
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    torch.save(
        result, os.path.join(save_path, f"{dataset}_{folder_prefix}_sensitive_attr.pt")
    )


if __name__ == "__main__":
    with_sensitive_attr = False
    datasets = {
        "pokec_z": {
            "sensitive_attr": "region",
            "label": "I_am_working_in_field",
            "unused_features": [],
        },
        "pokec_n": {
            "sensitive_attr": "region",
            "label": "I_am_working_in_field",
            "unused_features": [],
        },
        "bail": {
            "sensitive_attr": "WHITE",
            "label": "RECID",
            "unused_features": [],
        },
    }
    for dataset in datasets:
        root_path = os.path.join("data", "raw", f"{dataset}")
        if dataset == "bail":
            build_regular_dataset(
                dataset=dataset,
                root_path=root_path,
                label=datasets[dataset]["label"],
                sensitive_attr=datasets[dataset]["sensitive_attr"],
                unused_features=datasets[dataset]["unused_features"],
                with_sensitive_attr=with_sensitive_attr,
            )
        else:
            build_pokec_dataset(
                dataset=dataset,
                root_path=root_path,
                label=datasets[dataset]["label"],
                sensitive_attr=datasets[dataset]["sensitive_attr"],
                unused_features=datasets[dataset]["unused_features"],
                with_sensitive_attr=with_sensitive_attr,
            )
