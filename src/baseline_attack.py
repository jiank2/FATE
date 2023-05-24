import argparse
import copy
import os

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components

from fagnn.attack.attack import attack
from utils.data_loader import GraphDataset
from utils.helper_functions import *


def largest_connected_components(adj, n_components=1):
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][
        :n_components
    ]  # reverse order to sort descending
    nodes_to_keep = [
        idx
        for (idx, component) in enumerate(component_indices)
        if component in components_to_keep
    ]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep


def generate_splits(dataset, num_splits, split_seed=1684992425, no_cuda=False):
    # set random seed
    if split_seed is not None:
        np.random.seed(split_seed)
        torch.manual_seed(split_seed)
        if not no_cuda:
            torch.cuda.manual_seed(split_seed)

    # generate splits
    res = list()
    for _ in range(num_splits):
        res.append(random_split(dataset))
    return res


parser = argparse.ArgumentParser()
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="Disables CUDA training."
)
"""
Dataset args
"""
parser.add_argument(
    "--dataset",
    type=str,
    default="bail",
    choices=[
        "pokec_z",
        "pokec_n",
        "bail",
    ],
)
parser.add_argument(
    "--train_percent_atk",
    type=float,
    default=0.5,
    help="Percentage of labeled data available to the attacker.",
)
parser.add_argument(
    "--train_percent_gnn",
    type=float,
    default=0.5,
    help="Percentage of labeled data as train set.",
)
parser.add_argument(
    "--val_percent",
    type=float,
    default=0.25,
    help="Percentage of labeled data as validation set.",
)
"""
Model args
"""
parser.add_argument(
    "--model",
    type=str,
    default=["gcn"],
    nargs="+",
    choices=[
        "gcn",
        "gat",
        "gsage",
        "fairgnn",
    ],
)
parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate.")
parser.add_argument(
    "--weight_decay",
    type=float,
    default=5e-4,
    help="Weight decay (L2 loss on parameters).",
)
parser.add_argument("--hidden", type=int, default=64, help="Number of hidden units.")
parser.add_argument(
    "--dropout", type=float, default=0.6, help="Dropout rate (1 - keep probability)."
)
parser.add_argument(
    "--attack_type",
    type=str,
    default="fagnn",
    choices=[
        "none",
        "random",
        "dice",
        "fagnn",
    ],
    help="Adversarial attack type.",
)
parser.add_argument(
    "--sensitive",
    type=str,
    default="region",
    choices=["gender", "region"],
    help="Sensitive attribute of Pokec.",
)
parser.add_argument(
    "--preprocess_pokec",
    type=bool,
    default=False,
    help="Include only completed accounts in Pokec datasets (only valid when dataset==pokec_n/pokec_z])",
)
parser.add_argument(
    "--ptb_rate",
    type=float,
    nargs="+",
    default=[0.05],
    help="Attack perturbation rate [0-1]",
)
parser.add_argument("--num_layers", type=int, default=2, help="number of hidden layers")
# ----args for FairAttack
parser.add_argument(
    "--direction",
    type=str,
    default="y1s1",
    choices=["y1s1", "y1s0", "y0s0", "y0s1"],
    help="FairAttack direction",
)
parser.add_argument(
    "--strategy",
    type=str,
    default="DD",
    choices=["DD", "DE", "ED", "EE"],
    help="FairAttack strategy indicating [D]ifferent/[E]qual label(y)|sens(s)",
)
parser.add_argument(
    "--deg",
    type=int,
    default=0,  # may not finish on small datasets
    choices=[0, 1, 2, 3],
    help="Degree parameter, 0 for not considering degree, " "d(high)>deg*d(low).",
)
parser.add_argument(
    "--deg_direction",
    type=str,
    default="null",
    choices=["hl", "lh", "null"],
    help="Direction of degree difference, "
    "hl for (subject-influencer)=(high-low), and vice versa,"
    "null for not considering degree.",
)

# ----args for GSAGE
parser.add_argument(
    "--agg_type",
    type=str,
    default="mean",
    choices=["gcn", "mean", "pool", "lstm"],
    help="Aggregator for GraphSAGE",
)

# ----args for GAT
parser.add_argument(
    "--num_heads", type=int, default=8, help="number of hidden attention heads"
)
parser.add_argument(
    "--num_out_heads", type=int, default=1, help="number of output attention heads"
)
parser.add_argument("--in_drop", type=float, default=0.6, help="input feature dropout")
parser.add_argument("--attn_drop", type=float, default=0.6, help="attention dropout")
parser.add_argument(
    "--negative_slope", type=float, default=0.2, help="the negative slope of leaky relu"
)
parser.add_argument(
    "--residual", action="store_true", default=False, help="use residual connection"
)

# ----args for fairgnn
parser.add_argument(
    "--base_model",
    type=str,
    default="GCN",
    choices=["GCN", "GAT"],
    help="Base GNN model for FairGNN",
)
parser.add_argument(
    "--alpha", type=float, default=2, help="The hyperparameter of alpha"
)
parser.add_argument(
    "--beta", type=float, default=0.1, help="The hyperparameter of beta"
)
parser.add_argument(
    "--sens_number", type=int, default=200, help="the number of sensitive attributes"
)

"""
Optimization args
"""
parser.add_argument(
    "--epochs", type=int, default=500, help="Number of epochs to train."
)
parser.add_argument(
    "--fastmode",
    action="store_true",
    default=False,
    help="Validate during training pass.",
)
parser.add_argument(
    "--acc",
    type=float,
    default=0.2,
    help="the selected FairGNN accuracy on val would be at least this high",
)
parser.add_argument(
    "--roc",
    type=float,
    default=0.5,
    help="the selected FairGNN ROC score on val would be at least this high",
)

args = parser.parse_known_args()[0]
args.cuda = True  # not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
seed_set = [25]

for model_name in args.model:
    for ptb_rate in args.ptb_rate:
        FINAL_RESULT = []
        N = len(seed_set)
        for repeat in range(N):
            # Load data
            print(args.dataset)
            dataset_configs = {
                "name": args.dataset,
                "no_sensitive_attribute": True,
                "is_ratio": True,
                "split_by_class": False,
                "num_train": 20,
                "num_val": 200,
                "ratio_train": 0.5,
                "ratio_val": 0.25,
            }

            dataset = GraphDataset(dataset_configs)
            splits = generate_splits(dataset=dataset, num_splits=1)
            split = splits[0]
            dataset.set_random_split(split)

            dataset_name = args.dataset

            adj = dataset.graph
            features = dataset.features
            labels = dataset.labels

            idx_train_atk = dataset.train_idx
            idx_train_gnn = copy.deepcopy(dataset.train_idx)
            idx_val = dataset.val_idx.cpu().numpy()
            idx_test = dataset.test_idx.cpu().numpy()

            sens = dataset.sensitive_labels
            idx_sens_train = torch.LongTensor(
                np.intersect1d(
                    (sens == 1).nonzero(as_tuple=True)[0].cpu().numpy(),
                    idx_train_gnn.cpu().numpy(),
                )
            )

            sens_attr = "region" if dataset_name in ["pokec_z", "pokec_n"] else "WHITE"
            sens_number = args.sens_number

            seed = seed_set[repeat]
            np.random.seed(seed)
            torch.manual_seed(seed)
            if args.cuda:
                torch.cuda.manual_seed(seed)

            attacked_adj = attack(
                args,
                ptb_rate,
                adj,
                features,
                labels,
                sens,
                idx_train_atk,
                idx_val,
                idx_test,
                seed,
                dataset,
                sens_attr,
                idx_sens_train,
            )

            file_path = os.path.join(
                "..",
                "data",
                "dice",
                dataset_name,
                "statistical_parity",
            )
            try:
                os.makedirs(file_path)
            except:
                pass
            file_name = (
                f"rate={ptb_rate}_mode=add_steps=1_lr=0.01_nepochs=100_seed={seed}.pt"
            )
            file_path = os.path.join(file_path, file_name)

            torch.save(
                {
                    "num_nodes": dataset.num_nodes,
                    "num_edges": int(
                        (attacked_adj.nnz + attacked_adj.diagonal().sum()) // 2
                    )
                    - attacked_adj.shape[0],
                    "num_node_features": dataset.num_node_features,
                    "num_classes": dataset.num_classes,
                    "adjacency_matrix": attacked_adj,
                    "node_features": dataset.features.cpu().numpy(),
                    "labels": dataset.labels.cpu().numpy(),
                    "sensitive_labels": dataset.sensitive_labels.cpu().numpy(),
                    "train_idx": dataset.train_idx.cpu().numpy(),
                    "val_idx": dataset.val_idx.cpu().numpy(),
                    "test_idx": dataset.test_idx.cpu().numpy(),
                },
                file_path,
            )
