import copy
import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from models.fairgnn import FairGNN
from utils.evaluator import Evaluator
from utils.helper_functions import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FairGNNTrainer:
    def __init__(
        self,
        dataset_configs,
        train_configs,
        attack_configs,
        no_cuda,
        device,
        random_seed_list,
        attack_method=None,
    ):
        # get configs
        self.dataset_configs = dataset_configs

        self.train_configs = train_configs

        self.attack_configs = attack_configs
        self.attack_folds = self.attack_configs["num_cross_validation_folds"]

        self.attack_method = attack_method if attack_method else None

        # get cuda-related info
        self.no_cuda = no_cuda
        self.device = device

        # get random seed list
        self.random_seed_list = random_seed_list

    def _init_params(self, random_seed):
        # set random seed
        self.random_seed = random_seed
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)
            if not self.no_cuda:
                torch.cuda.manual_seed(self.random_seed)

        # load data
        vanilla_data, attacked_data = self._load_data()
        self.original_graph = sparse_matrix_to_sparse_tensor(
            symmetric_normalize(
                vanilla_data["adjacency_matrix"]
                + sp.eye(vanilla_data["adjacency_matrix"].shape[0])
            )
        ).to_dense()
        self.attacked_graph = sparse_matrix_to_sparse_tensor(
            symmetric_normalize(
                attacked_data["adjacency_matrix"]
                + sp.eye(attacked_data["adjacency_matrix"].shape[0])
            )
        ).to_dense()
        self.features = torch.FloatTensor(attacked_data["node_features"])
        self.labels = torch.LongTensor(attacked_data["labels"])
        self.sensitive_labels = torch.LongTensor(attacked_data["sensitive_labels"])
        self.train_idx = torch.LongTensor(attacked_data["train_idx"])
        self.val_idx = torch.LongTensor(attacked_data["val_idx"])
        self.test_idx = torch.LongTensor(
            np.setdiff1d(
                np.arange(attacked_data["adjacency_matrix"].shape[0]),
                np.union1d(attacked_data["train_idx"], attacked_data["val_idx"]),
            )
        )

        # get models
        self.vanilla_model = FairGNN(
            nfeat=self.features.shape[1],
            nhid=self.train_configs["hidden_dimension"],
            dropout=self.train_configs["dropout"],
            lr=self.train_configs["lr"],
            weight_decay=self.train_configs["weight_decay"],
        )
        self.attacked_model = copy.deepcopy(self.vanilla_model)

        # move to corresponding device
        if not self.no_cuda:
            self.original_graph = self.original_graph.to(self.device)
            self.attacked_graph = self.attacked_graph.to(self.device)
            self.features = self.features.to(self.device)
            self.labels = self.labels.to(self.device)
            self.sensitive_labels = self.sensitive_labels.to(self.device)
            self.train_idx = self.train_idx.to(self.device)
            self.val_idx = self.val_idx.to(self.device)
            self.test_idx = self.test_idx.to(self.device)
            self.vanilla_model.to(self.device)
            self.attacked_model.to(self.device)

        # init optimizers
        self.vanilla_model.init_optimizers()
        self.attacked_model.init_optimizers()

        # init loss
        self.utility_criterion = nn.BCEWithLogitsLoss()

        # init evaluator
        self.evaluator = Evaluator()

    def train(
        self,
    ):
        result = dict()

        for random_seed in self.random_seed_list:
            self._init_params(random_seed)

            vanilla_best_eval, vanilla_best_test = self._train_vanilla()
            attacked_best_eval, attacked_best_test = self._train_attacked()
            result[random_seed] = {
                "vanilla": {
                    "eval": vanilla_best_eval,
                    "test": vanilla_best_test,
                },
                "attacked": {
                    "eval": attacked_best_eval,
                    "test": attacked_best_test,
                },
            }

        self._save_result(self._get_mean_and_std(result))

    def _train_vanilla(self):
        best_val = {
            "micro_f1": -1.0,
            "macro_f1": -1.0,
            "binary_f1": -1.0,
            "roc_auc": -1.0,
            "bias": 100,
        }
        best_test = copy.deepcopy(best_val)
        for _ in range(self.train_configs["num_epochs"]):
            # train
            self.vanilla_model.train()
            self.vanilla_model.optimize(
                adj=self.original_graph,
                x=self.features,
                labels=self.labels,
                sensitive_labels=self.sensitive_labels,
                idx_train=self.train_idx,
                alpha=self.train_configs["fairgnn_regularization"]["alpha"],
                beta=self.train_configs["fairgnn_regularization"]["beta"],
                retain_graph=False,
                enable_update=True,
            )
            vanilla_loss_train = self.vanilla_model.loss_classifiers.detach()
            vanilla_output = self.vanilla_model(self.original_graph, self.features)
            _ = self.evaluator.eval(
                loss=vanilla_loss_train.detach().item(),
                output=vanilla_output,
                labels=self.labels,
                sensitive_labels=self.sensitive_labels,
                idx=self.train_idx,
                stage="train",
            )

            # val
            self.vanilla_model.eval()
            vanilla_output = self.vanilla_model(self.original_graph, self.features)
            vanilla_loss_val = 0
            vanilla_eval_val_result = self.evaluator.eval(
                loss=vanilla_loss_val,
                output=vanilla_output,
                labels=self.labels,
                sensitive_labels=self.sensitive_labels,
                idx=self.val_idx,
                stage="validation",
            )

            # test
            if vanilla_eval_val_result["micro_f1"] > best_val["micro_f1"]:
                best_val = vanilla_eval_val_result
                vanilla_loss_test = 0
                best_test = self.evaluator.eval(
                    loss=vanilla_loss_test,
                    output=vanilla_output,
                    labels=self.labels,
                    sensitive_labels=self.sensitive_labels,
                    idx=self.test_idx,
                    stage="test",
                )
                # self._save_model_ckpts(self.vanilla_model)
        return best_val, best_test

    def _train_attacked(self):
        best_val = {
            "micro_f1": -1.0,
            "macro_f1": -1.0,
            "binary_f1": -1.0,
            "roc_auc": -1.0,
            "bias": 100,
        }
        best_test = copy.deepcopy(best_val)
        for _ in range(self.train_configs["num_epochs"]):
            # train
            self.attacked_model.train()
            self.attacked_model.optimize(
                adj=self.attacked_graph,
                x=self.features,
                labels=self.labels,
                sensitive_labels=self.sensitive_labels,
                idx_train=self.train_idx,
                alpha=self.train_configs["fairgnn_regularization"]["alpha"],
                beta=self.train_configs["fairgnn_regularization"]["beta"],
                retain_graph=False,
                enable_update=True,
            )
            attacked_loss_train = self.attacked_model.loss_classifiers.detach()
            attacked_output = self.attacked_model(self.attacked_graph, self.features)
            _ = self.evaluator.eval(
                loss=attacked_loss_train.detach().item(),
                output=attacked_output,
                labels=self.labels,
                sensitive_labels=self.sensitive_labels,
                idx=self.train_idx,
                stage="train",
            )

            # val
            self.attacked_model.eval()
            attacked_output = self.attacked_model(self.attacked_graph, self.features)
            attacked_loss_val = 0

            attacked_eval_val_result = self.evaluator.eval(
                loss=attacked_loss_val,
                output=attacked_output,
                labels=self.labels,
                sensitive_labels=self.sensitive_labels,
                idx=self.val_idx,
                stage="validation",
            )

            # test
            if attacked_eval_val_result["micro_f1"] > best_val["micro_f1"]:
                best_val = attacked_eval_val_result
                attacked_loss_test = 0
                best_test = self.evaluator.eval(
                    loss=attacked_loss_test,
                    output=attacked_output,
                    labels=self.labels,
                    sensitive_labels=self.sensitive_labels,
                    idx=self.test_idx,
                    stage="test",
                )
                # self._save_model_ckpts(self.attacked_model)
        return best_val, best_test

    @staticmethod
    def _get_accuracy(output, labels):
        if output.shape[1] == 1:
            preds = (output.squeeze() > 0).type_as(labels)
        else:
            preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

    def _load_data(self):
        # load vanilla data
        file_path = os.path.join(
            "..",
            "data",
            "clean",
            self.dataset_configs["name"],
            "{name}_{prefix_sensitive_attr}_sensitive_attr.pt".format(
                name=self.dataset_configs["name"],
                prefix_sensitive_attr="no"
                if self.dataset_configs["no_sensitive_attribute"]
                else "with",
            ),
        )

        vanilla_data = torch.load(file_path)

        # load attacked data
        attack_setting = "rate={rate}_mode={mode}_steps={steps}_lr={lr}_nepochs={nepochs}_seed={seed}".format(
            rate=self.attack_configs["perturbation_rate"],
            mode=self.attack_configs["perturbation_mode"],
            steps=self.attack_configs["attack_steps"],
            lr=self.attack_configs["pre_train_lr"],
            nepochs=self.attack_configs["pre_train_num_epochs"],
            seed=self.attack_configs["seed"],
        )

        folder_path = os.path.join(
            "..",
            "data",
            f"{self.attack_method}" if self.attack_method else "perturbed",
            self.attack_configs["dataset"],
            self.attack_configs["fairness_definition"],
        )

        # if not os.path.isdir(folder_path):
        try:
            os.makedirs(folder_path)
        except:
            pass

        attacked_data = torch.load(os.path.join(folder_path, f"{attack_setting}.pt"))

        return vanilla_data, attacked_data

    def _get_path(self, result_type="ckpts"):
        attack_setting = "rate={rate}_mode={mode}_steps={steps}_lr={lr}_nepochs={nepochs}_seed={seed}".format(
            rate=self.attack_configs["perturbation_rate"],
            mode=self.attack_configs["perturbation_mode"],
            steps=self.attack_configs["attack_steps"],
            lr=self.attack_configs["pre_train_lr"],
            nepochs=self.attack_configs["pre_train_num_epochs"],
            seed=self.attack_configs["seed"],
        )

        train_setting = "model={model}_lr={lr}_weight-decay={weight_decay}_hidden-dim={hidden_dim}_informreg={reg}".format(
            model=self.train_configs["model"],
            lr=self.train_configs["lr"],
            weight_decay=self.train_configs["weight_decay"],
            hidden_dim=self.train_configs["hidden_dimension"],
            reg=self.train_configs["inform_regularization"],
        )

        folder_path = os.path.join(
            "..",
            f"{result_type}_{self.attack_method}"
            if self.attack_method
            else result_type,
            self.attack_configs["dataset"],
            self.attack_configs["fairness_definition"],
            f"{attack_setting}",
        )

        # if not os.path.isdir(folder_path):
        try:
            os.makedirs(folder_path)
        except:
            pass

        return folder_path, train_setting

    def _save_result(self, result):
        folder, train_setting = self._get_path(result_type="result")
        file_path = os.path.join(folder, f"{train_setting}.txt")
        with open(file_path, "w") as outfile:
            json.dump(result, outfile, indent=4)

    def _save_model_ckpts(self, model):
        folder, train_setting = self._get_path(result_type="ckpts")
        file_path = os.path.join(folder, f"{train_setting}.pt")
        torch.save(model.state_dict(), file_path)

    def _get_mean_and_std(
        self,
        result,
    ):
        if len(result) == 0:
            return result

        metrics_list = [
            "micro_f1",
            "macro_f1",
            "binary_f1",
            "roc_auc",
            "bias",
        ]

        stats = dict()
        for result_type in ["vanilla", "attacked"]:
            stats[result_type] = dict()
            stats[result_type]["eval"] = dict()
            stats[result_type]["test"] = dict()
            for metric in metrics_list:
                if result[self.random_seed_list[0]][result_type]["eval"] == {
                    "micro_f1": -1.0
                }:
                    eval_res = [-1.0]
                else:
                    eval_res = [
                        v[result_type]["eval"][metric] for k, v in result.items()
                    ]
                test_res = [v[result_type]["test"][metric] for k, v in result.items()]
                stats[result_type]["eval"][metric] = {
                    "mean": np.mean(eval_res),
                    "std": np.std(eval_res),
                }
                stats[result_type]["test"][metric] = {
                    "mean": np.mean(test_res),
                    "std": np.std(test_res),
                }
        result["stats"] = stats
        return result
