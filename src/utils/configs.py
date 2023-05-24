def init_dataset_configs():
    dataset_configs = {
        "name": "pokec_n",
        "no_sensitive_attribute": True,
        "is_ratio": True,
        "split_by_class": False,
        "num_train": 20,
        "num_val": 200,
        "ratio_train": 0.5,
        "ratio_val": 0.25,
    }
    return dataset_configs


def init_attack_configs():
    attack_configs = {
        "dataset": "pokec_n",
        "model": "sgc",
        "perturbation_rate": 0.05,
        "perturbation_mode": "flip",
        "attack_steps": 3,
        "num_cross_validation_folds": 5,
        "hidden_dimension": 16,
        "pre_train_num_epochs": 100,
        "pre_train_lr": 1e-2,
        "pre_train_weight_decay": 5e-4,
        "pre_train_dropout": 0.5,
        "fairness_definition": "statistical_parity",
        "inform_similarity_measure": "cosine",
        "preferred_label": 1,
        # for KDE
        "bandwidth": 0.1,
        "tau": 0.5,
        "delta": 1.0,
    }
    return attack_configs


def init_train_configs():
    train_configs = {
        "model": "gcn",
        "lr": 1e-3,
        "weight_decay": 5e-4,
        "hidden_dimension": 128,
        "dropout": 0.5,
        "num_epochs": 400,
        # gat
        "num_heads": 8,
        # fairgnn
        "fairgnn_regularization": {
            "alpha": 100,
            "beta": 1,
        },
        "dropout_attn": 0.5,
        # individual fairness
        "inform_regularization": 0.1,
    }
    return train_configs
