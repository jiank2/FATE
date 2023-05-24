import argparse

import torch

from utils.configs import init_attack_configs, init_dataset_configs, init_train_configs
from utils.trainer_fairgnn import FairGNNTrainer
from utils.trainer_gcn import GCNTrainer


def train(
    args,
    dataset_configs,
    attack_configs,
    train_configs,
    random_seed_list,
):
    if train_configs["model"] in (
        "gcn",
        "gat",
        "inform_gcn",
    ):
        trainer = GCNTrainer(
            dataset_configs=dataset_configs,
            train_configs=train_configs,
            attack_configs=attack_configs,
            no_cuda=(not args.enable_cuda),
            device=(
                torch.device(f"cuda:{args.cuda_device}") if args.enable_cuda else "cpu"
            ),
            random_seed_list=random_seed_list,
            attack_method=args.attack_method,
        )
        trainer.train()
    elif train_configs["model"] in ("fairgnn",):
        trainer = FairGNNTrainer(
            dataset_configs=dataset_configs,
            train_configs=train_configs,
            attack_configs=attack_configs,
            no_cuda=(not args.enable_cuda),
            device=(
                torch.device(f"cuda:{args.cuda_device}") if args.enable_cuda else "cpu"
            ),
            random_seed_list=random_seed_list,
            attack_method=args.attack_method,
        )
        trainer.train()
    else:
        raise ValueError(
            "Model in train_configs should be one of (gcn, gat, fairgnn, inform_gcn)!"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="pokec_n",
        help="choose the dataset to attack.",
        choices=["pokec_n", "pokec_z", "bail"],
    )
    parser.add_argument(
        "--fairness",
        type=str,
        default="statistical_parity",
        help="choose the fairness definition to attack.",
        choices=["statistical_parity", "individual_fairness"],
    )
    parser.add_argument(
        "--ptb_mode",
        type=str,
        default="flip",
        help="flip or add edges.",
        choices=["flip", "add"],
    )
    parser.add_argument(
        "--ptb_rate", type=float, default=0.05, help="perturbation rate."
    )
    parser.add_argument(
        "--attack_steps", type=int, default=3, help="number of attacking steps."
    )
    parser.add_argument(
        "--attack_seed", type=int, default=25, help="random seed to set in attacker."
    )
    parser.add_argument(
        "--attack_method",
        type=str,
        default="fate",
        help="which attacking method's poisoned dataset.",
        choices=["fate", "random", "dice", "fagnn"],
    )
    parser.add_arugment(
        "--victim_model",
        type=str,
        default="gcn",
        help="victim model to train.",
        choices=[
            "gcn",
            "fairgnn",
            "inform_gcn",
            "gat",
        ],
    )
    parser.add_argument(
        "--hidden_dimension",
        type=int,
        default=128,
        help="hidden dimension of the victim model.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=400,
        help="number of epochs to train the victim model.",
    )
    parser.add_argument(
        "--enable_cuda", action="store_true", default=True, help="enable CUDA."
    )
    parser.add_argument(
        "--cuda_device", type=int, default="which GPU to use.", choices=[0, 1, 2, 3]
    )

    args = parser.parse_args()

    dataset_configs = init_dataset_configs()
    attack_configs = init_attack_configs()
    train_configs = init_train_configs()

    attack_configs["inform_similarity_measure"] = "cosine"
    train_configs["weight_decay"] = 1e-5
    train_configs["lr"] = 1e-3

    dataset_configs["name"] = attack_configs["dataset"] = args.dataset

    attack_configs["fairness_definition"] = args.fairness
    attack_configs["perturbation_mode"] = args.ptb_mode
    attack_configs["perturbation_rate"] = args.ptb_rate
    attack_configs["attack_steps"] = args.attack_steps
    attack_configs["seed"] = args.attack_seed

    train_configs["model"] = args.victim_model
    train_configs["hidden_dimension"] = args.hidden_dimension
    train_configs["num_epochs"] = args.num_epochs

    train(
        args=args,
        dataset_configs=dataset_configs,
        attack_configs=attack_configs,
        train_configs=train_configs,
        random_seed_list=[
            0,
            1,
            2,
            42,
            100,
        ],
    )
