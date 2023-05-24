import argparse

import numpy as np
import torch

from utils.attacker import Attacker
from utils.configs import init_attack_configs, init_dataset_configs
from utils.data_loader import GraphDataset
from utils.helper_functions import random_split


def generate_splits(args, dataset, num_splits, split_seed=1684992425):
    # set random seed
    if split_seed is not None:
        np.random.seed(split_seed)
        torch.manual_seed(split_seed)
        if args.enable_cuda:
            torch.cuda.manual_seed(split_seed)

    # generate splits
    res = list()
    for _ in range(num_splits):
        res.append(random_split(dataset))
    return res


def attack(
    args,
    dataset_configs,
    attack_configs,
):
    # load dataset
    dataset = GraphDataset(dataset_configs)

    # generate split
    splits = generate_splits(
        args=args,
        dataset=dataset,
        num_splits=1,
    )
    split = splits[0]

    dataset.set_random_split(split)

    attacker = Attacker(
        attack_configs=attack_configs,
        data=dataset,
        no_cuda=(not args.enable_cuda),
        device=torch.device(f"cuda:{args.cuda_device}") if args.enable_cuda else "cpu",
        random_seed=args.attack_seed,
    )

    attacker.attack()


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
        "--enable_cuda", action="store_true", default=True, help="enable CUDA."
    )
    parser.add_argument("--cuda_device", type=int, default="which GPU to use.")

    args = parser.parse_args()

    dataset_configs = init_dataset_configs()
    attack_configs = init_attack_configs()

    dataset_configs["name"] = attack_configs["dataset"] = args.dataset
    attack_configs["fairness_definition"] = args.fairness
    attack_configs["perturbation_mode"] = args.ptb_mode
    attack_configs["perturbation_rate"] = args.ptb_rate
    attack_configs["attack_steps"] = args.attack_steps

    attack(
        args=args,
        dataset_configs=dataset_configs,
        attack_configs=attack_configs,
    )
