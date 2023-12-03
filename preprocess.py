import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset

import random
import numpy as np
import time
import argparse
import copy

from src.aggregation import fedavg
from src.data_processing import load_data, data_split, split_and_store
from src.model import ResNet18
from src.training import local_training
from src.testing import global_testing

import os

def sanity_check(data_dir):
    files = os.listdir(data_dir)

    min_min = float("inf")
    max_max = float("-inf")
    for file in files:
        npz = np.load(data_dir + file)
        ys = npz["y"]

        ratios = [np.sum(ys == y) / ys.shape[0] for y in range(10)]
        _min = np.min(ratios)
        _max = np.max(ratios)

        min_min = min(min_min, _min)
        max_max = max(max_max, _max)
    
    return min_min, max_max

def _get_args():
    p = argparse.ArgumentParser()
    # Define command-line arguments
    p.add_argument("--data_name", help="data_name", type=str, default="cifar10")
    p.add_argument("--case", help="client split case", type=int, choices=[1, 2], default=1)
    p.add_argument("--beta", help="beta, the higher, the more iid, larger than 1e-2", type=float, default=1e4)
    return p.parse_args()

if __name__ == "__main__":
    args = _get_args()
    trainset, testset, data_dimension = load_data(args.data_name)
    num_clients = 100
    split_and_store(trainset, args.beta, num_clients, f"D:/DATA/{args.data_name}-splitted/", args.case)

    min_min, max_max = sanity_check(f"D:/DATA/{args.data_name}-splitted/case{args.case}/Beta-{int(args.beta) if args.beta.is_integer() else args.beta}/")
    print(min_min, max_max)