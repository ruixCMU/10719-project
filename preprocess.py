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

def sanity_check(data_dir = "D:/DATA/mnist_splitted/"):
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

if __name__ == "__main__":
    trainset, testset, data_dimension = load_data("mnist")
    num_clients = 100
    split_and_store(trainset, 1e4, num_clients)

    min_min, max_max = sanity_check()
    print(min_min, max_max)