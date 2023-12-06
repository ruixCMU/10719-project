import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset

import random
import numpy as np
import time
import argparse
import copy
from tqdm import tqdm
import pandas as pd
import os

from src.aggregation import *
from src.data_processing import load_data, data_split, split_and_store, MyDataset
from src.model import ResNet18, MLP
from src.training import local_training
from src.testing import global_testing

def _get_args():
    p = argparse.ArgumentParser()
    # Define command-line arguments
    p.add_argument("--data_name", help="name of datasett", type=str, choices=["cifar10", "mnist", "fmnist"])
    p.add_argument("--aggregation", help="name of aggregation method", type=str, choices=["FedSGD", "FedAvg", "FedAdaGrad", "FedYogi", "FedAdam"], default="FedAvg")
    p.add_argument("--beta", help="beta, the higher, the more iid, larger than 1e-2", type=float, default=1e4)
    p.add_argument("--model_name", help="name of pretrained model", type=str)
    p.add_argument("--local_epoch", help="number of local epochs", type=int, default=5)
    p.add_argument("--case", help="case of clients", type=int, default=1)
    p.add_argument("--num_clients", help="number of clients", type=int, default=100)
    p.add_argument("--num_rounds", help="number of rounds on server", type=int, default=50)
    p.add_argument("--ratio_mixed_iid", help="ratio of clients that are completely iid", type=float, default=0.)
    p.add_argument("--partitioned", help="whether data is already partitioned, only if ratio_mixed_iid is 0.0", action="store_true")
    return p.parse_args()

def find_goal_and_val_acc(model_name: str) -> tuple[float, float]:
    # helper function to parse goal and final validation accuracy from model name
    idx1 = model_name.index("goal=") + len("goal=")
    idx2 = model_name[idx1:].index("_")
    goal = float(model_name[idx1:idx1+idx2])

    idx1 = model_name.index("final-acc=") + len("final-acc=")
    pre_acc = float(model_name[idx1:])
    return goal, pre_acc

DATA_DIR = "D:/DATA/"
PROJ_DIR = "C:/Users/ruix/Desktop/10719/proj/10719-project/"
MODEL_DIR = PROJ_DIR + "models/"

if __name__=="__main__":
    args = _get_args() # Parse command-line arguments
    data_name = args.data_name
    beta = int(args.beta) if args.beta.is_integer() else args.beta

    DATA_DIR += f"{args.data_name}-splitted/"
    MODEL_DIR += f"{args.data_name}/"
    if args.model_name is None:
        goal, pre_acc = 0, 0
        print(f"data distribution beta = {beta}, aggregation {args.aggregation}, num rounds {args.num_rounds}")
    else:
        goal, pre_acc = find_goal_and_val_acc(args.model_name)
        print(f"using pretrained model with goal = {goal}, pretrained acc = {pre_acc}, data distribution beta = {beta}, aggregation {args.aggregation}, num rounds {args.num_rounds}")

    # Set the device (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set random seeds for reproducibility
    np.random.seed(200)
    torch.manual_seed(200)
    random.seed(200)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(200)

    # Hyperparameters
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 0.0001
    local_epoch = args.local_epoch
    partitioning_rate = 0.1
    mini_batch_size = 64
    num_client = args.num_clients
    num_round = args.num_rounds
    
    result_file_name = f"Beta-{beta}_" +f"lr-{learning_rate}_ClientRatio-{partitioning_rate}_BatchSize-{mini_batch_size}_" +\
        f"NumClient-{num_client}"
    
    # Load data
    trainset, testset, data_dimension = load_data(data_name)
    iid_clients = []
    if not args.partitioned:
        iid_clients = split_and_store(trainset, args.beta, num_client, data_dir=DATA_DIR, case=args.case, iid_ratio=args.ratio_mixed_iid)

    # Initialize global model
    if args.data_name == "cifar10":
        model_initializer = lambda: ResNet18(data_dimension)
    elif args.data_name == "fmnist":
        hidden_sizes = [512, 128]
        model_initializer = lambda: MLP(hidden_sizes, [28, 28, 1], 10, "relu", True)
    else:
        hidden_sizes = [512, 128]
        model_initializer = lambda: MLP(hidden_sizes, [28, 28, 1], 10, "relu", True)

    global_model = model_initializer().to(device)
    global_data_loader = DataLoader(testset, batch_size=mini_batch_size, shuffle=False)

    # choose the aggregation methods
    if args.aggregation == "FedSGD":
        aggregator = FedAvg()
        local_epoch = 1
        mini_batch_size = len(trainset) * num_client  # make sure B = in----finite
    elif args.aggregation == "FedAvg":
        aggregator = FedAvg()
    else:
        params = {
            "beta1": 0.9,
            "beta2": 0.99,
            "eta": 0.01,
            "tor": 1e-3,
            "params": global_model.state_dict()
        }
        if args.aggregation == "FedAdam":
            aggregator = FedAdam(**params)
        elif args.aggregation == "FedAdaGrad":
            aggregator = FedAdaGrad(**params)
        elif args.aggregation == "FedYogi":
            aggregator = FedYogi(**params)
        else: raise NotImplementedError()

    # if no pretrained model is provided.
    if args.model_name is not None:
        print("loading pretrained model...")
        state_dict = torch.load(MODEL_DIR + args.model_name)
        global_model.load_state_dict(state_dict)

    # test its global accuracy now
    global_test_accuracy = global_testing(global_model, global_data_loader, device)
    print(f"the model has global accuracy {global_test_accuracy} before training")

    print("Start training")

    stats = {
        "Global Testing Accuracy": [global_test_accuracy],
        "Local Training Accuracy": []
    }
    with tqdm(total=num_round, bar_format="{l_bar}{bar:40}{r_bar}") as pbar:
        for round in range(num_round):
            current_time = time.time()
            global_weight = global_model.state_dict()
            local_weights = []
            local_train_accuracies = []
            selected_clients_data_num = []

            # Randomly select clients to participate in each round
            client_size = int(partitioning_rate * num_client)
            num_non_iid = int(np.round(client_size * (1 - args.ratio_mixed_iid)))
            active_clients_idxs = np.random.choice(range(num_client),
                                                size=num_non_iid, replace=False)
            if len(iid_clients):
                active_clients_idxs = np.concatenate((active_clients_idxs, np.random.choice(iid_clients, 
                                                    size=client_size-num_non_iid, replace=False)), axis=0)
            for client_idx in active_clients_idxs:
                # local_model = ResNet18(data_dimension).to(device)
                local_model = model_initializer().to(device)
                local_weight = local_model.state_dict()
                # Initialize local model with global model weights
                for key in local_weight.keys():
                    local_weight[key] = global_weight[key]
                local_model.load_state_dict(local_weight)
                # Prepare client data
                client_data_np = np.load(DATA_DIR + f"case{args.case}/IID-Ratio-{args.ratio_mixed_iid}/Beta-{beta}/" + f"client_{client_idx}.npz")
                client_data = MyDataset(torch.from_numpy(client_data_np["X"]), torch.from_numpy(client_data_np["y"]))
                selected_clients_data_num.append(len(client_data))
                client_data_loader = DataLoader(client_data, batch_size=mini_batch_size, shuffle=True)
                # Train the local model
                local_train_accuracies = local_training(local_model, client_data_loader, local_epoch,
                                                        learning_rate, momentum, weight_decay, device)
                # Store local model weights
                local_weights.append(copy.deepcopy(local_model.state_dict()))
                
            # Aggregate local models to update the global model
            global_weight = aggregator(global_weight, local_weights, selected_clients_data_num)
            global_model.load_state_dict(global_weight)
            # Test the global model
            global_test_accuracy = global_testing(global_model, global_data_loader, device)

            # tqdm progress bar update
            pf = {
                "ROUND": round + 1,
                "Local Training Accuracy": np.round(np.mean(local_train_accuracies), 3),
                "Global Testing Accuracy": np.round(global_test_accuracy, 3)
            }
            pbar.set_postfix(**pf)
            pbar.update(1)

            stats["Global Testing Accuracy"].append(global_test_accuracy)
            stats["Local Training Accuracy"].append(np.mean(local_train_accuracies))

    print("-------------------")
    print("Finish training")

    stats["Local Training Accuracy"] = stats["Local Training Accuracy"][:1] + stats["Local Training Accuracy"]  # for alignment with global acc
    df = pd.DataFrame(stats)

    stats_dir = PROJ_DIR + f"stats/{args.aggregation}/{args.data_name}/case{args.case}/IID-Ratio-{args.ratio_mixed_iid}/LocalEpoch-{local_epoch}/goal={goal}_pre-acc={pre_acc}/"
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    df.to_csv(stats_dir + f"{result_file_name}.csv")