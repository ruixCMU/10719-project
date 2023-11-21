import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset

import random
import numpy as np
import time
import argparse
import copy
from tqdm import tqdm

from src.aggregation import fedavg
from src.data_processing import load_data, data_split, split_and_store
from src.model import ResNet18, MLP
from src.training import local_training
from src.testing import global_testing

def _get_args():
    p = argparse.ArgumentParser()
    # Define command-line arguments
    p.add_argument("--seed", help="seed", type=int, default=0)
    p.add_argument("--data_name", help="name of dataset, cifar10 or mnist", type=str, choices=["cifar10", "mnist"])
    p.add_argument("--beta", help="beta, the higher, the more iid, larger than 1e-2", type=float, default=1e4)
    p.add_argument("--partitioned", help="whether data is already partitioned", action="store_true")
    return p.parse_args()


DATA_DIR = "D:/DATA/"
PROJ_DIR = "C:/Users/ruix/Desktop/10719/proj/10719-project/"

class MyDataset(Dataset):

    def __init__(self, X, y) -> None:
        super().__init__()
        self.X = X
        self.y = y
    
    def __getitem__(self, index):
        return (self.X[index], self.y[index])
    
    def __len__(self):
        return self.X.shape[0]

if __name__=="__main__":
    args = _get_args() # Parse command-line arguments
    random_seed = args.seed
    data_name = args.data_name
    DATA_DIR += f"{args.data_name}-splitted/"

    # Set the device (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create a file to write results
    beta = int(args.beta) if args.beta.is_integer() else args.beta
    result_file_name = 'Data-'+data_name+'_'+'Seed-'+str(random_seed)+"_"+"Beta-"+str(beta)+'.txt'
    file =  open(PROJ_DIR + 'result/'+result_file_name, 'w')

    # Set random seeds for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(random_seed)

    # Hyperparameters
    learning_rate = 0.001
    momentum = 0.9
    weight_decay = 0.0001
    local_epoch = 5
    partitioning_rate = 0.1
    mini_batch_size = 50
    num_client = 100

    # Set the number of rounds based on the dataset
    if data_name =='mnist':
        num_round = 50
    elif data_name =='cifar10':
        num_round = 100
    
    # Load data
    trainset, testset, data_dimension = load_data(data_name)
    if not args.partitioned:
        split_and_store(trainset, args.beta, num_client, data_dir=DATA_DIR)

    # Initialize global model
    
    # global_model = ResNet18(data_dimension).to(device)
    if args.data_name == "cifar10":
        dim_in = 32 * 32 * 3
        model_initializer = lambda: ResNet18(data_dimension)
    else:
        dim_in = 28 * 28 * 1
        model_initializer = lambda: MLP(dim_in)

    global_model = model_initializer().to(device)
    global_data_loader = DataLoader(testset, batch_size=mini_batch_size, shuffle=False)

    print("Start training")
    print("Start training",file=file)

    with tqdm(total=num_round, bar_format="{l_bar}{bar:40}{r_bar}") as pbar:
        for round in range(num_round):
            current_time = time.time()
            global_weight = global_model.state_dict()
            local_weights = []
            local_train_accuracies = []
            selected_clients_data_num = []

            # Randomly select clients to participate in each round
            active_clients_idxs = np.random.choice(range(num_client),
                                                size=int(partitioning_rate*num_client), replace=False)
            for client_idx in active_clients_idxs:
                # local_model = ResNet18(data_dimension).to(device)
                local_model = model_initializer().to(device)
                local_weight = local_model.state_dict()
                # Initialize local model with global model weights
                for key in local_weight.keys():
                    local_weight[key] = global_weight[key]
                local_model.load_state_dict(local_weight)
                # Prepare client data
                # client_data_idxs = client_data_indices[client_idx]
                # selected_clients_data_num.append(len(client_data_idxs))
                # client_data = Subset(trainset, client_data_idxs)
                client_data_np = np.load(DATA_DIR + f"Beta-{beta}/" + f"client_{client_idx}.npz")
                client_data = MyDataset(torch.from_numpy(client_data_np["X"]), torch.from_numpy(client_data_np["y"]))
                selected_clients_data_num.append(len(client_data))
                client_data_loader = DataLoader(client_data, batch_size=mini_batch_size, shuffle=True)
                # Train the local model
                local_train_accuracies = local_training(local_model, client_data_loader, local_epoch,
                                                        learning_rate, momentum, weight_decay, device)
                # Store local model weights
                local_weights.append(copy.deepcopy(local_model.state_dict()))
                
            # Aggregate local models to update the global model
            global_weight = fedavg(local_weights,selected_clients_data_num)
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

            # Print and write results
            # print("-------------------")
            # print(f"Round [{round+1}/{num_round}]")
            # print(f"Local Training Accuracy: {np.mean(local_train_accuracies):.3f}")
            # print(f"Global Testing Accuracy: GAcc: {global_test_accuracy:.3f}")
            # print(f"Time: {(time.time()-current_time):.3f}")

            print("-------------------",file=file)
            print(f"Round [{round+1}/{num_round}]",file=file)
            print(f"Local Training Accuracy: {np.mean(local_train_accuracies):.3f}",file=file)
            print(f"Global Testing Accuracy: GAcc: {global_test_accuracy:.3f}",file=file)
            print(f"Time: {(time.time()-current_time):.3f}",file=file)

    print("-------------------")
    print("Finish training")
    print("-------------------",file=file)
    print("Finish training",file=file)

    file.close()