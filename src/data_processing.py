import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

from tqdm import tqdm

import os

DATA_PATH = "D:/DATA/"

def load_data(name_data):
    if name_data == 'cifar10':
        # Define transformations for CIFAR-10 data
        transforms_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transforms_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        # Load CIFAR-10 dataset
        trainset = datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=transforms_train)
        testset = datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=transforms_test)
        # CIFAR-10 has 3 color channels (RGB)
        data_dimension = 3
    elif name_data == 'mnist':
        # Define transformations for MNIST data
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
        # Load MNIST dataset        
        trainset = datasets.MNIST(root=DATA_PATH, train=True, download=True, transform=transform_train)
        testset = datasets.MNIST(root=DATA_PATH, train=False, download=True, transform=transform_test)     
        # MNIST has 1 color channel (grayscale)
        data_dimension = 1
    elif name_data == "fmnist":
        # Define transformations for FashionMNIST data
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
        # Load MNIST dataset        
        trainset = datasets.FashionMNIST(root=DATA_PATH, train=True, download=True, transform=transform_train)
        testset = datasets.FashionMNIST(root=DATA_PATH, train=False, download=True, transform=transform_test)     
        # MNIST has 1 color channel (grayscale)
        data_dimension = 1
    return trainset, testset, data_dimension

def data_split(dataset, data_distribution):
    # Generate a list of dataset indices and shuffle them
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)

    # Initialize empty list to store data indices for each client
    client_data_indices = []

    k = 0
    for client_data_number in data_distribution:
        # Append a sublist containing `client_data_number` data indices starting from index `k`
        client_data_indices.append(indices[k:k+client_data_number])

        # Update the starting index for the next client
        k += client_data_number

    return client_data_indices

def data_split_iidness(dataset, beta, num_clients, case):
    """
    :param case: case = 1: approximately same number each client; case = 2: some clients get 10 times less data
    """
    y_train = torch.from_numpy(np.array(dataset.targets))
    num_classes = torch.unique(y_train).shape[0]

    # shuffle the data indices
    data_indices = np.arange(len(dataset))
    np.random.shuffle(data_indices)
    clients_data_indices = []   # the indices of the data in each client

    num_each_client = len(dataset) // num_clients
    for k in range(num_clients):
        # arange num_each_client data into each clients
        num_this = num_each_client
        if case == 2 and np.random.uniform(0,1,1)[0] > 0.5:
            num_this //= 2
        clients_data_indices.append(data_indices[k * num_this: (k + 1) * num_this])

    # clients_data_indices = torch.from_numpy(np.array(clients_data_indices))

    # this is the data after splitting and sampling
    splitted_client_indices = []

    # min_size = float("inf") # truncation, the size of each client can be different due to round up and randomness
    for client_id in range(num_clients):
        client_data_indices = torch.from_numpy(np.array(clients_data_indices[client_id]))
        # the targets of thie client
        y_client = y_train[client_data_indices]
        # beta sample, higher beta lead to more averaged number of classes
        labels_distribution = np.random.dirichlet(np.repeat(beta, num_classes))
        # print(np.argmax(labels_distribution))

        # indices of data after beta sampling for each client
        sampled_client_data_indices = None
        for label in range(num_classes):
            # how many samples are there for each label? 
            num_labels = int(np.round(num_each_client * labels_distribution[label]))
            y_client_label_iis = torch.where(y_client == label)[0]  # where are those labels. they are indices of indices for original dataset

            num_labels_exist = y_client_label_iis.shape[0]
            if num_labels_exist < num_labels:
                # if there are not enough samples for this label, add some more
                more_label_iiis = torch.randint(0, num_labels_exist, (int(num_labels - num_labels_exist),)).flatten()   # they are indices for indices for indices of original dataset
                y_client_label_iis = torch.concat([y_client_label_iis, y_client_label_iis[more_label_iiis]], dim=0)
            else:
                # enough? Truncate!
                y_client_label_iis = y_client_label_iis[:num_labels]
            # recover indices in original dataset
            y_client_label_is = client_data_indices[y_client_label_iis]

            # concatenate them into the indices for dataset in each sample
            if sampled_client_data_indices == None:
                sampled_client_data_indices = y_client_label_is
            else:
                sampled_client_data_indices = torch.concat([sampled_client_data_indices, y_client_label_is], dim=0)
        
        # randomly permute them
        n_ids = sampled_client_data_indices.shape[0]
        sampled_client_data_indices = sampled_client_data_indices[torch.randperm(n_ids)]

        # split for this client is done
        splitted_client_indices.append(sampled_client_data_indices.numpy())
        # min_size = min(min_size, sampled_client_data_indices.shape[0])
    
    # truncate them so that each client has the same number of datas
    # splitted_client_indices = np.array([a[:min_size] for a in splitted_client_indices])
    # return torch.from_numpy(splitted_client_indices)
    return splitted_client_indices

def split_and_store(dataset, beta, num_clients, data_dir, case):
    # store the data for each client into a .npy file
    data_dir += f"case{case}/Beta-{int(beta) if beta.is_integer() else beta}/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # get the splitted indices
    splitted_client_indices = data_split_iidness(dataset, beta, num_clients, case)
    y_dataset = np.array(dataset.targets, dtype=np.int64)   # get labels of the dataset, in case cifar10 targets is a list
    for client_id in tqdm(range(num_clients)):
        client_indices = splitted_client_indices[client_id]
        client_dataset = Subset(dataset, client_indices)

        X_client = []
        for X, _ in client_dataset:
            X_client.append(X.numpy())
        
        X_client = np.array(X_client, dtype=np.float32)
        y_client = y_dataset[client_indices]

        np.savez(data_dir + f"client_{client_id}", X=X_client, y=y_client)