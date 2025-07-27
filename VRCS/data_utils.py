import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Subset
import random

def prepare_data(dataset_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                            (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                            (0.2023, 0.1994, 0.2010)),
    ])

    # Загрузка CIFAR-10
    trainset = torchvision.datasets.CIFAR10(
        root='/home/berezin/FederatedLearning/Datasets/Cifar10', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='/home/berezin/FederatedLearning/Datasets/Cifar10', train=False, download=True, transform=transform_test)
    
    return trainset, testset


def f_and_g_part_split(dataset):
    targets = np.array(dataset.targets)

    indices_1 = np.where(np.isin(targets, range(0,5)))[0]
    indices_2 = np.where(np.isin(targets, range(5,10)))[0]

    subset1 = Subset(dataset, indices_1)
    subset2 = Subset(dataset, indices_2)

    return subset1, subset2


def server_n_clients_split(dataset, server_part=0.09, client_count=10, seed=42):
    random.seed(seed)

    total_size = len(dataset)
    num_server = int(server_part * total_size)
    num_client_total = total_size - num_server
    data_per_client = num_client_total // client_count

    all_indeces = list(range(total_size))
    random.shuffle(all_indeces)

    server_indices = all_indeces[:num_server]
    client_indices = all_indeces[num_server:]

    client_subsets = []
    for i in range(client_count):
        start = i * data_per_client
        end = (i + 1) * data_per_client
        indices = client_indices[start:end]
        subset = Subset(dataset,indices)
        client_subsets.append(subset)

    server_subset = Subset(dataset, server_indices)
    all_clients_dataset = Subset(dataset, client_indices)
    return server_subset, all_clients_dataset, client_subsets




def create_all_dataloaders(batch_size=700):
    # Need hydra cfg
    client_count = 10
    batch_size=batch_size
    server_part = 0.09
    dataset_size=50_000


    Train, Test= prepare_data(dataset_size)


    Train_f, Train_g = f_and_g_part_split(Train)

    train_server_f, concat_client_f_dataset, train_clients_f_datasets = server_n_clients_split(Train_f,
                                                                                               server_part=server_part,
                                                                                               client_count=client_count//2)
    train_server_g, concat_client_g_dataset, train_clients_g_datasets = server_n_clients_split(Train_g,
                                                                                               server_part=server_part,
                                                                                               client_count=client_count//2)

    server_all_data = Subset(train_server_f.dataset, list(train_server_f.indices) + list(train_server_g.indices))

    train_all_loader = DataLoader(Train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_all_loader = DataLoader(Test, batch_size=batch_size, shuffle=True, num_workers=4)
    server_all_loader = DataLoader(server_all_data, batch_size=batch_size, shuffle=True, num_workers=4)
    server_f_loader = DataLoader(train_server_f, batch_size=batch_size, shuffle=True, num_workers=4)
    server_g_loader = DataLoader(train_server_g, batch_size=batch_size, shuffle=True, num_workers=4)
    all_clients_f_loader = DataLoader(concat_client_f_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    all_clients_g_loader = DataLoader(concat_client_g_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return {
        "train": train_all_loader,
        "test": test_all_loader,
        "server": server_all_loader,
        "server_f": server_f_loader,
        "server_g": server_g_loader,
        "clients_f": all_clients_f_loader,
        "clients_g": all_clients_g_loader,
        #"individual_clients": individual_client_loaders
    }

