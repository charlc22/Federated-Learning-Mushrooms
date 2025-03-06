import sys
sys.path.append("../")
from preprocessing.odir_dataset import OdirDataset
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# hyperparameters
with open("conf/config.yaml", "r") as f:
    config = yaml.safe_load(f)

dataset_crops = config["dataset_crops"]
batch_size = config["batch_size"]


#---------------------------------------------
def load_data(num_clients):
    """
        Load ODIR dataset & apply transforms
        Returns: list of train_loaders, list of val_loaders, 1 test_loader
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])
    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
            ])
    
    odir = OdirDataset(root_dir="../../data/odir/ODIR-5K", data_folder="raw", labels_csv="labels_perEye.csv", dataset_crops=dataset_crops,
                       transform=transform)
    data_splitting = (0.8, 0.2)
    trainvals_odir, test_odir = random_split(odir, data_splitting)

    train_odir, val_odir = random_split(trainvals_odir, data_splitting)

    client_partition_size = 1 / num_clients
    partition_sizes = [client_partition_size] * num_clients

    train_odirs = random_split(train_odir, partition_sizes) # list of train datasets for each client
    val_odirs = random_split(val_odir, partition_sizes)     # list of val datasets for each client

    train_loaders = []
    val_loaders = []

    for i in range(num_clients):
        train_loaders.append(
            DataLoader(dataset=train_odirs[i], batch_size=batch_size, shuffle=True)
        )
        val_loaders.append(
            DataLoader(dataset=val_odirs[i], batch_size=batch_size, shuffle=False)
        )
    test_loader = DataLoader(dataset=test_odir, batch_size=batch_size, shuffle=False)
    
    
    return train_loaders, val_loaders, test_loader