import sys
sys.path.append("../")
from ..preprocessing.mushroom_dataset import MushroomDataset
import yaml

import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

# hyperparameters
with open("conf/config.yaml", "r") as f:
    config = yaml.safe_load(f)

batch_size = config["batch_size"]

def load_data(num_clients):
    """
        Load Mushroom dataset & apply transforms
        Returns: list of train_loaders, list of val_loaders, 1 test_loader
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        normalize,
    ])

    # Use direct paths to train and val as you specified
    mushroom_train = MushroomDataset(
        root_dir="../../dataset",  # Base directory
        data_folder="train",       # Subdirectory
        transform=transform
    )

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    mushroom_val = MushroomDataset(
        root_dir="../../dataset",
        data_folder="val",
        transform=val_transform
    )

    # Split training data for federated clients
    client_partition_size = 1 / num_clients
    partition_sizes = [client_partition_size] * num_clients

    train_splits = random_split(mushroom_train, partition_sizes) # list of train datasets for each client
    val_splits = random_split(mushroom_val, partition_sizes)     # list of val datasets for each client

    train_loaders = []
    val_loaders = []

    for i in range(num_clients):
        train_loaders.append(
            DataLoader(dataset=train_splits[i], batch_size=batch_size, shuffle=True)
        )
        val_loaders.append(
            DataLoader(dataset=val_splits[i], batch_size=batch_size, shuffle=False)
        )

    # Use validation data as test data
    test_loader = DataLoader(dataset=mushroom_val, batch_size=batch_size, shuffle=False)

    return train_loaders, val_loaders, test_loader