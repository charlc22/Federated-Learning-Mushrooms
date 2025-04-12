import os
import sys

# Add the src directory to the path to make preprocessing accessible
current_dir = os.path.dirname(os.path.abspath(__file__))  # centralized directory
src_dir = os.path.dirname(current_dir)  # src directory
sys.path.append(src_dir)

from ..preprocessing.mushroom_dataset import MushroomDataset
import yaml

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# hyperparameters
with open("conf/config.yaml", "r") as f:
    config = yaml.safe_load(f)

batch_size = config["batch_size"]

def load_data():
    """
        Load Mushroom dataset & apply transforms for centralized training
        Returns: train_loader, val_loader
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

    # Create data loaders (no splitting for federated learning)
    train_loader = DataLoader(
        dataset=mushroom_train,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        dataset=mushroom_val,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader