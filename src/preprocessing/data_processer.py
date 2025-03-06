from odir_dataset import OdirDataset

from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])
transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])

odir = OdirDataset(root_dir="../../data/odir/ODIR-5K", data_folder="raw", labels_csv="labels_perEye.csv",
                    dataset_crop="diabetes", transform=transform)
print(odir.dataset_crop)