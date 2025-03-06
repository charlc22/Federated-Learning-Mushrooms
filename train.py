import sys
import os.path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import Metrics

# setup device/gpu
torch.cuda.set_device(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training using {device} with PyTorch {torch.__version__} and Flower {fl.__version__}")