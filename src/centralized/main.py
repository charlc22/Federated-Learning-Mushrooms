import os
import sys

# Add the src directory to the path to make preprocessing accessible
current_dir = os.path.dirname(os.path.abspath(__file__))  # centralized directory
src_dir = os.path.dirname(current_dir)  # src directory
sys.path.append(src_dir)

os.environ["RAY_memory_monitor_refresh_ms"] = "0"
import yaml
import pickle
from pathlib import Path
from typing import List

import numpy as np
import torch

# Use absolute imports (relative to src directory)
from ..preprocessing.mushroom_dataset import MushroomDataset
# Use imports from the same directory
from dataset import load_data
# Import model from fl (since there's no model.py in centralized)
from fl.model import getNet

# hyperparameters
with open("conf/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# For mushroom dataset, we have 4 classes: edible, fresh, non-fresh, poisonous
num_classes = 4
print(f"Number of mushroom classes: {num_classes}")

torch.cuda.empty_cache()

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

#-------------------------------------
def main():
    # Using our new dataset.py which returns train_loader, val_loader
    train_loader, val_loader = load_data()

    # Initialize the model
    model = getNet(num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Training logic for centralized training
    # Add your centralized training code here

    # Save model
    results_path = "../../models/mushroom_centralized.pkl"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    torch.save(model.state_dict(), results_path)
    print(f"Model saved to {results_path}")


if __name__ == "__main__":
    main()