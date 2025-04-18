import os
os.environ["RAY_memory_monitor_refresh_ms"] = "0"
import yaml
import pickle
from pathlib import Path
from typing import List

import numpy as np
import torch
import flwr as fl

from dataset import load_data
from model import getNet
from client import generate_client_fn
from server import get_on_fit_config, get_evaluate_fn

# hyperparameters
with open("conf/config.yaml", "r") as f:
    config = yaml.safe_load(f)

if config["dataset_crops"] is None:
    num_classes = 8
else:
    num_classes = 1 + len(config["dataset_crops"])
print(f"num_classes: {num_classes}")

torch.cuda.empty_cache()

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

#-------------------------------------
def main():
    train_loaders, val_loaders, test_loader = load_data(config["num_clients"])

    client_fn = generate_client_fn(
        trainloaders=train_loaders,
        valloaders=val_loaders,
        num_classes=num_classes
    )

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.0,       
        min_fit_clients=config["num_clients_per_round_fit"],
        fraction_evaluate=0.0,  
        min_evaluate_clients=config["num_clients_per_round_eval"],
        min_available_clients=config["num_clients"],
        on_fit_config_fn=get_on_fit_config(
            config["config_fit"]
        ),
        evaluate_fn=get_evaluate_fn(num_classes, test_loader),
        #initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(getNet(num_classes))),
        #server_learning_rate=0.5,
        #server_momentum=0.9,
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,                # a function that spawns a particular client
        num_clients=config["num_clients"],  # total number of clients
        config=fl.server.ServerConfig(
            num_rounds=config["num_rounds"]
        ),
        strategy=strategy,                  # our strategy of choice
        client_resources={
            "num_cpus": 2,                  
            "num_gpus": 0.5,                
        },
    )

    # save results
    results_path = "/home/raymond/Coding/2324researchFL/models/results.pkl"
    results = {"history": history}
    with open(str(results_path), "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()