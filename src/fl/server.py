# fed. learning server creation file

from collections import OrderedDict
from model import getNet, test

import torch


def get_on_fit_config(config):
    """Return function that prepares config to send to clients."""

    def fit_config_fn(server_round: int):
        return {
            "lr": config["lr"],
            "epochs": config["epochs"],
        }

    return fit_config_fn

def get_evaluate_fn(num_classes, testloader):
    """Define function for global evaluation on the server."""

    def evaluate_fn(server_round: int, parameters, config):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"USING DEVICE: {device}")
        
        model = getNet(num_classes).to(device)

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy = test(model, testloader, device)

        return loss, {"accuracy": accuracy}

    return evaluate_fn