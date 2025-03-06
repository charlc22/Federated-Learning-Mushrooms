# fed. learning client file

import sys
import pickle
from collections import OrderedDict
from typing import List, Tuple

sys.path.append("../")
from preprocessing.odir_dataset import OdirDataset
from model import getNet, train, test

import yaml
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import flwr as fl


with open("conf/config.yaml", "r") as f:
    config = yaml.safe_load(f)

EPOCHS = config["config_fit"]["epochs"]

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, numclasses):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        nn = getNet(numclasses).to(self.device)
        self.net = nn
        self.trainloader = trainloader
        self.valloader = valloader
        self.numclasses = numclasses


    def set_parameters(self, parameters):
        """ Takes given parameters, sets them to the current model """
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

        self.net.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        """ Return the current local model parameters """
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def fit(self, parameters, config):
        """
            Receive model parameters from the server & train the model parameters on the local data
            Returns: the (updated) model parameters to the server
        """
        self.set_parameters(parameters)
        train(self.net, self.trainloader, EPOCHS, device=self.device)
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        """
            Receive model parameters from the server & evaluate the model parameters on the local data
            Returns: the evaluation result to the server
        """
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader, device=self.device)
        return float(loss), len(self.testloader), {"accuracy": float(accuracy)}
    


def generate_client_fn(trainloaders, valloaders, num_classes):
    """
        Return a function that can be used by the VirtualClientEngine.
        to spawn a FlowerClient with client id `cid`.
    """
    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""

        return FlowerClient(
            trainloader=trainloaders[int(cid)],
            valloader=valloaders[int(cid)],
            numclasses=num_classes
        ).to_client()
    
    return client_fn