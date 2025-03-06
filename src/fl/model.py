import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from sklearn.metrics import confusion_matrix

# hyperparameters
with open("conf/config.yaml", "r") as f:
    config = yaml.safe_load(f)

batch_size = config["batch_size"]
learning_rate = config["config_fit"]["lr"]

#------------------------------------
def getNet(numclasses):
    """ Returns pretrained VGG19 w/ all weights frozen except the classifier chunk """
    net = torchvision.models.vgg19(weights="DEFAULT")
    net.classifier = nn.Sequential(
        nn.Linear(in_features=25088, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=numclasses, bias=True),
    )
    # freeze convolution weights
    for param in net.features.parameters():
        param.requires_grad = False
    
    return net

def train(net, train_loader, epochs, device):
    """
        Train the model on the training set
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.classifier.parameters(), learning_rate)
    print(f"USING DEVICE: {device}")

    for epoch in range(epochs):
        train_loss = 0
        for i, (images, labels) in enumerate(train_loader): # enumerate returns a tuple of (index, content)
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()                                   # zero the current gradients

            # forward pass
            outputs = net(images).float()

            loss = criterion(outputs, labels.float())               # run loss function on NON softmaxed data
            train_loss += loss.item()

            # backward pass
            loss.backward()                                         # calculate gradient
            optimizer.step()                                        # updates old gradients with the new calculated gradients

            print(f"Batch [{i+1}/{len(train_loader)}], Train Loss: {train_loss/(batch_size*(i+1)):.4f}")

def test(net, test_loader, device):
    """
        Test the nn on test dataset
        Returns: loss amt & accuracy %
    """
    criterion = nn.BCEWithLogitsLoss()
    test_loss = 0
    test_running_correct = 0
    print(f"USING DEVICE: {device}")
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = net(images).float()

            activ_outputs = F.sigmoid(outputs)
            rounded_outputs = (activ_outputs>=0.55).float()

            loss = criterion(outputs, labels.float())               # run loss function on NON softmaxed data
            test_loss += loss.item()

            row_equals = torch.eq(rounded_outputs.data, labels)
            row_equal_all = torch.all(row_equals, dim=1)        # Check if all elements in each row are equal
            num_equal_rows = torch.sum(row_equal_all).item()    # Count how many rows are equal
            test_running_correct += num_equal_rows

            test_accuracy = 100. * test_running_correct/(batch_size*(i+1))

            # gen confusion matrix
            """ batch_predictions = torch.argmax(rounded_outputs, dim=1).cpu().numpy()
            batch_labels = torch.argmax(labels, dim=1).cpu().numpy()
            conf_matrix += confusion_matrix(batch_labels, batch_predictions, labels=range(3)) """

            # currently, print status update EVERY BATCH
            print(f"Batch [{i+1}/{len(test_loader)}], Test Loss: {test_loss/(batch_size*(i+1)):.4f}, Test Acc: {test_accuracy:.4f}")

    final_test_loss = test_loss/(len(test_loader)*batch_size)
    final_test_acc = 100. * test_running_correct/(len(test_loader)*batch_size)

    return final_test_loss, final_test_acc