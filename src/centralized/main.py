import sys
sys.path.append("../")
from preprocessing.odir_dataset import OdirDataset

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms


torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# hyperparameters
dataset_crops = ["cataract", "glaucoma"]
epochs = 5
batch_size = 16
learning_rate = 0.001
if dataset_crops is None:
    num_classes = 8
else:
    num_classes = 1 + len(dataset_crops)
print(f"num_classes: {num_classes}")



net = torchvision.models.vgg19(weights="DEFAULT")

net.classifier = nn.Sequential(
    nn.Linear(in_features=25088, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=4096, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=4096, out_features=num_classes, bias=True),
)

# freeze convolution weights
for param in net.features.parameters():
    param.requires_grad = False
net.to(device)
num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Number of trainable parameters:", num_params)


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])
transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        normalize,
        ])

odir = OdirDataset(root_dir="../../data/odir/ODIR-5K", data_folder="raw", labels_csv="labels_perEye.csv", dataset_crops=dataset_crops, transform=transform)

train_odir, val_odir, test_odir = random_split(odir, (0.64, 0.16, 0.2))

train_loader = DataLoader(dataset=train_odir, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_odir, batch_size=16, shuffle=False)
test_loader = DataLoader(dataset=test_odir, batch_size=batch_size, shuffle=False)


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.classifier.parameters(), learning_rate)


# training loop
torch.set_printoptions(sci_mode=False)
net.train()
print("-------------TRAINING")
for epoch in range(epochs):
    train_loss = 0
    train_running_correct = 0
    for i, (images, labels) in enumerate(train_loader): # enumerate returns a tuple of (index, content)
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()                                   # zero the current gradients

        # forward pass
        outputs = net(images).float()

        activ_outputs = F.sigmoid(outputs)
        rounded_outputs = (activ_outputs>=0.55).float()

        loss = criterion(outputs, labels.float())               # run loss function on NON softmaxed data
        train_loss += loss.item()

        row_equals = torch.eq(rounded_outputs.data, labels)
        row_equal_all = torch.all(row_equals, dim=1)        # Check if all elements in each row are equal
        num_equal_rows = torch.sum(row_equal_all).item()    # Count how many rows are equal
        train_running_correct += num_equal_rows


        # backward pass
        loss.backward()                                         # calculate gradient
        optimizer.step()                                        # updates old gradients with the new calculated gradients

        train_accuracy = 100. * train_running_correct/((i+1)*batch_size)

        print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_loader)}], Train Loss: {train_loss/(batch_size*(i+1)):.4f}, Train Acc: {train_accuracy:.4f}")
        sys.stdout.flush()

    # val
    print("--VAL")
    val_loss = 0
    val_running_correct = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = net(images).float()

            activ_outputs = F.sigmoid(outputs)
            rounded_outputs = (activ_outputs>=0.55).float()


            loss = criterion(outputs, labels.float())               # run loss function on NON softmaxed data
            val_loss += loss.item()

            row_equals = torch.eq(rounded_outputs.data, labels)
            row_equal_all = torch.all(row_equals, dim=1)
            num_equal_rows = torch.sum(row_equal_all).item()
            val_running_correct += num_equal_rows

            val_accuracy = 100. * val_running_correct/(batch_size*(i+1))

            # currently, print status update EVERY BATCH
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(val_loader)}], Val Loss: {val_loss/(batch_size*(i+1)):.4f}, Val Acc: {val_accuracy:.4f}")
            sys.stdout.flush()
torch.save(net.state_dict(), f"../../models/norCatGlau_{epochs}ep.pth")

net.eval()
print("-------------TRAINING DONE --> TESTING")
test_loss = 0
test_running_correct = 0
# goes through each batch
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
        row_equal_all = torch.all(row_equals, dim=1)
        num_equal_rows = torch.sum(row_equal_all).item()
        test_running_correct += num_equal_rows

        test_accuracy = 100. * test_running_correct/(batch_size*(i+1))

        print(f"Batch [{i+1}/{len(test_loader)}], Test Loss: {test_loss/(batch_size*(i+1)):.4f}, Test Acc: {test_accuracy:.4f}")
        sys.stdout.flush()