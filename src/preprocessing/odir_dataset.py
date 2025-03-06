import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
import cv2

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


def goThruDir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def claheTransform(img_path):
    """
        Performs RGB CLAHE (feature augmentation/preprocessing)
        Params: image path
        Returns: tensor of transformed image
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)

    clahe = cv2.createCLAHE(clipLimit=10,tileGridSize=(8,8))

    # 0 to 'L' channel, 1 to 'a' channel, and 2 to 'b' channel
    img[:,:,0] = clahe.apply(img[:,:,0])

    img = cv2.cvtColor(img, cv2.COLOR_Lab2RGB)
    
    return Image.fromarray(img) #PIL

class OdirDataset(Dataset):
    def __init__(self, root_dir, data_folder, labels_csv, dataset_crops=None, transform=None, target_transform=None):
        super(OdirDataset, self).__init__()
        self.root_dir = root_dir
        self.data_folder = data_folder

        self.labels_csv = pd.read_csv(os.path.join(self.root_dir, labels_csv))
        self.dataset_crops = dataset_crops    #label column to filter by
        if self.dataset_crops is not None:   #dataset_crop has a value (eg. "cataract")
            # Gets ALL normal images & ALL dataset_crop images
            #self.labels_csv = self.labels_csv[(self.labels_csv["normal"] == 1) | (self.labels_csv[self.dataset_crop] == 1)]

            condition = self.labels_csv["normal"] == 1
            for crop_column in self.dataset_crops:
                condition |= self.labels_csv[crop_column] == 1

            self.labels_csv = self.labels_csv[condition]

            crop_columns_str = ", ".join(self.dataset_crops)
            print(f"Labels gathered: Normal, {crop_columns_str}")
        else:
            print("Labels gathered: ALL")

        self.transform = transform
        self.target_transform = target_transform # used to transform the label
    
    def __len__(self):
        return len(self.labels_csv)
    
    def getIds(self):
        # testing/debugging func
        return self.labels_csv.iloc[:, 0]
    
    def getLabels(self):
        # testing/debugging func
        return self.labels_csv

    def __getitem__(self, idx, crop=True):
        """ 
            Params:
                idx: row # to look for in labels_csv
            Shows:
                Nothing
                (used to visualize image using plt)
            Returns:
                image (img converted to a tensor), label (tensor: [oneHotLabelFromCsv])
        """
        
        row = self.labels_csv.iloc[idx]
        
        img_path = os.path.join(self.root_dir, self.data_folder, f"{row['ID']}")
        
        image = claheTransform(img_path)

        if self.dataset_crops is not None:
            label_values = [int(row["normal"])]
            for crop_column in self.dataset_crops:
                label_values.append(int(row[crop_column]))

            label = torch.tensor(label_values)
        else:
            label = torch.tensor([int(i) for i in row["normal":"others"]])

        if crop: # crop black pixels
            image = np.array(image)

            mask = image > 0

            coordinates = np.argwhere(mask)

            x0, y0, s0 = coordinates.min(axis=0)
            x1, y1, s1 = coordinates.max(axis=0) + 1

            image = image[x0:x1, y0:y1]

        if self.transform:
            new_image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return new_image, label