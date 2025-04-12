import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

class MushroomDataset(Dataset):
    def __init__(self, root_dir, data_folder, transform=None, target_transform=None):
        """
        Mushroom classification dataset.

        Args:
            root_dir (string): Directory with all the images.
            data_folder (string): Folder within root_dir ('train' or 'val')
            transform (callable, optional): Optional transform to be applied on an image.
            target_transform (callable, optional): Optional transform to be applied on the target.
        """
        super(MushroomDataset, self).__init__()
        self.root_dir = root_dir
        self.data_folder = data_folder
        self.transform = transform
        self.target_transform = target_transform

        # Define the classes and create class-to-index mapping
        self.classes = ['edible', 'fresh', 'non-fresh', 'poisonous']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Get all image paths and labels
        self.samples = self._make_dataset()

        print(f"Created MushroomDataset with {len(self.samples)} images in {data_folder}")

    def _make_dataset(self):
        """
        Create a list of (image_path, class_idx) tuples
        """
        samples = []
        data_path = os.path.join(self.root_dir, self.data_folder)

        for class_name in self.classes:
            class_path = os.path.join(data_path, class_name)
            if not os.path.isdir(class_path):
                continue

            class_idx = self.class_to_idx[class_name]
            for image_name in os.listdir(class_path):
                if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(class_path, image_name)
                    samples.append((image_path, class_idx))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get the image and label at index idx.

        Returns:
            image (tensor): Transformed image.
            label (tensor): One-hot encoded label.
        """
        image_path, class_idx = self.samples[idx]

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Create one-hot encoded label
        label = torch.zeros(len(self.classes))
        label[class_idx] = 1.0

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label