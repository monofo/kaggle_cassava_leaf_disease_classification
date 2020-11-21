import os
import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset


DATA_DIR = "./data"

class KaggleDataset(Dataset):
    def __init__(self, data_dir, df, transforms=None, mode="train"):
        self.data_dir = data_dir
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.mode=mode


    def __getitem__(self, idx):
        image = os.path.join(DATA_DIR, self.data_dir, self.df.loc[idx, "image_id"])
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.df.loc[idx, "label"]
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']

        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return self.df.shape[0]


