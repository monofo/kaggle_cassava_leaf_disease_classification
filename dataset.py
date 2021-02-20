import os
import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset


DATA_DIR = "./data"
label2idx = {"cgm": 2, "cbb": 0, "cbsd": 1, "cmd": 3, "healthy": 4}

def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec


class KaggleDataset(Dataset):
    def __init__(self, df, transforms=None, preprocessing="rgb", soft_label_filename=False, mode="train", ind=False):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.mode=mode
        
        self.ind = ind

        self.soft_labels=soft_label_filename
        if soft_label_filename:
            print("soflt label")


    def __getitem__(self, idx):
        data_dir = self.df.loc[idx, "data_dir"]
        image = os.path.join(DATA_DIR, data_dir, self.df.loc[idx, "image_id"])
        image = cv2.imread(image)

        if self.preprocessing == "gray":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.stack([image, image, image]).transpose(1,2,0)
        elif self.preprocessing == "canny":
            imgYUV = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            imgY = imgYUV[:,:,0]
            image = cv2.Canny(imgY, 100, 200)
            image = np.stack([image, image, image]).transpose(1,2,0)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # soft label
        if self.soft_labels:
            label = torch.tensor(
                onehot(5, self.df.loc[idx, "label"]) * 0.7
                    + (self.df.iloc[idx, 3:8].values * 0.3).astype(np.float)
            )
        else:
            label = onehot(5, self.df.loc[idx, "label"])

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']

        if not self.ind:
            return image, label
        else:
            return image, label, idx
            
    def __len__(self):
        return self.df.shape[0]

    def get_labels(self):
        return list(self.df.label.values)


class PretrainedDataset(Dataset):
    def __init__(self, df, transforms=None, mode="train"):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.mode=mode


    def __getitem__(self, idx):

        image = os.path.join(DATA_DIR, "mendeley_leaf_data", self.df.loc[idx, "image_id"])
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.df.loc[idx, "label"]
        
        # label = onehot(2, label)

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']

        return torch.tensor(image, dtype=torch.float), label

    def __len__(self):
        return self.df.shape[0]

    def get_labels(self):
        return list(self.df.label.values)



import torch
import torch.utils.data
import torchvision


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        if self.callback_get_label:
            return self.callback_get_label(dataset, idx)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels[idx].item()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return dataset.imgs[idx][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[idx][1]
        else:
            return dataset[idx][1]
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples




import torch
is_torchvision_installed = True
try:
    import torchvision
except:
    is_torchvision_installed = False
import torch.utils.data
import random

class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max
        
        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1]*len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1]*len(self.keys)
    
    def _get_label(self, dataset, idx, labels = None):
        if self.labels is not None:
            return self.labels[idx].item()
        else:
            # Trying guessing
            dataset_type = type(dataset)
            if is_torchvision_installed and dataset_type is torchvision.datasets.MNIST:
                return dataset.train_labels[idx].item()
            elif is_torchvision_installed and dataset_type is torchvision.datasets.ImageFolder:
                return dataset.imgs[idx][1]
            else:
                raise Exception("You should pass the tensor of labels to the constructor as second argument")

    def __len__(self):
        return self.balanced_max*len(self.keys)