import torch
import torch

import torch.nn as nn
import torch.nn.functional as F

import timm

class timmNet(nn.Module):
    def __init__(self, net_type='resnext50_32x4d', pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, CFG.target_size)

    def forward(self, x):
        x = self.model(x)
        return x