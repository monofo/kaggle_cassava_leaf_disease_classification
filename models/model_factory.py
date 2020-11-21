import torch
import torch

import torch.nn as nn
import torch.nn.functional as F

import timm

class timmNet(nn.Module):
    def __init__(self, net_type='resnext50_32x4d', pretrained=False, n_class=5):
        super().__init__()
        self.model = timm.create_model(net_type, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x


class timmEffNet(nn.Module):
    def __init__(self, net_type, pretrained=False, n_class=5):
        super().__init__()
        self.model = timm.create_model(net_type, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)
        
    def forward(self, x):
        x = self.model(x)
        return x


MODEL_LIST ={
    "resnet": timmNet,
    "effcientnet": timmEffNet,
}