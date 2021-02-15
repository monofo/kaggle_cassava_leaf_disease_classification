import torch
import torch

import torch.nn as nn
import torch.nn.functional as F

import timm
from timm.models.layers.adaptive_avgmax_pool import SelectAdaptivePool2d

sigmoid = torch.nn.Sigmoid()
class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
swish = Swish.apply

class Swish_module(nn.Module):
    def forward(self, x):
        return swish(x)

swish_layer = Swish_module()


class timmNet(nn.Module):
    def __init__(self, net_type='resnext50_32x4d', pretrained=False, n_class=5):
        super().__init__()
        self.model = timm.create_model(net_type, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.fc = nn.Linear(n_features, n_class)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        output = self.fc(features)
        return features, output


class CassavaNet(nn.Module):
    def __init__(self, net_type="net_type", pretrained=False, n_class=5, bn=True):
        super().__init__()
        self.net_type = net_type
        if net_type == 'deit_base_patch16_224':
            self.model = torch.hub.load('facebookresearch/deit:main', net_type, pretrained=pretrained)
        else:
            self.model = timm.create_model(net_type, pretrained=pretrained)
        if 'efficientnet' in net_type:
            self.n_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
        elif net_type == 'vit_large_patch16_384' or net_type == 'deit_base_patch16_224':
            self.n_features = self.model.head.in_features
            self.model.head = nn.Identity()
        elif 'res' in net_type:
            self.n_features = self.model.fc.in_features
            self.model.fc = nn.Identity()

        elif "hrnet" in net_type:
            self.n_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()

        # 
        if bn:
            if self.net_type != "resnet18":
                self.fc = nn.Sequential(
                    nn.Linear(self.n_features, 512),
                    nn.BatchNorm1d(512),
                    nn.Dropout(p=0.6),
                    Swish_module(),
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.Dropout(p=0.6),
                    Swish_module(),
                    nn.Linear(256, 5)
                )
            else:
                self.fc = nn.Sequential(
                    nn.Linear(self.n_features, 256),
                    nn.BatchNorm1d(256),
                    nn.Dropout(p=0.8),
                    Swish_module(),
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.Dropout(p=0.6),
                    Swish_module(),
                    nn.Linear(128, 5)
                )

        else:
            self.fc = nn.Linear(self.n_features, 5)
        
    def forward(self, x):
        feature = self.model(x)
        # out = self.fc(feature)
        out = self.fc(feature)
        return feature, out
    
    def freeze(self):
        # To freeze the residual layers
        for param in self.model.parameters():
            param.requires_grad = False
            
        if 'efficientnet' in self.net_type:
            for param in self.fc.parameters():
                param.requires_grad = True
        elif self.net_type == 'vit_large_patch16_384' or 'deit_base_patch16_224':
            for param in self.fc.parameters():
                param.requires_grad = True
        elif 'resnext' in self.net_type:
            for param in self.fc.parameters():
                param.requires_grad = True
            
    def unfreeze(self):
        # Unfreeze all layers
        for param in self.model.parameters():
            param.requires_grad = True

    def freeze_batchnorm_stats(self):
        try:
            for m in self.model.modules():
                if isinstance(m,nn.BatchNorm2d) or isinstance(m,nn.LayerNorm):
                    m.eval()
        except ValuError:
            print('error with batchnorm2d or layernorm')
            return




class timmHrNet(nn.Module):
    def __init__(self, net_type, pretrained=False, n_class=5):
        super().__init__()
        self.model = timm.create_model(net_type, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, n_class)
        
    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        output = self.fc(features)
        return features, output

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.fc.parameters():
            param.requires_grad = True

        
    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True


class pretrained_timmEffNet(nn.Module):
    def __init__(self, net_type, pretrained_path=None, n_class=5):
        super().__init__()
        self.model = timmEffNet(net_type=net_type, pretrained=False, n_class=2)
        n_features = self.model.model.classifier.in_features
        self.model.load_state_dict(torch.load(pretrained_path)["model_state_dict"])
        self.model.model.classifier = nn.Linear(n_features, 5)
        
    def forward(self, x):
        x = self.model(x)
        return x


MODEL_LIST = {
    "resnet": CassavaNet,
    "effcientnet": CassavaNet,
    "vit": CassavaNet,
    "hrnet": CassavaNet,
    "pretrained_enet": pretrained_timmEffNet
}