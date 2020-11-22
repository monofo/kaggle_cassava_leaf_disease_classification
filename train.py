import argparse
import glob
import importlib
import os
import random
import sys

import cv2
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader

from dataset import KaggleDataset
from utils.trainer import PyTorchTrainer
# from models.model_factory import MODEL_LIST
import timm
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
    # "resnet": timmNet,
    "effcientnet": timmEffNet,
}
sys.path.append("configs")

seed = 42
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deteministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(seed)
device = torch.device('cuda')

def main(args):
    
    wandb.init(project="kaggle_cassava_leaf_disease_classification")
    config = importlib.import_module(f"{args.config}")
    os.makedirs(f"./result/{args.config}", exist_ok=True)

    df = pd.read_csv("./data/train.csv")
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df["label"].values
    skf = StratifiedKFold(n_splits=5)
    for (fold_num), (train_index, val_index) in enumerate(skf.split(X=df, y=y)):
        df.loc[df.iloc[val_index].index, "kfold"] = fold_num

    for fn in config.fold_nums:
        config.fold_num = fn
        train_dataset = KaggleDataset(
            data_dir="train_images",
            df=df.loc[df["kfold"] != config.fold_num],
            transforms=config.train_transforms,
            mode="train"
        )

        validation_dataset = KaggleDataset(
            data_dir="train_images",
            df=df.loc[df["kfold"] == config.fold_num],
            transforms=config.valid_transforms,
            mode="val"
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            pin_memory=True,
            num_workers=4,
        )

        valid_loader = DataLoader(
            validation_dataset,
            batch_size=config.batch_size,
            pin_memory=True,
            num_workers=4,
        )

        if config.net_type.startswith("efficientnet"):
            net = MODEL_LIST["effcientnet"](net_type=config.net_type, pretrained=True)
            
        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net, device_ids=[0,1,2,3]) 
            config.lr = config.lr * torch.cuda.device_count()

        net = net.to(device)
        wandb.watch(net, log="all")

        runner = PyTorchTrainer(model=net, device=device, config=config)
        runner.fit(train_loader=train_loader, validation_loader=valid_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        required=True,
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    main(args)
