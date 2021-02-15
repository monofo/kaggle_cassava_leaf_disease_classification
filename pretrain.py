import argparse
import glob
import importlib
import os
import random
import sys

import cv2
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader

from dataset import KaggleDataset, PretrainedDataset
from utils.trainer import PyTorchTrainer, BinaryPyTorchTrainer

from model_factory import MODEL_LIST
from catalyst.data.sampler import BalanceClassSampler



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
    wandb.run.name = args.config
    config = importlib.import_module(f"{args.config}")
    os.makedirs(f"./result/{args.config}", exist_ok=True)

    df = pd.read_csv("./data/mendeley_leaf_data.csv")

    train_df, valid_df = train_test_split(
        df, test_size = 0.1, random_state=seed,stratify=df.label.values
    )


    train_dataset = PretrainedDataset(
        df=train_df,
        transforms=config.train_transforms,
        mode="train"
    )

    validation_dataset = PretrainedDataset(
        df=valid_df,
        transforms=config.valid_transforms,
        mode="val"
    )

    train_loader = DataLoader(
        train_dataset,
        # sampler=BalanceClassSampler(labels=train_dataset.get_labels(), mode="downsampling"),
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

    if "efficientnet" in config.net_type:
        net = MODEL_LIST["effcientnet"](net_type=config.net_type, pretrained=True, n_class=2)
    elif "vit" in config.net_type:
        net = MODEL_LIST["vit"](net_type=config.net_type, pretrained=True, n_class=2)
    elif "resnext" in config.net_type:
        net = MODEL_LIST["resnet"](net_type=config.net_type, pretrained=True, n_class=2)
        
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net, device_ids=[0,1,2,3]) 
        config.lr = config.lr * torch.cuda.device_count()

    net = net.to(device)


    wandb.watch(net, log="all")

    runner = PyTorchTrainer(model=net, device=device, config=config)
    if config.resume:
        print("load model")
        runner.load(f"./result/{config.dir}/{args.config}_fold_{fn}/{config.dir}_fold_{fn}_last-checkpoint.bin")
    runner.fit(train_loader=train_loader, validation_loader=valid_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        required=True,
    )
    parser.add_argument(
        "--gpus",
        "-g",
        type=str,
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main(args)
