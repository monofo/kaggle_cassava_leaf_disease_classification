import argparse
import numpy as np
import importlib
import os
import random
import sys
import pickle

import pandas as pd
import torch
from catalyst.data.sampler import BalanceClassSampler
from sklearn.model_selection import StratifiedKFold
from torch.functional import istft
from torch.utils.data import DataLoader

import wandb
from dataset import KaggleDataset, ImbalancedDatasetSampler, BalancedBatchSampler
from models.model_factory import MODEL_LIST
from sklearn import svm
from sklearn.metrics import accuracy_score

sys.path.append("configs")

seed = 2021
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

seed_everything(seed)
device = torch.device('cuda')

# chs = [
#     "result/upload_result/effb0_exp028/effb0_exp028_fold_0/effb0_exp028_fold_0_best-checkpoint-008epoch.bin",
#     "result/upload_result/effb0_exp028/effb0_exp028_fold_1/effb0_exp028_fold_1_best-checkpoint-010epoch.bin",
#     "result/upload_result/effb0_exp028/effb0_exp028_fold_2/effb0_exp028_fold_2_best-checkpoint-006epoch.bin",
#     "result/upload_result/effb0_exp028/effb0_exp028_fold_3/effb0_exp028_fold_3_best-checkpoint-009epoch.bin",
#     "result/upload_result/effb0_exp028/effb0_exp028_fold_4/effb0_exp028_fold_4_best-checkpoint-004epoch.bin",
# ]

chs = [
    "result/effb4_exp032/effb4_exp032_fold_0/effb4_exp032_fold_0_best-checkpoint-011epoch.bin",
    "result/effb4_exp032/effb4_exp032_fold_1/effb4_exp032_fold_1_best-checkpoint-011epoch.bin",
    "result/effb4_exp032/effb4_exp032_fold_2/effb4_exp032_fold_2_best-checkpoint-011epoch.bin",
    "result/effb4_exp032/effb4_exp032_fold_3/effb4_exp032_fold_3_best-checkpoint-011epoch.bin",
    "result/effb4_exp032/effb4_exp032_fold_4/effb4_exp032_fold_4_best-checkpoint-011epoch.bin",
]

def predict(net, train_loader, test_loader):
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    for step, (images, targets) in enumerate(train_loader):
        images = images.to(device).float()
        # targets = targets.to(device).float()

        with torch.no_grad():
            features, _ = net(images)

        x_train.extend(features.detach().cpu().numpy())
        y_train.extend(targets.argmax(1).detach().numpy())


    for step, (images, targets) in enumerate(test_loader):
        images = images.to(device).float()
        # targets = targets.to(device).float()

        with torch.no_grad():
            features, _ = net(images)

        x_test.extend(features.detach().cpu().numpy())
        y_test.extend(targets.argmax(1).detach().numpy())

    return x_train, y_train, x_test, y_test

def main(args):
    config = importlib.import_module(f"stage1.{args.config}")
    os.makedirs(f"./result/{args.config}", exist_ok=True)
    config.fold_num = args.fold_num
    print(config.fold_num)


    invalid_ids =  ['274726002.jpg',
                    '9224019.jpg',
                    '159654644.jpg',
                    '199112616.jpg',
                    '226533928.jpg',
                    '262902341.jpg',
                    '269713568.jpg',
                    '384390206.jpg',
                    '390601409.jpg',
                    '421035788.jpg',
                    '457405364.jpg',
                    '600736721.jpg',
                    '580111608.jpg',
                    '616718743.jpg',
                    '695438825.jpg',
                    '723564013.jpg',
                    '826231979.jpg',
                    '847847826.jpg',
                    '927165736.jpg',
                    '1004389140.jpg',
                    '1008244905.jpg',
                    '1338159402.jpg',
                    '1339403533.jpg',
                    '1366430957.jpg',
                    '9224019.jpg',
                    '4269208386.jpg',
                    '4239074071.jpg',
                    '3810809174.jpg',
                    '3652033201.jpg',
                    '3609350672.jpg',
                    '3609986814.jpg',
                    '3477169212.jpg',
                    '3435954655.jpg',
                    '3425850136.jpg',
                    '3251960666.jpg',
                    '3252232501.jpg',
                    '3199643560.jpg',
                    '3126296051.jpg',
                    '3040241097.jpg',
                    '2981404650.jpg',
                    '2925605732.jpg',
                    '2839068946.jpg',
                    '2698282165.jpg',
                    '2604713994.jpg',
                    '2415837573.jpg',
                    '2382642453.jpg',
                    '2321669192.jpg',
                    '2320471703.jpg',
                    '2278166989.jpg',
                    '2276509518.jpg',
                    '2262263316.jpg',
                    '2182500020.jpg',
                    '2139839273.jpg',
                    '2084868828.jpg',
                    '1848686439.jpg',
                    '1689510013.jpg',
                    '1359893940.jpg']

    if config.use_prev_data:
        df = pd.read_csv("./data/merged.csv")
        df = df[~df.image_id.isin(invalid_ids)]  

        # df_20 = df.loc[(df["source"] == 2020)].copy().reset_index(drop=True)
        # df_20["data_dir"] = "train_images"

        df_20 = df.loc[(df["source"] == 2020) & (df["label"] != 3)].copy().reset_index(drop=True)
        df_20["data_dir"] = "train_images"
        df_20_3 = df.loc[(df["source"] == 2020) & (df["label"] == 3)].copy().reset_index(drop=True)

        df_20_3 = df_20_3.sample(frac=0.7)
        df_20_3["data_dir"] = "train_images"


        df_19_0 = df.loc[(df["source"] == 2019) & (df["label"] == 0)].copy().reset_index(drop=True)
        df_19_0["data_dir"] = "train/cbb/"

        df_19_2 = df.loc[(df["source"] == 2019) & (df["label"] == 2)].copy().reset_index(drop=True)
        df_19_2["data_dir"] = "train/cgm/"

        df_19_4 = df.loc[(df["source"] == 2019) & (df["label"] == 4)].copy().reset_index(drop=True)
        df_19_4["data_dir"] = "train/healthy/"

        df = pd.concat([df_20, df_20_3, df_19_0, df_19_2, df_19_4], axis=0).reset_index(drop=True)
        # df = pd.concat([df_20, df_19_0, df_19_2, df_19_4], axis=0).reset_index(drop=True)
    else:
        df = pd.read_csv("./data/train.csv")
        # df = df[~df.image_id.isin(invalid_ids)]  

    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df["label"].values
    skf = StratifiedKFold(n_splits=5)
    
    for (fold_num), (train_index, val_index) in enumerate(skf.split(X=df, y=y)):
        df.loc[df.iloc[val_index].index, "kfold"] = fold_num

    train_df = df.loc[df["kfold"] != args.fold_num].reset_index(drop=True).copy()
    valid_df = df.loc[df["kfold"] == args.fold_num].reset_index(drop=True).copy()

    if config.upsampling:
        df_0 = train_df.loc[train_df["label"] == 0].sample(frac=2, replace=True).copy()
        df_1 = train_df.loc[train_df["label"] == 1].sample(frac=1.5, replace=True).copy()
        df_2 = train_df.loc[train_df["label"] == 2].sample(frac=1.5, replace=True).copy()
        df_3 = train_df.loc[train_df["label"] == 3].sample(frac=0.8, replace=False).copy()
        df_4 = train_df.loc[train_df["label"] == 4].sample(frac=1.5, replace=True).copy()

        train_df = pd.concat([df_0, df_1, df_2, df_4, df_3], axis=0).reset_index(drop=True)

    print("finish data setting")
    print(train_df.head())
    print(valid_df.head())

    train_dataset = KaggleDataset(
        df=train_df,
        transforms=config.train_transforms,
        preprocessing=config.preprocessing,
        mode="train",
    )

    validation_dataset = KaggleDataset(
        df=valid_df,
        transforms=config.valid_transforms,
        preprocessing=config.preprocessing,
        mode="val",

    )

    train_loader = DataLoader(
        train_dataset,
        # sampler=BalanceClassSampler(labels=train_dataset.get_labels(), mode="upsampling"),
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

    print("model setting")

    if "efficientnet" in config.net_type:
        net = MODEL_LIST["effcientnet"](net_type=config.net_type, pretrained=True)
    elif "vit" in config.net_type:
        net = MODEL_LIST["vit"](net_type=config.net_type, pretrained=True)
    elif "res" in config.net_type:
        net = MODEL_LIST["resnet"](net_type=config.net_type, pretrained=True)
    elif "hrnet" in config.net_type:
        net = net = MODEL_LIST["hrnet"](net_type=config.net_type, pretrained=True)

    ch = torch.load(chs[args.fold_num])
    net.load_state_dict(ch["model_state_dict"], strict=True)
    net = net.to(device)

    x_train, y_train, x_test, y_test  = predict(net, train_loader, valid_loader)
    svc = svm.LinearSVC(C=0.1, verbose=1, max_iter=100000, loss='squared_hinge', penalty='l2', dual=True )
    svc.fit(x_train, y_train)

    y_pred = svc.predict(x_test)

    print(accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        required=True,
    )

    parser.add_argument(
        "--fold_num",
        "-fn",
        type=int,
    )
    parser.add_argument(
        "--gpus",
        "-g",
        type=str,
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main(args)


