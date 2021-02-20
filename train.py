import argparse
import numpy as np
import importlib
import os
import random
import sys

import pandas as pd
import torch
from catalyst.data.sampler import BalanceClassSampler
from sklearn.model_selection import StratifiedKFold
from torch.functional import istft
from torch.utils.data import DataLoader

import wandb
from dataset import KaggleDataset, ImbalancedDatasetSampler, BalancedBatchSampler
from models.model_factory import MODEL_LIST
from utils.trainer import PyTorchTrainer

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

  


def main(args):

    wandb.init(project="kaggle_cassava_leaf_disease_classification")
    wandb.run.name = args.config
    config = importlib.import_module(f"stage1.{args.config}")
    wandb.save(f"configs/{args.config}.py")
    os.makedirs(f"./result/{args.config}", exist_ok=True)
    config.fold_num = args.fold_num
    print(config.fold_num)

    if config.use_prev_data:
        df = pd.read_csv("./data/split_df.csv")
        # # df = df_[~df_.image_id.isin(invalid_ids)].copy()

        # df_20 = df.loc[(df["source"] == 2020)].copy().reset_index(drop=True)
        # df_20["data_dir"] = "train_images"

        df_20 = df.loc[(df["source"] == 2020)].copy().reset_index(drop=True)
        df_20["data_dir"] = "train_images"

        df_19_0 = df.loc[(df["source"] == 2019) & (df["label"] == 0)].copy().reset_index(drop=True)
        df_19_0["data_dir"] = "train/cbb/"

        df_19_2 = df.loc[(df["source"] == 2019) & (df["label"] == 2)].copy().reset_index(drop=True)
        df_19_2["data_dir"] = "train/cgm/"

        df_19_4 = df.loc[(df["source"] == 2019) & (df["label"] == 4)].copy().reset_index(drop=True)
        df_19_4["data_dir"] = "train/healthy/"

        df = pd.concat([df_20, df_19_0, df_19_2, df_19_4], axis=0).reset_index(drop=True)
        # df = pd.concat([df_20, df_19_0, df_19_2, df_19_4], axis=0).reset_index(drop=True)
    else:
        df = pd.read_csv("./data/train.csv")


        # df = df_[~df_.image_id.isin(invalid_ids)].copy()

    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df["label"].values
    skf = StratifiedKFold(n_splits=5)
    
    for (fold_num), (train_index, val_index) in enumerate(skf.split(X=df, y=y)):
        df.loc[df.iloc[val_index].index, "kfold"] = fold_num

    train_df = df.loc[df["kfold"] != args.fold_num].reset_index(drop=True).copy()
    valid_df = df.loc[df["kfold"] == args.fold_num].reset_index(drop=True).copy()

    sampler = None
    if config.upsampling:
        target = train_df.label
        class_sample_count = np.unique(target, return_counts=True)[1]


        class_sample_count[0] *= 1
        class_sample_count[1] *= 1
        class_sample_count[2] *= 1
        class_sample_count[3] *= 0.7
        class_sample_count[4] *= 1

        weight = 1. / class_sample_count
        samples_weight = weight[target]
        samples_weight = torch.from_numpy(samples_weight)

        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

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
        sampler=sampler,
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

    if config.resume_dir is None:
        if "efficientnet" in config.net_type:
            net = MODEL_LIST["effcientnet"](net_type=config.net_type, pretrained=True, bn=config.bn)
        elif "vit" in config.net_type:
            net = MODEL_LIST["vit"](net_type=config.net_type, pretrained=True)
        elif "res" in config.net_type:
            net = MODEL_LIST["resnet"](net_type=config.net_type, pretrained=True)
        elif "hrnet" in config.net_type:
            net = net = MODEL_LIST["hrnet"](net_type=config.net_type, pretrained=True)
    else:
        net = MODEL_LIST["pretrained_enet"](net_type=config.net_type, pretrained_path=f"./result/{config.resume_dir}/{config.resume_dir}_fold_{config.fold_num}/{config.resume_dir}_fold_{config.fold_num}_last-checkpoint.bin")
    

    # if torch.cuda.device_count() > 1:
    #     net = torch.nn.DataParallel(net, device_ids=[0,1,2,3]) 
    #     config.lr = config.lr * torch.cuda.device_count()

    net = net.to(device)

    wandb.watch(net, log="all")

    runner = PyTorchTrainer(model=net, device=device, config=config, fold_num=args.fold_num)
    if config.resume:
        print("load model")

        runner.load(f"./result/{config.dir}/{config.dir}_fold_{config.fold_num}/{config.dir}_fold_{config.fold_num}_last-checkpoint.bin")

    runner.fit(train_loader=train_loader, validation_loader=valid_loader)

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
