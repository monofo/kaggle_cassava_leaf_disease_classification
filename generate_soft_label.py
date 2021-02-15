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
from torch._C import default_generator
from torch.functional import istft
from torch.serialization import default_restore_location
from torch.utils.data import DataLoader
from scipy.special import softmax

import wandb
from dataset import KaggleDataset
from models.model_factory import MODEL_LIST
from utils.trainer import PyTorchTrainer
from tqdm.auto import tqdm

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
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True

seed_everything(seed)
device = torch.device('cuda')


def main(args):

    config = importlib.import_module(f"soft_label.{args.config}")
    PATH = config.PATH

    if config.use_prev_data:
        df = pd.read_csv("./data/merged.csv")
        df_20 = df.loc[df["source"] == 2020]
        df_20["data_dir"] = "train_images"
        df_19_0 = df.loc[(df["source"] == 2019) & (df["label"] == 0)]
        df_19_0["data_dir"] = "train/cbb/"
        df_19_2 = df.loc[(df["source"] == 2019) & (df["label"] == 2)]
        df_19_2["data_dir"] = "train/cgm/"
        df_19_4 = df.loc[(df["source"] == 2019) & (df["label"] == 4)]
        df_19_4["data_dir"] = "train/healthy/"

        df = pd.concat([df_20, df_19_0, df_19_2, df_19_4], axis=0).reset_index(drop=True)
    else:
        df = pd.read_csv("./data/train.csv")


    if "efficientnet" in config.net_type:
        net = MODEL_LIST["effcientnet"](net_type=config.net_type, pretrained=True)
    elif "vit" in config.net_type:
        net = MODEL_LIST["vit"](net_type=config.net_type, pretrained=True)
    elif "res" in config.net_type:
        net = MODEL_LIST["resnet"](net_type=config.net_type, pretrained=True)
    elif "hrnet" in config.net_type:
        net = net = MODEL_LIST["hrnet"](net_type=config.net_type, pretrained=True)

    net = net.to(device)
    
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df["label"].values
    skf = StratifiedKFold(n_splits=5)
    train_data_cp = []
    for (fold_num), (train_index, val_index) in enumerate(skf.split(X=df, y=y)):
        df.loc[df.iloc[val_index].index, "kfold"] = fold_num

        train_df = df.loc[df["kfold"] != fold_num].reset_index(drop=True).copy()
        valid_df = df.loc[df["kfold"] == fold_num].reset_index(drop=True).copy()

        val_data_cp = valid_df[["image_id", "label"]].copy()
        val_data_cp["CBB"]=0.25
        val_data_cp["CBSD"]=0.25
        val_data_cp["CGM"]=0.25
        val_data_cp["CMD"]=0.25
        val_data_cp["Healthy"]=0.25
        validation_dataset = KaggleDataset(
            df=valid_df,
            transforms=config.valid_transforms,
            mode="val",

        )


        valid_loader = DataLoader(
            validation_dataset,
            batch_size=config.batch_size,
            pin_memory=True,
            num_workers=4,
        )


        submission = []
        net.load_state_dict(torch.load(PATH[fold_num])["model_state_dict"])
        net.to("cuda")
        net.eval()

        for i in range(1):
            val_preds = []
            labels = []
            with torch.no_grad():
                for image, label in tqdm(valid_loader):
                    _, logit = net(image.to("cuda"))
                    val_preds.append(logit)
                    labels.append(label)

                labels = torch.cat(labels)
                val_preds = torch.cat(val_preds)
                submission.append(val_preds.cpu().numpy())

        submission_ensembled = 0
        for sub in submission:
            submission_ensembled += softmax(sub, axis=1) / len(submission)
        val_data_cp.iloc[:, 2:] = submission_ensembled
        train_data_cp.append(val_data_cp)

    soft_labels = df[["image_id"]].merge(pd.concat(train_data_cp), how="left", on="image_id")
    soft_labels.to_csv(f"./good_result/{args.config}/soft_labels.csv", index=False)


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
