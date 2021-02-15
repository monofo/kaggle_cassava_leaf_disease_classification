import os
import sys

import albumentations as A
import torch
from albumentations.pytorch.transforms import ToTensorV2
from RandAugment import RandAugment
from torchvision.transforms import transforms

sys.path.append("../")
from utils.randaug import RandAugment
from utils.loss import LabelSmoothing

use_prev_data = False

fold_nums = [0]
fold_num=0


dir="resnext101_32x8d_exp001"
net_type = "resnext101_32x8d"

resume=False

cutmix_ratio = 0
fmix_ratio = 0.5

num_workers = 4
batch_size = 16
n_epochs = 15
lr = 1e-4
img_size = (600, 800)
criterion = LabelSmoothing()

N = 5
M = 5

train_transforms = A.Compose([
            RandAugment(N, M),
            A.Transpose(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            A.Resize(img_size[0], img_size[1], p=1.0),
            A.Normalize(),
            ToTensorV2(p=1.0),
        ], p=1.0)


valid_transforms = A.Compose([
            A.Resize(img_size[0], img_size[1], p=1.0),
            A.Normalize(),
            ToTensorV2(p=1.0),
        ], p=1.0)

# -------------------
verbose = True
verbose_step = 1
# -------------------

# --------------------
step_scheduler = True  # do scheduler.step after optimizer.step
validation_scheduler = False  # do scheduler.step after validation stage loss
