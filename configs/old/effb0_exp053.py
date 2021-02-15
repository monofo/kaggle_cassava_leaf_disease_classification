import os
import sys
import numpy as np

import albumentations as A
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.transforms import transforms

sys.path.append("../")
from utils.randaug import RandAugment
from utils.loss import LabelSmoothing, FocalLoss
from utils.temperd_loss import TemperedLoss

use_prev_data = False

# over sampling
fold_nums = [0]
fold_num=0

dir="effb0_exp053"
net_type = "tf_efficientnet_b0_ns"

resume=False
resume_dir = None



preprocessing = "rgb"
cutmix_ratio = 0.
fmix_ratio = 0.

num_workers=4
batch_size=64
n_epochs=20
lr=1e-3
img_size=(512, 512)

criterion = TemperedLoss(t1=0.2, t2=1.0, smoothing=0.05, num_classes=5).cuda()

N = 5
M = 5
train_transforms = A.Compose([
            A.RandomResizedCrop(img_size[0], img_size[1]),
            A.Transpose(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            A.Resize(img_size[0], img_size[1]),
            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.7),
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
