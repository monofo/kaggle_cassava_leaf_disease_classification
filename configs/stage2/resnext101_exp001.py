import os
import sys
import numpy as np
import cv2

import albumentations as A
import torch
from albumentations.pytorch.transforms import ToTensorV2

from utils.randaug import RandAugment

stage2=True
amp=True
use_prev_data = True
upsampling=True


dir="resnext101_32x8d_exp001_stage2"
net_type = "resnext101_32x8d"

soft_label_filename="good_result/resnext101_32x8d_exp001/soft_labels.csv"

resume=False
resume_dir = None

amp=True
use_prev_data=True
upsampling=False

num_workers=4
batch_size=32
n_epochs=10
img_size=(512, 512)


criterion_name = "CrossEntropyLossOneHot"
criterion_params = {
    # "alpha": 1,
    # "gamma": 1,
    # "xent": .1,
    # "smoothing": 0.05,
}

optimizer_name = "adam"
optimizer_params = {
    "lr": 1e-4/3,
    "weight_decay": 1e-6,
    # "opt_eps": 1e-8,
    "lookahead": False
}

scheduler_name = "GradualWarmupSchedulerV2"
scheduler_params = {
    "warmup_factor": 7,
    "T_0": n_epochs-1,
    "T_multi": 1
}

freeze_bn_epoch=0

FREEZE=False
START_FREEZE=8

FIRST_FREEZE=False
END_FREEZE=2

######## data processings
preprocessing = "rgb"
cutmix_ratio = 0
fmix_ratio = 0.
smix_ratio = 0
p=0.5

train_transforms = A.Compose(
        [
            A.Resize(img_size[0], img_size[1], p=1.0),
            A.Transpose(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            A.Cutout(num_holes=12, max_h_size=32, max_w_size=32, fill_value=0, p=0.5),
            A.Normalize(mean=[0.4309, 0.4968, 0.3135], std=[0.2131, 0.2179, 0.1940]),
            ToTensorV2(p=1.0),
        ]
    )



valid_transforms = A.Compose([
            A.Resize(img_size[0], img_size[1], p=1.0),
            A.Normalize(mean=[0.4309, 0.4968, 0.3135], std=[0.2131, 0.2179, 0.1940]),
            ToTensorV2(p=1.0),
        ], p=1.0)

# -------------------
verbose = True
verbose_step = 1
# -------------4------

# --------------------
step_scheduler = True  # do scheduler.step after optimizer.step
validation_scheduler = False  # do scheduler.step after validation stage loss
