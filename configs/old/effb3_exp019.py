import os
import sys
import numpy as np

import albumentations as A
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.transforms import transforms

sys.path.append("../")
from utils.randaug import RandAugment
from utils.loss import LabelSmoothingLoss
from utils.temperd_loss import TemperedLoss

amp=True
use_prev_data = True
upsampling=True


dir="effb3_exp019"
net_type = "tf_efficientnet_b3_ns"

resume=False
resume_dir = None

preprocessing = "rgb"
cutmix_ratio = 0.
fmix_ratio = 0.
smix_ratio = 0

num_workers=4
batch_size=32
n_epochs=20
lr= 1e-4
img_size=(512, 512)

criterion = LabelSmoothingLoss(smoothing=0.2, classes=5)
p=0.5
train_transforms = A.Compose(
        [
            # A.Resize(img_size[0], img_size[1], p=1.0),
            A.RandomResizedCrop(img_size[0], img_size[1], p=1.0),
            A.Transpose(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            A.Cutout(num_holes=16, max_h_size=64, max_w_size=64, fill_value=0, p=0.7),
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
# -------------------

# --------------------
step_scheduler = True  # do scheduler.step after optimizer.step
validation_scheduler = False  # do scheduler.step after validation stage loss
