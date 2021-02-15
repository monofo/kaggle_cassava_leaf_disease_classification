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
drop3=False

fold_num=2

dir="hrnet_exp008"
net_type = "hrnet_w32"


resume=False
resume_dir = None

preprocessing = "rgb"
cutmix_ratio = 0.
fmix_ratio = 0.
smix_ratio = 0

num_workers=4
batch_size=32
n_epochs=15
lr=1e-4

img_size=(512, 512)

criterion = LabelSmoothingLoss(smoothing=0.2, classes=5)
p=0.5
train_transforms = A.Compose(
        [
            # A.Resize(image_size, image_size),
            A.RandomResizedCrop(img_size[0], img_size[1]),
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.4, rotate_limit=45, p=p),
            A.Cutout(p=p),
            A.RandomRotate90(p=p),
            A.Flip(p=p),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50
                    ),
                ],
                p=p,
            ),
            A.OneOf(
                [
                    A.IAAAdditiveGaussianNoise(),
                    A.GaussNoise(),
                ],
                p=p,
            ),
            A.CoarseDropout(max_holes=10, p=p),
            A.OneOf(
                [
                    A.MotionBlur(p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ],
                p=p,
            ),
            A.OneOf(
                [
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=0.1),
                    A.IAAPiecewiseAffine(p=0.3),
                ],
                p=p,
            ),
            A.Normalize(),
            ToTensorV2(p=1.0),
        ]
    )


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
