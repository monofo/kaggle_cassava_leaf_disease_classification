import os
import sys
import numpy as np

import albumentations as A
import torch
from albumentations.pytorch.transforms import ToTensorV2
from RandAugment import RandAugment
from torchvision.transforms import transforms

sys.path.append("../")
from utils.randaug import RandAugment
from utils.loss import LabelSmoothing, FocalLoss
from utils.temperd_loss import TemperedLoss

use_prev_data = True

fold_nums = [0]
fold_num=0

# over sampling

dir="effb0_exp041"
net_type = "tf_efficientnet_b0_ns"


resume_dir=None
resume=False


preprocessing = "rgb"
cutmix_ratio = 0.4
fmix_ratio = 0.4

num_workers = 4
batch_size = 64
n_epochs = 20
lr = 1e-4
img_size = (512, 512)

criterion = TemperedLoss(t1=0.2, t2=1.0, smoothing=0.05, num_classes=5).cuda()
 
train_transforms = A.Compose([
            A.Transpose(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightness(limit=0.2, p=0.75),
            A.RandomContrast(limit=0.2, p=0.75),
            A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5),
                A.GaussNoise(var_limit=(5.0, 30.0)),
            ], p=0.7),

            A.OneOf([
                A.OpticalDistortion(distort_limit=1.0),
                A.GridDistortion(num_steps=5, distort_limit=1.),
                A.ElasticTransform(alpha=3),
            ], p=0.7),

            A.CLAHE(clip_limit=4.0, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            A.Resize(img_size[0], img_size[1], p=1.0),
            # A.CoarseDropout(p=0.5),
            # A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.7),
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
