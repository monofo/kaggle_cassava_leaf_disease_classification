import os
import sys
import numpy as np
import cv2

import albumentations as A
import torch
from albumentations.pytorch.transforms import ToTensorV2



amp=True
use_prev_data=True
upsampling=False
preprocessing = "rgb"

batch_size = 32

dir="resnext101_32x8d_exp001"
net_type = "resnext101_32x8d"

img_size = [512, 512]
valid_transforms = A.Compose([
            A.Resize(img_size[0], img_size[1], p=1.0),
            A.Normalize(mean=[0.4309, 0.4968, 0.3135], std=[0.2131, 0.2179, 0.1940]),
            ToTensorV2(p=1.0),
        ], p=1.0)

PATH = [
    "good_result/resnext101_32x8d_exp001/resnext101_32x8d_exp001_fold_0/resnext101_32x8d_exp001_fold_0_best-checkpoint-009epoch.bin",
    "good_result/resnext101_32x8d_exp001/resnext101_32x8d_exp001_fold_1/resnext101_32x8d_exp001_fold_1_best-checkpoint-009epoch.bin",
    "good_result/resnext101_32x8d_exp001/resnext101_32x8d_exp001_fold_2/resnext101_32x8d_exp001_fold_2_best-checkpoint-009epoch.bin",
    "good_result/resnext101_32x8d_exp001/resnext101_32x8d_exp001_fold_3/resnext101_32x8d_exp001_fold_3_best-checkpoint-009epoch.bin",
    "good_result/resnext101_32x8d_exp001/resnext101_32x8d_exp001_fold_4/resnext101_32x8d_exp001_fold_4_best-checkpoint-009epoch.bin",
]