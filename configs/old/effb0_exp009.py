import os
import sys

import albumentations as A
import torch
from albumentations.pytorch.transforms import ToTensorV2
from RandAugment import RandAugment
from torchvision.transforms import transforms

sys.path.append("../")
from utils.randaug import RandAugment


fold_nums = [0]
fold_num=0


dir="effb0_exp009"
net_type = "efficientnet_b0"

cutmix_ratio = 0.5
fmix_ratio = 0.

num_workers = 4
batch_size = 32
n_epochs = 20
lr = 5e-4
img_size = 512
criterion = torch.nn.CrossEntropyLoss()

N = 10
M = 5

train_transforms = A.Compose([
            A.RandomRotate90(),
            A.Flip(),
            A.Transpose(),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=.1),
                A.IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            A.OneOf([
                # A.CLAHE(clip_limit=2),
                A.IAASharpen(),
                A.IAAEmboss(),
                A.RandomBrightnessContrast(),            
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
            A.Resize(img_size, img_size),
            A.Normalize(),
            ToTensorV2(p=1.0),
        ], p=1.0)

valid_transforms = A.Compose([
            A.Resize(height=img_size, width=img_size, p=1.0),
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
