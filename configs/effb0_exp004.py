import os
import sys
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch

sys.path.append("../")

fold_nums = [0]
fold_num=0


dir="effb0_exp004"
net_type = "efficientnet_b0"

cutmix_ratio = 0.3
fmix_ratio = 0.3

num_workers = 4
batch_size = 32
n_epochs = 15
lr = 1e-4
img_size = 256
criterion = torch.nn.CrossEntropyLoss()

train_transforms = A.Compose([
            A.Transpose(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
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
