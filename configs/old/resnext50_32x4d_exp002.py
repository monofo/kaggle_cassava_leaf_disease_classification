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
from utils.loss import LabelSmoothing, ClassBalancedLoss

use_prev_data = True

fold_nums = [0]
fold_num=0



dir="resnext50_32x4d_exp002"
net_type = "resnext50_32x4d"


resume_dir=None
resume=False


preprocessing = "rgb"
cutmix_ratio = 0
fmix_ratio = 0.

num_workers = 4
batch_size = 64
n_epochs = 15
lr = 1e-4
img_size = (512, 512)

# beta=2.0
# gamma=0.9999
# samples_per_cls = [1/1553.0, 1/3632.0, 1/3159.0, 1/15816.0, 1/2893.0]
# effective_num = 1.0 - np.power(beta,samples_per_cls)
# weights = (1.0 - beta) / np.array(effective_num)
# weights = weights / np.sum(weights) * 5

# criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights).cuda().float())
criterion = torch.nn.CrossEntropyLoss()

train_transforms = A.Compose([
            A.RandomResizedCrop(img_size[0], img_size[1]),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=0.2, 
                sat_shift_limit=0.2, 
                val_shift_limit=0.2, 
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1,0.1), 
                contrast_limit=(-0.1, 0.1), 
                p=0.5
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
            A.CoarseDropout(p=0.5),
            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.7),
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
