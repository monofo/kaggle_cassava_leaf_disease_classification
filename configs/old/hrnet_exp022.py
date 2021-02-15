import os
import sys
import numpy as np
import cv2

import albumentations as A
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.transforms import transforms

sys.path.append("../")
from utils.randaug import RandAugment
from utils.loss import LabelSmoothingLoss, FocalCosineLoss, ClassificationFocalLossWithLabelSmoothing, TaylorCrossEntropyLoss
from utils.temperd_loss import TemperedLoss

amp=True
use_prev_data = True
upsampling=False


dir="hrnet_exp022"
net_type = "hrnet_w32"

resume=False
resume_dir = None

preprocessing = "rgb"
cutmix_ratio = 0.5
fmix_ratio = 0.
smix_ratio = 0
warmup_factor=10

num_workers=4
batch_size=32
n_epochs=10
lr= 1e-4/3
img_size=(512, 512)

beta = 0.999
samples_per_cls = [1492, 2189, 3017, 13158, 2890]
effective_num = 1.0 - np.power(beta, samples_per_cls)
weights = (1.0 - beta) / np.array(effective_num)
weights = weights / np.sum(weights) * 5

# criterion = ClassificationFocalLossWithLabelSmoothing(alpha=[0.2, 0.2, 0.2, 0.2, 0.2], gamma=1, n_classes=5).cuda()
# criterion = LabelSmoothingLoss(classes=5, smoothing=0.2)
criterion = TaylorCrossEntropyLoss(n=2, smoothing=0.2)
p=0.5
train_transforms = A.Compose(
        [
            A.Resize(height=img_size[0], width=img_size[1]),
            A.OneOf([A.RandomBrightness(limit=0.1, p=1), A.RandomContrast(limit=0.1, p=1)]),
            A.OneOf([A.MotionBlur(blur_limit=3), A.MedianBlur(blur_limit=3), A.GaussianBlur(blur_limit=3),], p=0.5,),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=0.2,
                rotate_limit=20,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101,
                p=1,
            ),
            A.Cutout(num_holes=6, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
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
