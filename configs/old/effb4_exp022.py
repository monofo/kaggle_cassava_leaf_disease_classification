import os
import sys
import numpy as np
import cv2

import albumentations as A
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torch._C import ThroughputBenchmark
from torchvision.transforms import transforms

sys.path.append("../")
from utils.randaug import RandAugment
from utils.loss import LabelSmoothingLoss, FocalCosineLoss, ClassificationFocalLossWithLabelSmoothing, SVMLabelSmoothingLoss, TaylorCrossEntropyLoss
from utils.temperd_loss import TemperedLoss

amp=True
use_prev_data = True
upsampling=False

FREEZE = False #If you fine tune after START_FREEZE epochs
START_FREEZE = 9


FIRST_FREEZE = True
END_FREEZE = 3


dir="eff4_exp022"
net_type = "tf_efficientnet_b4_ns"


resume=False
resume_dir = None


preprocessing = "rgb"
cutmix_ratio = 0.5
fmix_ratio = 0.
smix_ratio = 0


num_workers=4
batch_size=32
n_epochs=10
lr= 1e-4
warmup_factor=5
img_size=(512, 512)


beta = 0.9999
samples_per_cls = [1492, 2189, 3017, 13158, 2890]
effective_num = 1.0 - np.power(beta, samples_per_cls)
weights = (1.0 - beta) / np.array(effective_num)
weights = list(weights / np.sum(weights) * 5)

# criterion = LabelSmoothingLoss(smoothing=0.2, classes=5, weights=None)
criterion = TaylorCrossEntropyLoss(n=2, smoothing=0.2)

p=0.5

train_transforms = A.Compose(
        [
            A.Resize(height=img_size[0], width=img_size[1]),
            # RandAugment(N, M),
            A.OneOf([A.RandomBrightness(limit=0.1, p=1), A.RandomContrast(limit=0.1, p=1)], p=0.5),
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
# -------------4------

# --------------------
step_scheduler = True  # do scheduler.step after optimizer.step
validation_scheduler = False  # do scheduler.step after validation stage loss
