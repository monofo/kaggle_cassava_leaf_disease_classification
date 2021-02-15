import sys
import numpy as np
import cv2

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from utils.randaug import RandAugment

"""
exp001と比較して軽めのaug
"""

stage2=False

dir="resnet_exp002"
net_type = "resnet18"


resume=False
resume_dir = None

amp=False

use_prev_data=False
down_sampling3=False
upsampling=False

num_workers=4
batch_size=128
n_epochs=70
img_size=(512, 512)

num_workers=4
batch_size=32
n_epochs=70
img_size=(512, 512)

criterion_name = "CrossEntropyLossOneHot"
criterion_params = {
    # "smoothing": 0.2,
}

optimizer_name = "adam"
optimizer_params = {
    "lr": 1e-4/3,
    "weight_decay": 1e-6,
    # "momentum": 0.9,
    # "opt_eps": 1e-8,
    "lookahead": False
}


scheduler_name = "WarmRestart"
scheduler_params = {
    "warmup_factor": 7,
    "T_max": 15,
    "T_mul": 1,
    "eta_min": 1e-6
}

# scheduler_name = "CosineAnnealingWarmRestarts"
# scheduler_params = {
#     "warmup_factor": 7,
#     "T_0": n_epochs-1,
#     "T_multi": 1,
#     "eta_min": 1e-6
# }

# }


############################################################################################## data processings
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
            A.CenterCrop(img_size[0], img_size[1], p=1),
            A.OneOf([A.RandomBrightness(limit=0.1, p=1), A.RandomContrast(limit=0.1, p=1)], p=p),
            A.OneOf([A.MotionBlur(blur_limit=3), A.MedianBlur(blur_limit=3), A.GaussianBlur(blur_limit=3),], p=p),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=0.2,
                rotate_limit=20,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101,
                p=.9,
            ),
            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=.9),
            A.Normalize(mean=[0.4309, 0.4968, 0.3135], std=[0.2131, 0.2179, 0.1940]),
            ToTensorV2(p=1.0),
        ]
    )


valid_transforms = A.Compose([
            A.CenterCrop(img_size[0], img_size[1], p=1),
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