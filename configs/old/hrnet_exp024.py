import sys
import numpy as np
import cv2

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


sys.path.append("../")
from utils.randaug import RandAugment

dir="hrnet_exp024"
net_type = "hrnet_w32"

resume=False
resume_dir = None

amp=True
use_prev_data=True
upsampling=False

num_workers=4
batch_size=32
n_epochs=10
img_size=(512, 512)


# criterion_name = "TemperedLoss"
# criterion_params = {
#     "t1": 0.8,
#     "t2": 1.4,
#     "smoothing": 0.2,
# }

criterion_name = "TaylorCrossEntropyLoss"
criterion_params = {
    "n": 2,
    "smoothing": 0.25,
}

optimizer_name = "RAdam"
optimizer_params = {
    "lr": 1e-4/3,
    "weight_decay": 1e-6,
    "opt_eps": 1e-8,
    "lookahead": True
}

scheduler_name = "GradualWarmupSchedulerV2"
scheduler_params = {
    "warmup_factor": 7
}

FREEZE=True
START_FREEZE=8

FIRST_FREEZE=True
END_FREEZE=2

######## data processings
preprocessing = "rgb"
cutmix_ratio = 0.5
fmix_ratio = 0.
smix_ratio = 0
p=0.5

train_transforms = A.Compose(
        [
            A.Resize(height=img_size[0], width=img_size[1]),
            A.OneOf([A.RandomBrightness(limit=0.1, p=1), A.RandomContrast(limit=0.1, p=1)]),
            A.OneOf([A.MotionBlur(blur_limit=3), A.MedianBlur(blur_limit=3), A.GaussianBlur(blur_limit=3),]),
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
