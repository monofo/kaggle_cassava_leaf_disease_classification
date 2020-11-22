import os
import random
import sys

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import OrderedDict


seed = 42
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deteministic = True
    torch.backends.cudnn.benchmark = True

def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict

seed_everything(seed)
device = torch.device('cuda')

class config():
    fold_num=0
    dir="effb3_exp001"
    net_type = "efficientnet_b3"
    num_workers = 4
    batch_size = 32
    n_epochs = 20
    lr = 1e-4
    img_size = 512
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

TEST_PATH = "./data/test_images"
DATA_DIR = "./data/"

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['image_id'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f'{TEST_PATH}/{file_name}'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image


class timmEffNet(nn.Module):
    def __init__(self, net_type, pretrained=False, n_class=5):
        super().__init__()
        self.model = timm.create_model(net_type, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)
        
    def forward(self, x):
        x = self.model(x)
        return x


def inference(model, states, test_loader, device):
    model.to(device)
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    probs = []
    for i, (images) in tk0:
        images = images.to(device)
        avg_preds = []
        for state in states:
            model.load_state_dict(fix_model_state_dict(state['model_state_dict']))
            model.eval()
            with torch.no_grad():
                y_preds = model(images)
            avg_preds.append(y_preds.softmax(1).to('cpu').numpy())
        avg_preds = np.mean(avg_preds, axis=0)
        probs.append(avg_preds)
    probs = np.concatenate(probs)
    return probs



# main()

test = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
test_dataset = TestDataset(test, transform=config.valid_transforms)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,    
                        num_workers=2, pin_memory=True)

if config.net_type.startswith("efficientnet"):
    net = timmEffNet(net_type=config.net_type, pretrained=False)
    
state_dict = torch.load(os.path.join("./result/effb3_exp001", "best-checkpoint-014epoch.bin"))

predictions = inference(net, [state_dict], test_loader, device)
# submission
test['label'] = predictions.argmax(1)
test[['image_id', 'label']].to_csv('submission.csv', index=False)
test.head()
