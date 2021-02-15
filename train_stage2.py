import argparse
import numpy as np
import importlib
import os
import random
import sys
import time
import torch.nn.init as init
import pandas as pd
import torch.nn as nn
import torch
from catalyst.data.sampler import BalanceClassSampler
from sklearn.model_selection import StratifiedKFold
from torch.functional import istft
from torch.utils.data import DataLoader
import csv
import gc

import wandb
from dataset import KaggleDataset
from models.model_factory import MODEL_LIST
from losses.loss import CustomeLoss, CrossEntropyLossOneHot
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from losses.loss import get_criterion
from optimizers.optimizer import get_optimizer

sys.path.append("configs")
chs = [
    "result/upload_result/effb4_exp003/effb4_exp032_fold_0_best-checkpoint-012epoch.bin",
    "result/upload_result/effb4_exp003/effb4_exp032_fold_1_best-checkpoint-012epoch.bin",
    "result/upload_result/effb4_exp003/effb4_exp032_fold_2_best-checkpoint-011epoch.bin",
    "result/upload_result/effb4_exp003/effb4_exp032_fold_3_best-checkpoint-010epoch.bin",
    "result/upload_result/effb4_exp003/effb4_exp032_fold_4_best-checkpoint-011epoch.bin",
]

#########################args + config
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    "-c",
    required=True,
)

parser.add_argument(
    "--fold_num",
    "-fn",
    type=int,
)
parser.add_argument(
    "--gpus",
    "-g",
    type=str,
)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = importlib.import_module(f"stage2.{args.config}")

##################Utiltys
seed = 2021
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

seed_everything(seed)
device = torch.device('cuda')

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

last_time = time.time()
begin_time = last_time
TOTAL_BAR_LENGTH = 65.

start_prune=10
best_acc = 0
scaler = GradScaler() 

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

################################################################################# loss fn

class SoftTarget(nn.Module):
	'''
	Distilling the Knowledge in a Neural Network
	https://arxiv.org/pdf/1503.02531.pdf
	'''
	def __init__(self, T):
		super(SoftTarget, self).__init__()
		self.T = T

	def forward(self, out_s, out_t):
		loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
						F.softmax(out_t/self.T, dim=1),
						reduction='batchmean') * self.T * self.T

		return loss

#################################################################################### train and test
def train(epoch, trainloader, t_net, s_net, criterion, optimizer):
    print('\nEpoch: %d' % epoch)
    s_net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # targets = targets.argmax(1)
        optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            _, t_outputs = t_net(inputs)

        if 1:
            with autocast():
                
                _, s_outputs = s_net(inputs)
                Kd_loss = criterion(s_outputs, t_outputs.detach())
                cls_loss = CrossEntropyLossOneHot()(s_outputs, targets)
                
                loss = Kd_loss + cls_loss
            
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(s_net.parameters(), 1000)
                scaler.step(optimizer)
                scaler.update()
        else:
            _, s_outputs = s_net(inputs)
            Kd_loss = criterion(s_outputs, t_outputs.detach())

            grad_norm = torch.nn.utils.clip_grad_norm_(s_net.parameters(), 1000)
            loss.backward()
            optimizer.step()
           

             
        train_loss += loss.item()
        _, predicted = torch.max(s_outputs.data, 1)
        total += targets.argmax(1).size(0)
        correct += predicted.eq(targets.argmax(1).data).cpu().sum()
        correct = correct.item()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return (train_loss / batch_idx, 100. * correct / total)


def test(epoch, testloader, t_net, s_net, criterion):
    global best_acc
    s_net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # targets = targets.argmax(1)
            inputs, targets = inputs.to(device), targets.to(device)
            _, t_outputs = t_net(inputs)
            _, s_outputs = s_net(inputs)
            Kd_loss = criterion(s_outputs, t_outputs.detach())
            cls_loss = CrossEntropyLossOneHot()(s_outputs, targets)
                
            loss = Kd_loss + cls_loss
            test_loss += loss.item()
            _, predicted = torch.max(s_outputs.data, 1)
            total += targets.argmax(1).size(0)
            correct += predicted.eq(targets.argmax(1).data).cpu().sum()
            correct = correct.item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        checkpoint(acc, epoch, s_net)
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    state = {
        'current_net': s_net.state_dict(),
    }
    torch.save(state, f'./checkpoint/{config.dir}/current_net')
    return (test_loss / batch_idx, 100. * correct / total)


def checkpoint(acc, epoch, net):
    # Save checkpoint.
    print('Saving..')
    state = {
        'model_state_dict': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir(f'checkpoint/{config.dir}'):
        os.mkdir(f'checkpoint/{config.dir}')
    torch.save(state, f'./checkpoint/{config.dir}/best_acc_{args.fold_num}.pth')


def main():

    # wandb.init(project="kaggle_cassava_leaf_disease_classification")
    # wandb.run.name = args.config
    # wandb.save(f"configs/{args.config}.py")
    os.makedirs(f"./checkpoint/{config.dir}", exist_ok=True)
    config.fold_num = args.fold_num
    print(config.fold_num)

    invalid_ids =  ['274726002.jpg',
                    '9224019.jpg',
                    '159654644.jpg',
                    '199112616.jpg',
                    '226533928.jpg',
                    '262902341.jpg',
                    '269713568.jpg',
                    '384390206.jpg',
                    '390601409.jpg',
                    '421035788.jpg',
                    '457405364.jpg',
                    '600736721.jpg',
                    '580111608.jpg',
                    '616718743.jpg',
                    '695438825.jpg',
                    '723564013.jpg',
                    '826231979.jpg',
                    '847847826.jpg',
                    '927165736.jpg',
                    '1004389140.jpg',
                    '1008244905.jpg',
                    '1338159402.jpg',
                    '1339403533.jpg',
                    '1366430957.jpg',
                    '9224019.jpg',
                    '4269208386.jpg',
                    '4239074071.jpg',
                    '3810809174.jpg',
                    '3652033201.jpg',
                    '3609350672.jpg',
                    '3609986814.jpg',
                    '3477169212.jpg',
                    '3435954655.jpg',
                    '3425850136.jpg',
                    '3251960666.jpg',
                    '3252232501.jpg',
                    '3199643560.jpg',
                    '3126296051.jpg',
                    '3040241097.jpg',
                    '2981404650.jpg',
                    '2925605732.jpg',
                    '2839068946.jpg',
                    '2698282165.jpg',
                    '2604713994.jpg',
                    '2415837573.jpg',
                    '2382642453.jpg',
                    '2321669192.jpg',
                    '2320471703.jpg',
                    '2278166989.jpg',
                    '2276509518.jpg',
                    '2262263316.jpg',
                    '2182500020.jpg',
                    '2139839273.jpg',
                    '2084868828.jpg',
                    '1848686439.jpg',
                    '1689510013.jpg',
                    '1359893940.jpg']

    if config.use_prev_data:
        df = pd.read_csv("./data/merged.csv")
        df = df[~df.image_id.isin(invalid_ids)]    

        # df_20 = df.loc[(df["source"] == 2020)].copy().reset_index(drop=True)
        # df_20["data_dir"] = "train_images"

        df_20 = df.loc[(df["source"] == 2020) & (df["label"] != 3)].copy().reset_index(drop=True)
        df_20["data_dir"] = "train_images"
        df_20_3 = df.loc[(df["source"] == 2020) & (df["label"] == 3)].copy().reset_index(drop=True)

        df_20_3 = df_20_3.sample(frac=0.7)
        df_20_3["data_dir"] = "train_images"



        df_19_0 = df.loc[(df["source"] == 2019) & (df["label"] == 0)].copy().reset_index(drop=True)
        df_19_0["data_dir"] = "train/cbb/"

        df_19_2 = df.loc[(df["source"] == 2019) & (df["label"] == 2)].copy().reset_index(drop=True)
        df_19_2["data_dir"] = "train/cgm/"

        df_19_4 = df.loc[(df["source"] == 2019) & (df["label"] == 4)].copy().reset_index(drop=True)
        df_19_4["data_dir"] = "train/healthy/"

        df = pd.concat([df_20, df_20_3, df_19_0, df_19_2, df_19_4], axis=0).reset_index(drop=True).sample(frac=0.2)
        # df = pd.concat([df_20, df_19_0, df_19_2, df_19_4], axis=0).reset_index(drop=True)
    else:
        df = pd.read_csv("./data/train.csv")

    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df["label"].values
    skf = StratifiedKFold(n_splits=5)
    
    for (fold_num), (train_index, val_index) in enumerate(skf.split(X=df, y=y)):
        df.loc[df.iloc[val_index].index, "kfold"] = fold_num

    train_df = df.loc[df["kfold"] != args.fold_num].reset_index(drop=True).copy()
    valid_df = df.loc[df["kfold"] == args.fold_num].reset_index(drop=True).copy()


    print("finish data setting")
    print(train_df.head())
    print(valid_df.head())

    train_dataset = KaggleDataset(
        df=train_df,
        transforms=config.train_transforms,
        preprocessing=config.preprocessing,
        mode="train",
        ind=False,
    )

    validation_dataset = KaggleDataset(
        df=valid_df,
        transforms=config.valid_transforms,
        preprocessing=config.preprocessing,
        mode="val",
        ind=False,

    )

    train_loader = DataLoader(
        train_dataset,
        # sampler=BalanceClassSampler(labels=train_dataset.get_labels(), mode="upsampling"),
        batch_size=45,
        pin_memory=True,
        num_workers=4,
    )
    

    valid_loader = DataLoader(
        validation_dataset,
        batch_size=32,
        pin_memory=True,
        num_workers=4,
    )

    print("model setting")


    if "efficientnet" in config.net_type:
        t_net = MODEL_LIST["effcientnet"](net_type=config.net_type, pretrained=True)
    elif "vit" in config.net_type:
        t_net = MODEL_LIST["vit"](net_type=config.net_type, pretrained=True)
    elif "res" in config.net_type:
        t_net = MODEL_LIST["resnet"](net_type=config.net_type, pretrained=True)
    elif "hrnet" in config.net_type:
        t_net = net = MODEL_LIST["hrnet"](net_type=config.net_type, pretrained=True)


    if "efficientnet" in config.net_type:
        s_net = MODEL_LIST["effcientnet"](net_type="tf_efficientnet_b4_ns", pretrained=True, bn=False)

    ch = torch.load(chs[args.fold_num])
    t_net.load_state_dict(ch["model_state_dict"], strict=True)
    t_net = t_net.cuda()
    t_net.eval()

    optimizer, scheduler = get_optimizer(s_net, config.optimizer_name, config.optimizer_params, 
                                                config.scheduler_name, config.scheduler_params, config.n_epochs)

    criterion = SoftTarget(T=4.0).cuda()


    # wandb.watch(net, log="all")

    logname = f"checkpoint/{config.dir}/" + s_net.__class__.__name__ + \
            '_' + "stage2_" + f'{args.fold_num}.csv'
    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(
                ['epoch', 'train loss', 'train acc', 'test loss', 'test acc'])

    start_epoch=0

    if 0:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(f'checkpoint/{config.dir}'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(f"checkpoint/{config.dir}/best_acc_{args.fold_num}.pth", map_location='cpu')
        s_net.load_state_dict(checkpoint['model_state_dict'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        torch.set_rng_state(checkpoint['rng_state'])

    s_net = s_net.to(device)
    for epoch in range(start_epoch, config.n_epochs):
        print("lr: ", optimizer.param_groups[0]['lr'])
        train_loss, train_acc = train(epoch, train_loader, t_net, s_net, criterion, optimizer)
        test_loss, test_acc = test(epoch, valid_loader, t_net, s_net, criterion)

        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss, train_acc, test_loss, test_acc])
        scheduler.step()



if __name__ == "__main__":
    main()