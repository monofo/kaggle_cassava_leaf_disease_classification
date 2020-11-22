
from glob  import glob
import os
import time
import torch.nn as nn
import warnings
from datetime import datetime
import numpy as np
import wandb
import torch
from sklearn import metrics

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler

import sys
sys.path.append("utils")
from meters import AverageMeter
from mixs import cutmix, fmix

class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]




def set_wandb(config, c):
    wandb.config.exp_id = config.dir
    wandb.config.batch_size = config.batch_size
    wandb.config.num_workers = config.num_workers
    wandb.config.n_epochs = config.n_epochs
    wandb.config.img_size = config.img_size
    wandb.config.lr = config.lr
    wandb.config.cutmix = config.cutmix_ratio
    wandb.config.fmix = config.fmix_ratio
   

    wandb.config.criterion = config.criterion.__class__.__name__
    wandb.config.scheduler = c.scheduler.__class__.__name__
    wandb.config.optimizer = c.optimizer.__class__.__name__


warnings.filterwarnings("ignore")

class PyTorchTrainer:
    
    def __init__(self, model, device, config):
        self.config = config
        self.epoch = 0
        
        self.base_dir = f'./result/{config.dir}/{config.fold_num}'
        os.makedirs(self.base_dir, exist_ok=True)
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10**5

        self.model = model
        self.device = device
        self.wandb = True

        self.cutmix = self.config.cutmix_ratio
        self.fmix = self.config.fmix_ratio

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ] 

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, )

        self.scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, config.n_epochs - 1)
        self.scheduler = GradualWarmupSchedulerV2(self.optimizer, multiplier=10, total_epoch=1, after_scheduler=self.scheduler_cosine)


        self.criterion = config.criterion
        self.log(f'Fitter prepared. Device is {self.device}')
        set_wandb(self.config, self)

    def fit(self, train_loader, validation_loader):
        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')
                wandb.log({"Epoch": self.epoch, "lr": lr }, step=e)
            if self.config.step_scheduler:
                self.scheduler.step(e)

            t = time.time()
            summary_loss, summary_scores = self.train_one_epoch(train_loader)
            torch.cuda.empty_cache()
            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, Fold Num: {self.config.fold_num}, summary_loss: {summary_loss.avg:.5f},  time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')
            wandb.log({ 
                f"Train_loss_{self.config.fold_num}": summary_loss.avg,
                f"Train_ACC_{self.config.fold_num}": summary_scores[0].avg,
                f"Train_F1_{self.config.fold_num}": summary_scores[0].avg,
                }, step=e)

            t = time.time()
            summary_loss, summary_scores, example_images = self.validation(validation_loader)
            torch.cuda.empty_cache()
            self.log(f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f},  time: {(time.time() - t):.5f}')
            wandb.log({
                f"Val_loss_{self.config.fold_num}": summary_loss.avg,
                f"Val_ACC_{self.config.fold_num}": summary_scores[0].avg,
                f"Val_F1_{self.config.fold_num}": summary_scores[0].avg,
                f"Example_{self.config.fold_num}": example_images
                }, step=e)

            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                    os.remove(path)

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)

            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        summary_acc = AverageMeter()
        summary_f1 = AverageMeter()

        t = time.time()
        example_images = []
        for step, (images, targets) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                targets = targets.to(self.device).long()
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                summary_loss.update(loss.detach().item(), batch_size)
                summary_acc.update((outputs.argmax(1)==targets).sum().item()/batch_size, batch_size)
                summary_f1.update(metrics.f1_score(targets.detach().cpu().numpy(), outputs.argmax(1).detach().cpu().numpy(), average="macro"), batch_size)
                example_images.append(wandb.Image(
                    images[0], caption=f"Pred: {outputs.argmax(1)[0].detach().cpu().item()} Truth: {targets[0].detach().cpu().item()}"
                ))

        return summary_loss, (summary_acc, summary_f1), example_images

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        summary_acc = AverageMeter()
        summary_f1 = AverageMeter()
        t = time.time()
        for step, (images, targets) in enumerate(train_loader):
            choice = np.random.rand(1)

            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            targets = targets.to(self.device).long()
            images = images.to(self.device).float()
            batch_size = images.shape[0]
            
            if choice <= self.cutmix:
                images, targets = cutmix(images, targets, 1.)
            elif choice <= self.cutmix + self.fmix:
                images, targets = fmix(images, targets, alpha=1., decay_power=3., shape=(self.config.img_size,self.config.img_size))

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            summary_loss.update(loss.detach().item(), batch_size)
            summary_acc.update((outputs.argmax(1)==targets).sum().item()/batch_size,batch_size)
            summary_f1.update(metrics.f1_score(targets.detach().cpu().numpy(), outputs.argmax(1).detach().cpu().numpy(), average="macro"), batch_size)

            self.optimizer.step()


        return summary_loss, (summary_acc, summary_f1)

    def predict(self, test_loader, sub):
        self.model.eval()
        all_outputs = torch.tensor([], device=self.device)
        for step, (images, fnames) in enumerate(test_loader):
            with torch.no_grad():
                images = images.to(self.device).float()
                outputs = self.model.forward(images)
                all_outputs = torch.cat((all_outputs, outputs), 0)

        sub.iloc[:, 1] = all_outputs.detach().cpu().numpy()
        return sub

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

        wandb.save(path.split("/")[-1])

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1
        
    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')