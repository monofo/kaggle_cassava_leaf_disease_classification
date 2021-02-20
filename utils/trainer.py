
import os
import sys
import time
import warnings
from datetime import datetime
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from losses.loss import get_criterion
from optimizers.optimizer import get_optimizer
from torch._C import device
from torch.cuda.amp import GradScaler, autocast
from torch.optim.swa_utils import SWALR, AveragedModel
from wandb.sklearn import plot_confusion_matrix

from utils.early_stopping import EarlyStopping
from utils.meters import AverageMeter
from utils.mixs import cutmix, fmix, mix_criterion, snapmix


def set_wandb(config, fold_num):
    wandb.config.exp_id = config.dir
    wandb.config.fold_num = fold_num
    wandb.config.batch_size = config.batch_size
    wandb.fold_num = config.fold_num
    wandb.config.num_workers = config.num_workers
    wandb.config.n_epochs = config.n_epochs
    wandb.config.img_size = config.img_size
    wandb.config.lr = config.optimizer_params["lr"]
    wandb.config.cutmix = config.cutmix_ratio
    wandb.config.fmix = config.fmix_ratio
   

    wandb.config.criterion = config.criterion_name
    wandb.config.scheduler = config.optimizer_name
    wandb.config.optimizer = config.scheduler_name

warnings.filterwarnings("ignore")

class PyTorchTrainer:
    def __init__(self, model, device, config, fold_num):
        self.config = config
        self.epoch = 0
        self.start_epoch = 0
        self.fold_num = fold_num
        if self.config.stage2:
            self.base_dir = f'./result/stage2/{config.dir}/{config.dir}_fold_{config.fold_num}'
        else:
            self.base_dir = f'./result/{config.dir}/{config.dir}_fold_{config.fold_num}'
        os.makedirs(self.base_dir, exist_ok=True)
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10**5

        self.model = model
        self.swa_model = AveragedModel(self.model)
        self.device = device
        self.wandb = True

        self.cutmix = self.config.cutmix_ratio
        self.fmix = self.config.fmix_ratio
        self.smix = self.config.smix_ratio

        self.es = EarlyStopping(patience=8)


        self.scaler = GradScaler()  
        self.amp = self.config.amp
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ] 


        self.optimizer, self.scheduler = get_optimizer(self.model, self.config.optimizer_name, self.config.optimizer_params, 
                                                    self.config.scheduler_name, self.config.scheduler_params, self.config.n_epochs)

        self.criterion = get_criterion(self.config.criterion_name, self.config.criterion_params)
        self.log(f'Fitter prepared. Device is {self.device}')
        set_wandb(self.config, fold_num)

    def fit(self, train_loader, validation_loader):
        if self.config.FIRST_FREEZE:
            self.model.freeze()

        for e in range(self.start_epoch, self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')
                wandb.log({"Epoch": self.epoch, "lr": lr }, step=e)

            if self.config.step_scheduler:
                self.scheduler.step(e)


            if e >= self.config.START_FREEZE and self.config.FREEZE:
                 print('Model Frozen -> Train Classifier Only')
                 self.model.freeze()
                 self.config.FREEZE = False

            if e >= self.config.END_FREEZE and self.config.FIRST_FREEZE:
                 print('Model UnFrozen -> Train Classifier Only')
                 self.model.unfreeze()
                 self.config.FIRST_FREEZE = False


            t = time.time()
            summary_loss, summary_scores, example_images = self.train_one_epoch(train_loader)
            torch.cuda.empty_cache()
            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, Fold Num: {self.fold_num}, summary_loss: {summary_loss.avg:.5f}, summary_acc: {summary_scores.avg},  time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/{self.config.dir}_fold_{self.fold_num}_last-checkpoint.bin')
            wandb.log({ 
                f"Train_loss": summary_loss.avg,
                f"Train_ACC": summary_scores.avg,
                f"Example_{self.config.fold_num}": example_images
                }, step=e)

            t = time.time()
            summary_loss, summary_scores = self.validation(validation_loader)
            torch.cuda.empty_cache()
            self.log(f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, summary_acc: {summary_scores.avg},  time: {(time.time() - t):.5f}')


            # if summary_loss.avg < self.best_summary_loss:
            self.best_summary_loss = summary_loss.avg
            self.model.eval()
            self.save(f'{self.base_dir}/{self.config.dir}_fold_{self.config.fold_num}_best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                # for path in sorted(glob(f'{self.base_dir}/{self.config.dir}_fold_{self.config.fold_num}_best-checkpoint-*epoch.bin'))[:-3]:
                #     os.remove(path)

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)

            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        summary_acc = AverageMeter()

        t = time.time()
        
        y_true = []
        y_pred = []
        for step, (images, targets) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )


            with torch.no_grad():
                targets = targets.to(self.device).float()
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                _, outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                # targets = targets.argmax(1)
                y_true.extend(targets.argmax(1).detach().cpu().numpy())
                y_pred.extend(outputs.argmax(1).detach().cpu().numpy())
                summary_loss.update(loss.detach().item(), batch_size)
                summary_acc.update((outputs.argmax(1)==targets.argmax(1)).sum().item()/batch_size, batch_size)



        wandb.log({
            f"Val_loss": summary_loss.avg,
            f"Val_ACC": summary_acc.avg,
            }, step=self.epoch)


        if self.es.step(torch.tensor(summary_loss.avg)):
            self.log("Stop Early Stopiing")
            plot_confusion_matrix(y_true, y_pred)
            exit(0)


        if self.epoch == self.config.n_epochs - 1:
            plot_confusion_matrix(y_true, y_pred)
        return summary_loss, summary_acc

    def train_one_epoch(self, train_loader):
        self.model.train()
        if self.epoch < self.config.freeze_bn_epoch:
            self.model.freeze_batchnorm_stats()
            
        summary_loss = AverageMeter()
        summary_acc = AverageMeter()

        example_images = []

        t = time.time()
        for step, (images, targets) in enumerate(train_loader):
            choice = np.random.rand(1)
            self.optimizer.zero_grad()
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            targets = targets.to(self.device).float()
            images = images.to(self.device).float()
            batch_size = images.shape[0]
            if self.config.FIRST_FREEZE and self.config.END_FREEZE > self.epoch:
                if self.amp:
                    with autocast():
                        _, outputs = self.model(images)
                        loss = self.criterion(outputs, targets)

                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1000)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            else:
                if self.amp:
                    with autocast():
                        if choice < self.cutmix:
                            aug_images, aug_targets = cutmix(images, targets, 1.)
                            _, outputs = self.model(aug_images)
                            loss = mix_criterion(outputs, aug_targets, self.criterion)
                        elif choice < self.cutmix + self.fmix:
                            aug_images, aug_targets = fmix(images, targets, alpha=1., decay_power=3., shape=self.config.img_size, device=device)
                            aug_images = aug_images.to(self.device).float()
                            _, outputs = self.model(aug_images)
                            loss = mix_criterion(outputs, aug_targets, self.criterion)
                        elif choice < self.cutmix + self.fmix+self.smix:
                            X, ya, yb, lam_a, lam_b = snapmix(images, targets, alpha=0.5, model=self.model)
                            _, outputs, _ = self.model(X)
                            loss = self.snapmix_criterion(self.criterion, outputs, ya, yb, lam_a, lam_b)
                        else:
                            _, outputs = self.model(images)
                            loss = self.criterion(outputs, targets)


                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1000)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                else:
                    if choice < self.cutmix:
                        aug_images, aug_targets = cutmix(images, targets, 1.)
                        _, outputs = self.model(aug_images)
                        loss = mix_criterion(outputs, aug_targets, self.criterion)
                    elif choice < self.cutmix + self.fmix:
                        aug_images, aug_targets = fmix(images, targets, alpha=1., decay_power=3., shape=self.config.img_size, device=device)
                        aug_images = aug_images.to(self.device).float()
                        _, outputs = self.model(aug_images)
                        loss = mix_criterion(outputs, aug_targets, self.criterion)
                    elif choice < self.cutmix + self.fmix+self.smix:
                        X, ya, yb, lam_a, lam_b = snapmix(images, targets, alpha=0.5, model=self.model)
                        _, outputs, _ = self.model(X)
                        loss = self.snapmix_criterion(self.criterion, outputs, ya, yb, lam_a, lam_b)
                    else:
                        _, outputs = self.model(images)
                        loss = self.criterion(outputs, targets)

                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1000)
                    loss = self.criterion(outputs, targets )            
                    loss.backward()
                    self.optimizer.step()

            if len(example_images) < 16:
                example_images.append(wandb.Image(
                    images[0], # caption=f"Truth: {targets[0].argmax(1).detach().cpu().item()}"
                ))

            summary_loss.update(loss.detach().item(), batch_size)
            summary_acc.update((outputs.argmax(1)==targets.argmax(1)).sum().item()/batch_size,batch_size)

        return summary_loss, summary_acc, example_images

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
            # 'optimizer_state_dict': self.optimizer.state_dict(),
            # 'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

        wandb.save(path.split("/")[-1])

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1
        self.start_epoch = checkpoint['epoch'] + 1
        
    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')



class PyTorchTrainerStudent:
    def __init__(self, teacher_model, student_model,  device, config, fold_num):
        self.config = config
        self.epoch = 0
        self.start_epoch = 0
        self.fold_num = fold_num
        if self.config.stage2:
            self.base_dir = f'./result/stage2/{config.dir}/{config.dir}_fold_{config.fold_num}'
        else:
            self.base_dir = f'./result/{config.dir}/{config.dir}_fold_{config.fold_num}'
        os.makedirs(self.base_dir, exist_ok=True)
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10**5

        self.teacher_model = teacher_model
        self.teacher_mode.eval()

        self.student_model = student_model
        self.device = device
        self.wandb = True

        self.cutmix = self.config.cutmix_ratio
        self.fmix = self.config.fmix_ratio
        self.smix = self.config.smix_ratio

        self.es = EarlyStopping(patience=5)

        self.scaler = GradScaler()  
        self.amp = self.config.amp
        param_optimizer = list(self.student_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ] 


        self.optimizer, self.scheduler = get_optimizer(self.student_model, self.config.optimizer_name, self.config.optimizer_params, 
                                                    self.config.scheduler_name, self.config.scheduler_params, self.config.n_epochs)

        self.criterion = get_criterion(self.config.criterion_name, self.config.criterion_params)
        self.log(f'Fitter prepared. Device is {self.device}')
        set_wandb(self.config, fold_num)

    def fit(self, train_loader, validation_loader):
        if self.config.FIRST_FREEZE:
            self.student_model.freeze()

        for e in range(self.start_epoch, self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')
                wandb.log({"Epoch": self.epoch, "lr": lr }, step=e)
            
            # if self.config.step_scheduler:
            #     self.scheduler.step(e)


            t = time.time()
            summary_loss, summary_scores, example_images = self.train_one_epoch(train_loader)
            torch.cuda.empty_cache()
            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, Fold Num: {self.fold_num}, summary_loss: {summary_loss.avg:.5f}, summary_acc: {summary_scores.avg},  time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/{self.config.dir}_fold_{self.fold_num}_last-checkpoint.bin')
            wandb.log({ 
                f"Train_loss": summary_loss.avg,
                f"Train_ACC": summary_scores.avg,
                f"Example_{self.config.fold_num}": example_images
                }, step=e)

            t = time.time()
            summary_loss, summary_scores = self.validation(validation_loader)
            torch.cuda.empty_cache()
            self.log(f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, summary_acc: {summary_scores.avg},  time: {(time.time() - t):.5f}')


            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.student_model.eval()
                self.save(f'{self.base_dir}/{self.config.dir}_fold_{self.config.fold_num}_best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                # for path in sorted(glob(f'{self.base_dir}/{self.config.dir}_fold_{self.config.fold_num}_best-checkpoint-*epoch.bin'))[:-3]:
                #     os.remove(path)

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)

            self.epoch += 1

    def validation(self, val_loader):
        self.student_model.eval()
        summary_loss = AverageMeter()
        summary_acc = AverageMeter()

        t = time.time()
        
        y_true = []
        y_pred = []
        for step, (images, targets) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )


            with torch.no_grad():
                targets = targets.to(self.device).float()
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                _, ys_soft = self.student_model(images)
                _, yt_soft = self.teacher_model(images)
                loss = self.criterion(yt_soft, ys_soft,targets)

                # targets = targets.argmax(1)
                y_true.extend(targets.argmax(1).detach().cpu().numpy())
                y_pred.extend(ys_soft.argmax(1).detach().cpu().numpy())
                summary_loss.update(loss.detach().item(), batch_size)
                summary_acc.update((ys_soft.argmax(1)==targets.argmax(1)).sum().item()/batch_size, batch_size)



        wandb.log({
            f"Val_loss": summary_loss.avg,
            f"Val_ACC": summary_acc.avg,
            }, step=self.epoch)


        if self.es.step(torch.tensor(summary_loss.avg)):
            self.log("Stop Early Stopiing")
            plot_confusion_matrix(y_true, y_pred)
            exit(0)


        if self.epoch == self.config.n_epochs - 1:
            plot_confusion_matrix(y_true, y_pred)
        return summary_loss, summary_acc

    def train_one_epoch(self, train_loader):
        self.student_model.train()
            
        summary_loss = AverageMeter()
        summary_acc = AverageMeter()

        example_images = []

        t = time.time()
        for step, (images, targets) in enumerate(train_loader):
            if self.config.step_scheduler:
                self.scheduler.step(self.e + step / self.config.batch_size)
            choice = np.random.rand(1)
            self.optimizer.zero_grad()
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            targets = targets.to(self.device).float()
            images = images.to(self.device).float()
            batch_size = images.shape[0]


            if self.amp:
                with autocast():
                    if choice < self.cutmix:
                        aug_images, aug_targets = cutmix(images, targets, 1.)
                        _, ys_soft = self.teacher_model(aug_images)
                        _, yt_soft = self.teacher_model(aug_images)
                    else:
                        _, ys_soft = self.student_model(images)
                        _, yt_soft = self.teacher_model(images)


                    
                loss = self.criterion(yt_soft, ys_soft, targets)

                grad_norm = torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1000)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                if choice < self.cutmix:
                    aug_images, aug_targets = cutmix(images, targets, 1.)
                    _, outputs = self.student_model(aug_images)
                    loss = mix_criterion(outputs, aug_targets, self.criterion)
                else:
                    _, outputs = self.student_model(images)
                    loss = self.criterion(outputs, targets)

                grad_norm = torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1000)
                loss = self.criterion(outputs, targets )            
                loss.backward()
                self.optimizer.step()

            if len(example_images) < 16:
                example_images.append(wandb.Image(
                    images[0], # caption=f"Truth: {targets[0].argmax(1).detach().cpu().item()}"
                ))

            summary_loss.update(loss.detach().item(), batch_size)
            summary_acc.update((ys_soft.argmax(1)==targets.argmax(1)).sum().item()/batch_size,batch_size)

        return summary_loss, summary_acc, example_images

    def predict(self, test_loader, sub):
        self.student_model.eval()
        all_outputs = torch.tensor([], device=self.device)
        for step, (images, fnames) in enumerate(test_loader):
            with torch.no_grad():
                images = images.to(self.device).float()
                outputs = self.student_model.forward(images)
                all_outputs = torch.cat((all_outputs, outputs), 0)

        sub.iloc[:, 1] = all_outputs.detach().cpu().numpy()
        return sub

    def save(self, path):
        self.student_model.eval()
        torch.save({
            'model_state_dict': self.student_model.state_dict(),
            # 'optimizer_state_dict': self.optimizer.state_dict(),
            # 'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

        wandb.save(path.split("/")[-1])

    def load(self, path):
        checkpoint = torch.load(path)
        self.student_model.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1
        self.start_epoch = checkpoint['epoch'] + 1
        
    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')

