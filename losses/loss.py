
from typing import List
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.temperd_loss import tempered_softmax, log_t


class CrossEntropyLossOneHot(nn.Module):
    def __init__(self):
        super(CrossEntropyLossOneHot, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, preds, labels):
        return torch.mean(torch.sum(-labels * self.log_softmax(preds), -1))


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing = 0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim = -1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
    
            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss()(inputs, targets)

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class FocalCosineLoss(nn.Module):
    def __init__(self, alpha=1, gamma=1, xent=.1, smoothing=0.05):
        super(FocalCosineLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

        self.xent = xent
        self.smoothing=smoothing
        self.lab_smooth = LabelSmoothing(smoothing=smoothing)

        self.y = torch.Tensor([1]).cuda()

    def forward(self, input, target, reduction="mean"):
        cosine_loss = F.cosine_embedding_loss(input, target, self.y, reduction=reduction)

        if self.smoothing == 0:
            cent_loss = F.cross_entropy(F.normalize(input), target.float(), reduce=False)
        else:
            cent_loss = self.lab_smooth(F.normalize(input), target)
        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * cent_loss

        if reduction == "mean":
            focal_loss = torch.mean(focal_loss)

        return cosine_loss + self.xent * focal_loss


class ClassificationFocalLossWithLabelSmoothing(nn.Module):
    def __init__(self, n_classes: int, alpha=[0.2, 0.2, 0.2, 0.2, 0.2], gamma=2, weights: List[float]=None):
        """
        :param alpha: parameter of Label Smoothing.
        :param n_classes:
        :param gamma: 簡単なサンプルの重み. 大きいほど簡単なサンプルを重視しない.
        :param weights: weights by classes,
        :param logits:
        """
        super().__init__()
        self._alpha = torch.tensor(alpha)
        self._noise_val = self._alpha / n_classes
        self._n_classes = n_classes
        self.gamma = gamma
        self.class_weight_tensor = torch.tensor(weights).view(-1, ).cuda() if weights else None


    def forward(self, pred: torch.Tensor, teacher: torch.Tensor) -> float:
        """
        :param pred: batch_size, n_classes
        :param teacher: batch_size,
        :return:
        """
        if teacher.ndim == 1:  # 1次元ならonehotの2次元tensorにする
            teacher = torch.eye(self._n_classes)[teacher]
        # Label smoothing.
        teacher = teacher * (1 - self._alpha) + self._noise_val
        teacher = teacher.cuda()

        ce_loss = F.binary_cross_entropy_with_logits(pred, teacher, reduce=False)
        pt = torch.exp(-ce_loss)

        if self.class_weight_tensor is not None:
            class_weight_tensor = self.class_weight_tensor.expand(pred.shape[0],
                                                                  self.class_weight_tensor.shape[0], )
            focal_loss = (1. - pt) ** self.gamma * (ce_loss * self.class_weight_tensor)
        else:
            focal_loss = (1. - pt) ** self.gamma * ce_loss

        return torch.mean(focal_loss)


class SymmetricCrossEntropy(nn.Module):

    def __init__(self, alpha=0.1, beta=1.0, num_classes=5):
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, logits, targets, reduction='mean'):
        onehot_targets = torch.eye(self.num_classes)[targets].cuda()
        ce_loss = F.cross_entropy(logits, targets, reduction=reduction)
        rce_loss = (-onehot_targets*logits.softmax(1).clamp(1e-7, 1.0).log()).sum(1)
        if reduction == 'mean':
            rce_loss = rce_loss.mean()
        elif reduction == 'sum':
            rce_loss = rce_loss.sum()
        return self.alpha * ce_loss + self.beta * rce_loss


class TemperedLoss(nn.Module):
    def __init__(self, t1, t2, smoothing=0.0, num_iter=5, num_classes=5):
        super(TemperedLoss, self).__init__()
        self.t1 = t1
        self.t2 = t2
        self.smoothing = smoothing
        self.num_iter = num_iter
        self.num_classes = num_classes

    def forward(self, input, targets):
      
        if self.smoothing > 0.0:
            targets = (1 - self.num_classes / (self.num_classes - 1) * self.smoothing) * targets + self.smoothing / (self.num_classes - 1)
 
        probabilities = tempered_softmax(input, self.t2, self.num_iter)
  
        temp1 = (log_t(targets + 1e-10, self.t1) - log_t(probabilities, self.t1)) * targets
        temp2 = (1 / (2 - self.t1)) * (torch.pow(targets, 2 - self.t1) - torch.pow(probabilities, 2 - self.t1))
        loss_values = temp1 - temp2

        # return torch.sum(loss_values, dim=-1)
        return torch.sum(loss_values)


class TaylorSoftmax(nn.Module):
    '''
    This is the autograd version
    '''
    def __init__(self, dim=1, n=2):
        super(TaylorSoftmax, self).__init__()
        assert n % 2 == 0
        self.dim = dim
        self.n = n

    def forward(self, x):
        '''
        usage similar to nn.Softmax:
            >>> mod = TaylorSoftmax(dim=1, n=4)
            >>> inten = torch.randn(1, 32, 64, 64)
            >>> out = mod(inten)
        '''
        fn = torch.ones_like(x)
        denor = 1.
        for i in range(1, self.n+1):
            denor *= i
            fn = fn + x.pow(i) / denor
        out = fn / fn.sum(dim=self.dim, keepdims=True)
        return out

    
class TaylorCrossEntropyLoss(nn.Module):
    def __init__(self, n=2, ignore_index=-1, reduction='mean', smoothing=0.05):
        super(TaylorCrossEntropyLoss, self).__init__()
        assert n % 2 == 0
        self.taylor_softmax = TaylorSoftmax(dim=1, n=n)
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.lab_smooth = LabelSmoothing(smoothing=smoothing)

    def forward(self, logits, labels):
        log_probs = self.taylor_softmax(logits).log()
        #loss = F.nll_loss(log_probs, labels, reduction=self.reduction,
        #        ignore_index=self.ignore_index)
        loss = self.lab_smooth(log_probs, labels)
        return loss


class CustomeLoss(nn.Module):
    def __init__(self, l=0.9, t=1.0):
        super(CustomeLoss, self).__init__()
        self.l = l
        self.t = t
        self.criterion = CrossEntropyLossOneHot().cuda()

    def forward(self, yt_soft, ys_soft, y_true):
        loss = (1-self.l) * self.criterion(yt_soft, ys_soft) + self.l * self.t * self.t * self.criterion(y_true, ys_soft)

        return loss


class TruncatedLoss(nn.Module):

    def __init__(self, q=0.7, k=0.5, trainset_size=50000):
        super(TruncatedLoss, self).__init__()
        self.q = q
        self.k = k
        self.weight = torch.nn.Parameter(data=torch.ones(trainset_size, 1), requires_grad=False)
             
    def forward(self, logits, targets, indexes):
   
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))

        loss = ((1-(Yg**self.q))/self.q)*self.weight[indexes] - ((1-(self.k**self.q))/self.q)*self.weight[indexes]
        loss = torch.mean(loss)

        return loss

    def update_weight(self, logits, targets, indexes):
   
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        Lq = ((1-(Yg**self.q))/self.q)
        Lqk = np.repeat(((1-(self.k**self.q))/self.q), targets.size(0))
        Lqk = torch.from_numpy(Lqk).type(torch.cuda.FloatTensor)
        Lqk = torch.unsqueeze(Lqk, 1)
        

        condition = torch.gt(Lqk, Lq)
        self.weight[indexes] = condition.type(torch.cuda.FloatTensor)

        # del p, Yg, Lq, Lqk
        # torch.empty_cache()

def get_criterion(optimizer_name, params):

    if optimizer_name == "CrossEntropyLossOneHot":
        criterion = CrossEntropyLossOneHot()
    elif optimizer_name == "LabelSmoothing":
        criterion = LabelSmoothing(smoothing=params["smoothing"])
    elif optimizer_name == "TaylorCrossEntropyLoss":
        criterion = TaylorCrossEntropyLoss(n=params["n"], smoothing=params["smoothing"])
    elif optimizer_name == "TemperedLoss":
        criterion = TemperedLoss(t1=params["t1"], t2=params["t2"], smoothing=params["smoothing"])
    elif optimizer_name == "FocalCosineLoss":
        criterion = FocalCosineLoss(alpha=params["alpha"], gamma=params["gamma"], xent=params["xent"], smoothing=params["smoothing"])
    elif optimizer_name == "CustomeLoss":
        criterion = CustomeLoss(l=params["l"], t=params["t"])

    elif optimizer_name == "TruncatedLoss":
        criterion = TruncatedLoss(q=params["q"], k=params["k"], trainset_size=params["training_size"])
    
    return criterion