import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import *


class DiceLoss(nn.Module):
    
    def __init__(self, smooth: float=1e-9, reduction: str='mean'):
        super(DiceLoss, self).__init__()
        assert reduction in ('none', 'sum', 'mean')
        self.smooth = smooth
        self.reduction = reduction
        
    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        num_classes = logits.shape[1]
        
        one_hot = torch.eye(num_classes)[labels.cpu().squeeze(dim=1)]
        one_hot = one_hot.permute(0,3,1,2).float()
        logits = F.softmax(logits, dim=1)
        one_hot = one_hot.type(logits.type())
        dims = (0,) + tuple(range(2, labels.ndimension()))
        
        intersection = torch.sum(logits * one_hot, dims)
        summation = torch.sum(logits + one_hot, dims)
        dice = (2. * intersection + self.smooth) / (summation + self.smooth)
        loss = 1. - dice

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss.mean()
        

class OhemCELoss(nn.Module):

    def __init__(self, thresh: float=0.7, ignore_lb: int=255):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        if labels.shape[1] == 1:
            labels = labels.squeeze(dim=1)
        n_min = labels[labels != self.ignore_lb].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)


class CombiLoss(nn.Module):
    """
    Ohem CE Loss + Dice Loss with each weight
    """
    def __init__(self, separate_out=True, dice_loss_weight: float=0.5, ohem_loss_weight: float=0.5, 
                 thresh: float=0.7, ignore_lb: int=255, smooth: float=1e-9, reduction: str='mean'):
        super(CombiLoss, self).__init__()
        assert reduction in ('none', 'sum', 'mean')
        self.dice_loss_weight = dice_loss_weight
        self.ohem_loss_weight = ohem_loss_weight
        self.ohem_loss = OhemCELoss(thresh=thresh, ignore_lb=ignore_lb)
        self.dice_loss = DiceLoss(smooth=smooth, reduction=reduction)
        self.separate_out = separate_out

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        ohem_output = self.ohem_loss(logits, labels.squeeze(dim=1))
        dice_output = self.dice_loss(logits, labels)
        ohem_loss = self.ohem_loss_weight * ohem_output
        dice_loss = self.dice_loss_weight * dice_output

        if self.separate_out:
            return ohem_loss, dice_loss
        else:
            loss = ohem_loss + dice_loss
            return loss
