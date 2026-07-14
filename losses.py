"""Loss functions for GF-Transformer training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def dice_round(preds, trues, threshold=0.5, empty_score=1.0):
    """
    Compute Dice coefficient between predicted probabilities and binary ground truth.

    Args:
        preds: tensor of shape (B, H, W) with predicted probabilities (sigmoid outputs)
        trues: tensor of shape (B, H, W) with binary ground truth (0/1 or 0/255)
        threshold: threshold to binarize predictions
        empty_score: score returned when both prediction and ground truth are empty

    Returns:
        float: average dice coefficient over the batch
    """
    preds = (preds > threshold).float()
    trues = (trues > 0.5).float()

    batch_size = preds.shape[0]
    dice_sum = 0.0

    for i in range(batch_size):
        p = preds[i].contiguous().view(-1)
        t = trues[i].contiguous().view(-1)

        intersection = (p * t).sum()
        union = p.sum() + t.sum()

        if union == 0:
            dice_sum += empty_score
        else:
            dice_sum += (2.0 * intersection / union).item()

    return dice_sum / batch_size


class FocalLoss(nn.Module):
    """Focal Loss for binary segmentation."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, H, W) logits
            targets: (B, H, W) binary ground truth (0/1)
        """
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1).float()

        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce

        return focal_loss.mean()


class SoftDiceLoss(nn.Module):
    """Soft Dice Loss (differentiable)."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, H, W) logits
            targets: (B, H, W) binary ground truth (0/1)
        """
        inputs = torch.sigmoid(inputs)
        batch_size = inputs.shape[0]
        inputs = inputs.contiguous().view(batch_size, -1)
        targets = targets.contiguous().view(batch_size, -1).float()

        intersection = (inputs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (inputs.sum(dim=1) + targets.sum(dim=1) + self.smooth)

        return (1 - dice).mean()


class ComboLoss(nn.Module):
    """
    Combined loss: weighted sum of Dice loss and Focal loss.

    Args:
        weights: dict with keys 'dice' (float) and 'focal' (float)
        per_image: if True, compute loss per image before averaging
    """

    def __init__(self, weights, per_image=False):
        super().__init__()
        self.weights = weights
        self.per_image = per_image
        self.dice_loss = SoftDiceLoss()
        self.focal_loss = FocalLoss()

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, H, W) logits
            targets: (B, H, W) binary ground truth

        Returns:
            scalar loss
        """
        d_loss = self.dice_loss(inputs, targets) * self.weights.get('dice', 1.0)
        f_loss = self.focal_loss(inputs, targets) * self.weights.get('focal', 1.0)

        return d_loss + f_loss
