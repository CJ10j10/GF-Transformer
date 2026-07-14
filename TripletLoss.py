"""Triplet Margin Loss for deep metric learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletMarginLoss(nn.Module):
    """
    Triplet margin loss: ensures embeddings of the same class are closer
    than embeddings of different classes by at least `margin`.

    This is a standard implementation compatible with the GF-Transformer
    training script. Note: the loss6 (triplet loss) is commented out in
    train_segformer_cls.py, so this class serves as a placeholder that
    can be enabled when needed.
    """

    def __init__(self, margin=1.0, p=2.0, eps=1e-6, swap=False):
        super().__init__()
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap

    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: (B, D) anchor embeddings
            positive: (B, D) positive (same-class) embeddings
            negative: (B, D) negative (different-class) embeddings

        Returns:
            scalar loss
        """
        d_ap = F.pairwise_distance(anchor, positive, p=self.p, eps=self.eps)
        d_an = F.pairwise_distance(anchor, negative, p=self.p, eps=self.eps)

        if self.swap:
            d_pn = F.pairwise_distance(positive, negative, p=self.p, eps=self.eps)
            d_an = torch.min(d_an, d_pn)

        loss = F.relu(d_ap - d_an + self.margin)

        return loss.mean()
