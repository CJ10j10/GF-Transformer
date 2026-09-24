"""Lightweight residual feature refinement between GF and CSGF."""

import torch
from torch import nn
from torch.nn import functional as F


class KalmanRefine(nn.Module):
    def __init__(self, feature_channels: int, global_channels: int):
        super().__init__()
        # GF expands channels, so each observation branch needs one 1x1 map.
        self.obs_proj = nn.Conv2d(feature_channels, global_channels, 1)
        self.p_net = nn.Conv2d(global_channels, global_channels, 1)
        self.r_net = nn.Conv2d(feature_channels, global_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(()))
        self.record_k_stats = False
        self.last_k_stats = None

    def forward(self, pre_feat, post_feat, global_feat):
        innovation = post_feat - pre_feat
        z = self.obs_proj(innovation)
        p = F.softplus(self.p_net(global_feat)) + 1e-6
        r = F.softplus(self.r_net(innovation.abs())) + 1e-6
        k = p / (p + r)
        if self.record_k_stats:
            values = k.detach().float()
            self.last_k_stats = {
                'min': values.min().item(),
                'max': values.max().item(),
                'mean': values.mean().item(),
                'std': values.std(unbiased=False).item(),
            }
        return global_feat + self.gamma * k * z
