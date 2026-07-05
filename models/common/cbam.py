# -*- coding: utf-8 -*-
"""CBAM: Convolutional Block Attention Module.

Ported from OVS-Net. Contains channel and spatial attention sub-modules.
"""

import torch
import torch.nn as nn


class ChannelAttentionModule(nn.Module):
    """Channel attention: squeeze via avg+max pool → shared MLP → sigmoid."""

    def __init__(self, channel: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    """Spatial attention: channel-wise mean+max → conv → sigmoid."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv2d = nn.Conv2d(
            in_channels=2, out_channels=1,
            kernel_size=kernel_size, stride=1, padding=padding,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel → spatial)."""

    def __init__(self, channel: int, ratio: int = 16):
        super().__init__()
        self.channel_attention = ChannelAttentionModule(channel, ratio)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
