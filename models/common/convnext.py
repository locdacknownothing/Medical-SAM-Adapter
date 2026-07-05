# -*- coding: utf-8 -*-
"""ConvNeXt backbone with CBAM integration.

Ported from OVS-Net. Provides a 4-stage hierarchical CNN encoder that
produces multi-scale feature maps for use with an FPN.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

from .cbam import CBAM


class ConvNeXtLayerNorm(nn.Module):
    """LayerNorm supporting both channels-last and channels-first formats.

    Distinct from the existing ``LayerNorm2d`` which only supports
    channels-first.  ConvNeXt blocks need channels-last normalisation
    internally.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6,
                 data_format: str = "channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps,
            )
        # channels_first
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNeXtBlock(nn.Module):
    """Single ConvNeXt residual block.

    DWConv7×7 → LN → Linear(dim→4·dim) → GELU → Linear(4·dim→dim)
    → LayerScale → DropPath + residual.
    """

    def __init__(self, dim: int, drop_path: float = 0.0,
                 layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = ConvNeXtLayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)          # (B, C, H, W) → (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)          # (B, H, W, C) → (B, C, H, W)
        x = residual + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    """4-stage ConvNeXt backbone with CBAM at each stage.

    Args:
        in_chans: Number of input image channels.
        depths: Number of ConvNeXtBlocks per stage.
        dims: Channel dimensions for each stage.
        drop_path_rate: Stochastic depth rate.
        layer_scale_init_value: Init value for LayerScale.
    """

    def __init__(
        self,
        in_chans: int = 3,
        depths: List[int] = [3, 3, 9, 3],
        dims: List[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()

        # ---- Downsample layers ----
        self.downsample_layers = nn.ModuleList()
        # Stage-0 stem
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            ConvNeXtLayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        # Stages 1–3
        for i in range(3):
            downsample_layer = nn.Sequential(
                ConvNeXtLayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # ---- ConvNeXtBlock stages ----
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    ConvNeXtBlock(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        # ---- CBAM per stage ----
        self.cbams = nn.ModuleList([CBAM(dim) for dim in dims])

        # ---- Weight init ----
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features.

        Returns:
            features: list of 4 tensors, one per stage, collected
                      **after** CBAM residual and **before** the
                      ConvNeXtBlock stage.
        """
        features: List[torch.Tensor] = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = x + self.cbams[i](x)
            features.append(x)
            x = self.stages[i](x)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return features[-1]
