# -*- coding: utf-8 -*-
"""Feature Pyramid Network (FPN).

Ported from OVS-Net. Merges multi-scale ConvNeXt features into a
single-scale representation matching the ViT embedding spatial size.
"""

from typing import List

import torch
import torch.nn as nn


class FPN(nn.Module):
    """Top-down Feature Pyramid Network.

    Args:
        in_dims: Channel dimensions of each input feature level
                 (e.g. ``[96, 192, 384, 768]``).
        out_channels: Unified output channel dimension (should match
                      ViT ``out_chans``, typically 256).
        final_spatial_size: Target spatial H=W of the output feature map
                            (e.g. ``img_size // patch_size``).
    """

    def __init__(
        self,
        in_dims: List[int] = [96, 192, 384, 768],
        out_channels: int = 256,
        final_spatial_size: int = 64,
        patch_size: int = 16,
        stem_stride: int = 4,
    ):
        super().__init__()
        self.out_channels = out_channels
        num_levels = len(in_dims)

        # Lateral 1×1 convolutions (from deepest to shallowest)
        self.lat_layers = nn.ModuleList(
            [nn.Conv2d(dim, out_channels, kernel_size=1) for dim in reversed(in_dims)]
        )

        # Transposed-conv up-sampling layers (num_levels - 1)
        self.upconv_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
                for _ in range(num_levels - 1)
            ]
        )

        # High-res refinement layers (num_levels - 1)
        self.high_res_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_levels - 1)
            ]
        )

        self.gelu = nn.GELU()

        # Compute the final downsampling stride required to match final_spatial_size.
        # The FPN upconv outputs have spatial size = img_size / stem_stride (usually img_size / 4).
        # We need final_conv to downsample from (img_size / stem_stride) to (img_size / patch_size).
        # Therefore, stride = patch_size / stem_stride.
        stride = patch_size // stem_stride
        kernel_size = stride + 1
        padding = 1

        self.final_conv = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
        )

        self.layer_norm = nn.LayerNorm([out_channels, final_spatial_size, final_spatial_size])

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: list of tensors from shallowest (stage-0) to deepest
                      (stage-3), as returned by ``ConvNeXt.forward_features``.

        Returns:
            Fused feature map ``(B, out_channels, final_spatial, final_spatial)``.
        """
        # Reverse so we iterate deepest → shallowest
        features = features[::-1]

        x = self.lat_layers[0](features[0])

        for i in range(1, len(features)):
            x = self.upconv_layers[i - 1](x)
            x = x + self.lat_layers[i](features[i])
            x = self.high_res_layers[i - 1](x)
            x = self.gelu(x)

        # Downsample to match ViT embedding spatial size
        x = self.final_conv(x)
        x = self.gelu(x)
        x = self.layer_norm(x)
        return x
