# -*- coding: utf-8 -*-
"""Cross-Attention Fusion module.

Ported from OVS-Net. Fuses CNN (FPN) features with ViT features via
multi-head cross-attention.
"""

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """Fuse two feature maps via cross-attention.

    Query is derived from ``conv_features`` (CNN/FPN branch), while
    Key and Value come from ``vit_features`` (ViT branch).

    Args:
        d_model: Channel dimension of both input feature maps.
        n_heads: Number of attention heads.
    """

    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.query_conv = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.key_conv = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.value_conv = nn.Conv2d(d_model, d_model, kernel_size=1)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True,
        )

        self.out_conv = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        conv_features: torch.Tensor,
        vit_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            conv_features: ``(B, C, H, W)`` from CNN/FPN branch (query source).
            vit_features:  ``(B, C, H, W)`` from ViT neck (key/value source).

        Returns:
            Fused features ``(B, C, H, W)``.
        """
        b, c, h, w = conv_features.shape

        # Project and flatten spatial dims → (B, H*W, C)
        query = self.query_conv(conv_features).view(b, c, -1).permute(0, 2, 1)
        key = self.key_conv(vit_features).view(b, c, -1).permute(0, 2, 1)
        value = self.value_conv(vit_features).view(b, c, -1).permute(0, 2, 1)

        attn_output, _ = self.multihead_attn(query, key, value)

        # Reshape back to spatial
        attn_output = attn_output.permute(0, 2, 1).view(b, c, h, w)

        # Residual + LayerNorm (normalise over channel dim)
        output = self.norm(
            (conv_features + attn_output).permute(0, 2, 3, 1)
        )

        # Final projection
        output = self.out_conv(output.permute(0, 3, 1, 2))
        return output
