import math
import torch
import torch.nn as nn
from .cbam import CBAM
from .convnext import ConvNeXtBlock, ConvNeXtLayerNorm, GRN


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True, hidden_dim=None):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = hidden_dim if hidden_dim is not None else int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        # print("Adapter input shape:", x.shape)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class ConvAdapter(nn.Module):
    """Hybrid ConvNeXt-CBAM Adapter.
    
    Replaces the original linear adapter with:
    1x1 Conv (Down) -> ConvNeXtBlock -> CBAM -> 1x1 Conv (Up)
    """
    def __init__(self, D_features, mlp_ratio=0.25, skip_connect=True, hidden_dim=None):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden = hidden_dim if hidden_dim is not None else int(D_features * mlp_ratio)
        
        # 1×1 pointwise down-projection
        self.conv_down = nn.Conv2d(D_features, D_hidden, kernel_size=1)
        self.norm_first = ConvNeXtLayerNorm(D_hidden, eps=1e-6, data_format="channels_first")
        # ConvNeXt Block
        self.convnext = ConvNeXtBlock(dim=D_hidden)
        # CBAM Block
        self.cbam = CBAM(channel=D_hidden)
        # 1×1 pointwise up-projection
        self.conv_up = nn.Conv2d(D_hidden, D_features, kernel_size=1)
        
        # # Zero-init output conv for stable training start
        # nn.init.zeros_(self.conv_up.weight)
        # nn.init.zeros_(self.conv_up.bias)

    def forward(self, x):
        # print("Input:", x.shape)  # last
        xs = x.permute(0, 3, 1, 2).contiguous()
        xs = self.conv_down(xs)    # first
        xs = self.norm_first(xs)

        # ConvNeXt
        # print("ConvNeXt Input:", x.shape)
        xs = self.convnext(xs)
        # residual = xs
        # xs = self.dwconv(xs)
        # xs = xs.permute(0, 2, 3, 1)  # last
        # xs = self.norm(xs)
        # xs = self.pwconv1(xs)
        # xs = self.act(xs)
        # xs = self.pwconv2(xs)
        # if self.gamma is not None:
        #     xs = self.gamma * xs
        # xs = xs.permute(0, 3, 1, 2)  # first
        # xs = residual + self.drop_path(xs)

        # CBAM
        xs = self.cbam(xs)  

        xs = self.conv_up(xs)
        xs = xs.permute(0, 2, 3, 1).contiguous()  # last
        
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class TubularConvAdapter(nn.Module):
    """Multi-Scale Tubular Strip-Conv Adapter with GRN and CBAM.

    Designed for continuous, unpartitioned retinal vessel feature maps:
    1x1 Conv (Down) -> LN -> Parallel Multi-Scale Tubular Depthwise Core
    (3x3 + dilated 3x3 + 1x7/7x1 strip + 1x11/11x1 strip + LN + 2x MLP + GRN)
    -> CBAM -> 1x1 Conv (Up)
    """

    def __init__(self, D_features, mlp_ratio=0.25, skip_connect=True, hidden_dim=None):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden = hidden_dim if hidden_dim is not None else int(D_features * mlp_ratio)

        # 1x1 pointwise down-projection + channels-first LayerNorm
        self.conv_down = nn.Conv2d(D_features, D_hidden, kernel_size=1)
        self.norm_first = ConvNeXtLayerNorm(D_hidden, eps=1e-6, data_format="channels_first")

        # Parallel Multi-Scale Tubular Depthwise Core
        # Branch 1: Local isotropic (junctions & bifurcations)
        self.dw_3x3 = nn.Conv2d(D_hidden, D_hidden, kernel_size=3, padding=1, groups=D_hidden)
        # Branch 2: Dilated context
        self.dw_dilated = nn.Conv2d(
            D_hidden, D_hidden, kernel_size=3, padding=2, dilation=2, groups=D_hidden
        )
        # Branch 3: Medium tubular strip (1x7 -> 7x1)
        self.dw_strip7 = nn.Sequential(
            nn.Conv2d(D_hidden, D_hidden, kernel_size=(1, 7), padding=(0, 3), groups=D_hidden),
            nn.Conv2d(D_hidden, D_hidden, kernel_size=(7, 1), padding=(3, 0), groups=D_hidden),
        )
        # Branch 4: Long tubular strip (1x11 -> 11x1)
        self.dw_strip11 = nn.Sequential(
            nn.Conv2d(D_hidden, D_hidden, kernel_size=(1, 11), padding=(0, 5), groups=D_hidden),
            nn.Conv2d(D_hidden, D_hidden, kernel_size=(11, 1), padding=(5, 0), groups=D_hidden),
        )

        # Channel mixing with ConvNeXtLayerNorm + 2x inverted bottleneck + GRN
        self.norm = ConvNeXtLayerNorm(D_hidden, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(D_hidden, 2 * D_hidden)
        self.act = nn.GELU()
        self.grn = GRN(2 * D_hidden)
        self.pwconv2 = nn.Linear(2 * D_hidden, D_hidden)

        # CBAM Attention
        self.cbam = CBAM(channel=D_hidden)

        # 1x1 pointwise up-projection
        self.conv_up = nn.Conv2d(D_hidden, D_features, kernel_size=1)

    def _forward_spatial(self, xs: torch.Tensor) -> torch.Tensor:
        # xs: (B, C, H, W)
        xs = self.conv_down(xs)
        xs = self.norm_first(xs)

        x_in = xs
        b1 = self.dw_3x3(x_in)
        b2 = self.dw_dilated(x_in)
        b3 = self.dw_strip7(x_in)
        b4 = self.dw_strip11(x_in)
        dw_out = x_in + b1 + b2 + b3 + b4

        mixed = dw_out.permute(0, 2, 3, 1)  # (B, H, W, C)
        mixed = self.norm(mixed)
        mixed = self.pwconv1(mixed)
        mixed = self.act(mixed)
        mixed = self.grn(mixed)
        mixed = self.pwconv2(mixed)
        mixed = mixed.permute(0, 3, 1, 2)  # (B, C, H, W)

        xs = x_in + mixed
        xs = self.cbam(xs)
        xs = self.conv_up(xs)
        return xs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            # x is (B, H, W, C)
            xs = x.permute(0, 3, 1, 2).contiguous()
            xs = self._forward_spatial(xs)
            xs = xs.permute(0, 2, 3, 1).contiguous()
        elif x.dim() == 3:
            # x is (B, N, C)
            B, N, C = x.shape
            H = int(math.isqrt(N))
            W = N // H
            if H * W == N:
                xs = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                xs = self._forward_spatial(xs)
                xs = xs.permute(0, 2, 3, 1).contiguous().view(B, N, C)
            else:
                xs = x.permute(0, 2, 1).unsqueeze(2).contiguous()  # (B, C, 1, N)
                xs = self._forward_spatial(xs)
                xs = xs.squeeze(2).permute(0, 2, 1).contiguous()
        else:
            raise ValueError(f"Unsupported input tensor dimension {x.dim()} for TubularConvAdapter")

        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x
