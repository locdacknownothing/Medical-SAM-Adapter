import torch
import torch.nn as nn
from .cbam import CBAM
from .convnext import ConvNeXtBlock,ConvNeXtLayerNorm


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
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
    def __init__(self, D_features, mlp_ratio=0.25, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden = int(D_features * mlp_ratio)
        
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
        