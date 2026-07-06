import torch
import torch.nn as nn
from .cbam import CBAM
from .convnext import ConvNeXtBlock


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
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden = int(D_features * mlp_ratio)
        self.act = act_layer()
        
        # 1×1 pointwise down-projection
        self.conv_down = nn.Conv2d(D_features, D_hidden, kernel_size=1)
        # ConvNeXt Block
        self.convnext = ConvNeXtBlock(dim=D_hidden)
        # CBAM Block
        self.cbam = CBAM(channel=D_hidden)
        # 1×1 pointwise up-projection
        self.conv_up = nn.Conv2d(D_hidden, D_features, kernel_size=1)
        
        # Zero-init output conv for stable training start
        nn.init.zeros_(self.conv_up.weight)
        nn.init.zeros_(self.conv_up.bias)

    def forward(self, x):
        xs = x.permute(0, 3, 1, 2).contiguous()    # → (B, D, H, W)
        xs = self.act(self.conv_down(xs))           # → (B, D_hidden, H, W)
        xs = self.cbam(xs)                          # → (B, D_hidden, H, W)
        xs = self.convnext(xs)                      # → (B, D_hidden, H, W)
        xs = self.conv_up(xs)                        # → (B, D, H, W)
        xs = xs.permute(0, 2, 3, 1).contiguous()    # → (B, H, W, D)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x