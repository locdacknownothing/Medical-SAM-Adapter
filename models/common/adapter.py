import torch
import torch.nn as nn


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
    """Fully convolutional adapter with ResNet-style bottleneck.
    
    Replaces the original linear adapter with a 1×1 → 3×3 → 1×1 conv
    bottleneck. Designed for SAM's ViT which maintains 4D (B,H,W,D) format.
    """
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden = int(D_features * mlp_ratio)
        self.act = act_layer()
        
        # 1×1 pointwise down-projection
        self.conv_down = nn.Conv2d(D_features, D_hidden, kernel_size=1)
        # 3×3 spatial + cross-channel mixing at bottleneck
        self.conv_spatial = nn.Conv2d(D_hidden, D_hidden, kernel_size=3, padding=1)
        # 1×1 pointwise up-projection
        self.conv_up = nn.Conv2d(D_hidden, D_features, kernel_size=1)
        
        # Zero-init output conv for stable training start
        nn.init.zeros_(self.conv_up.weight)
        nn.init.zeros_(self.conv_up.bias)

    def forward(self, x):
        # print("Adapter input shape:", x.shape)
        xs = x.permute(0, 3, 1, 2).contiguous()    # → (B, D, H, W)
        xs = self.act(self.conv_down(xs))           # → (B, D/4, H, W)
        xs = self.act(self.conv_spatial(xs))         # → (B, D/4, H, W)
        xs = self.conv_up(xs)                        # → (B, D, H, W)
        xs = xs.permute(0, 2, 3, 1).contiguous()    # → (B, H, W, D)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x