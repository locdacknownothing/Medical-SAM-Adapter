from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cbam import CBAM
from .layer_norm import LayerNorm2d


class MultiLayerViTFusion(nn.Module):
    """
    Pillar 2: Multi-Layer ViT Global Block Aggregation (MLFA).
    Fuses intermediate global attention block features from the ViT encoder
    with the final neck output via a zero-initialized residual gate so that
    at initialization (step 0) the encoder output is identical to the
    pretrained SAM neck.
    """

    def __init__(
        self,
        embed_dim: int,
        out_chans: int = 256,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.out_chans = out_chans
        self.num_layers = num_layers

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(embed_dim) for _ in range(num_layers)]
        )
        self.projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
                    LayerNorm2d(out_chans),
                    nn.GELU(),
                )
                for _ in range(num_layers)
            ]
        )

        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

        self.refine = nn.Sequential(
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                groups=out_chans,
            ),
            LayerNorm2d(out_chans),
            nn.GELU(),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )

        # Zero-init residual scalar gate: preserves exact pretrained neck output at step 0
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        global_feats: List[torch.Tensor],
        neck_feat: torch.Tensor,
    ) -> torch.Tensor:
        assert len(global_feats) >= self.num_layers, (
            f"Expected at least {self.num_layers} global features, got {len(global_feats)}"
        )
        fused = 0.0
        for i in range(self.num_layers):
            feat = self.layer_norms[i](global_feats[i])
            feat = feat.permute(0, 3, 1, 2).contiguous()
            proj_feat = self.projs[i](feat)
            fused = fused + self.layer_weights[i] * proj_feat

        fused_multi_layer = self.refine(fused)
        gate_scale = self.gamma if self.gamma.abs().item() >= 1e-6 else (self.gamma + 1e-3)
        residual = (
            self.gamma * fused_multi_layer.detach()
            + gate_scale.detach() * (fused_multi_layer - fused_multi_layer.detach())
        )
        return neck_feat + residual


class TubularResBlock(nn.Module):
    """
    Lightweight residual block with parallel 3x3 depthwise conv and (1,7)->(7,1)
    strip depthwise conv for continuous retinal vessel representation.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dw_3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.dw_strip7 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(1, 7), padding=(0, 3), groups=dim),
            nn.Conv2d(dim, dim, kernel_size=(7, 1), padding=(3, 0), groups=dim),
        )
        self.norm = LayerNorm2d(dim)
        self.pw1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        feat = self.dw_3x3(x) + self.dw_strip7(x)
        feat = self.norm(feat)
        feat = self.pw1(feat)
        feat = self.act(feat)
        feat = self.pw2(feat)
        return residual + feat


class HighResVascularStem(nn.Module):
    """
    Pillar 3a: Lightweight High-Resolution Vascular Stem.
    Extracts multi-scale vascular features at 1/1 (16 ch), 1/2 (32 ch), and 1/4 (64 ch)
    resolutions from the preprocessed input image.
    """

    def __init__(
        self,
        in_chans: int = 3,
        dims: Tuple[int, int, int] = (16, 32, 64),
    ) -> None:
        super().__init__()
        c0, c1, c2 = dims

        # Stage 0: 1/1 scale (e.g. 592x592), c0 = 16 channels
        self.stage0 = nn.Sequential(
            nn.Conv2d(in_chans, c0, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(c0),
            nn.GELU(),
            TubularResBlock(c0),
        )

        # Stage 1: 1/2 scale (e.g. 296x296), c1 = 32 channels
        self.stage1 = nn.Sequential(
            nn.Conv2d(c0, c1, kernel_size=3, stride=2, padding=1, bias=False),
            LayerNorm2d(c1),
            nn.GELU(),
            TubularResBlock(c1),
        )

        # Stage 2: 1/4 scale (e.g. 148x148), c2 = 64 channels
        self.stage2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            LayerNorm2d(c2),
            nn.GELU(),
            TubularResBlock(c2),
            CBAM(c2),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s0 = self.stage0(x)
        s1 = self.stage1(s0)
        s2 = self.stage2(s1)
        return s0, s1, s2


class ProgressiveHQRefiner(nn.Module):
    """
    Pillar 3b: Progressive Full-Resolution HQ-MaskDecoder Refiner.
    Progressively fuses MaskDecoder's 1/4-scale upscaled embedding with
    HighResVascularStem skip features at 1/4, 1/2, and 1/1 scales, and adds a
    zero-initialized residual full-resolution logit correction with straight-through
    step-0 gradient flow.
    """

    def __init__(
        self,
        transformer_dim: int = 256,
        num_masks: int = 1,
        stem_dims: Tuple[int, int, int] = (16, 32, 64),
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.num_masks = num_masks
        c0, c1, c2 = stem_dims
        up_dim = transformer_dim // 8  # 32

        # 1/4 scale (148x148) fusion and upsampling to 1/2 scale (296x296)
        self.fuse_1_4 = nn.Sequential(
            nn.Conv2d(up_dim + c2, 32, kernel_size=1, bias=False),
            LayerNorm2d(32),
            nn.GELU(),
            TubularResBlock(32),
        )
        self.up_1_4_to_1_2 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            LayerNorm2d(32),
            nn.GELU(),
        )

        # 1/2 scale (296x296) fusion and upsampling to 1/1 scale (592x592)
        self.fuse_1_2 = nn.Sequential(
            nn.Conv2d(32 + c1, 32, kernel_size=1, bias=False),
            LayerNorm2d(32),
            nn.GELU(),
            TubularResBlock(32),
        )
        self.up_1_2_to_1_1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            LayerNorm2d(16),
            nn.GELU(),
        )

        # 1/1 scale (592x592) fusion with s0 and upsampled coarse logits
        self.fuse_1_1 = nn.Sequential(
            nn.Conv2d(16 + c0 + num_masks, 16, kernel_size=1, bias=False),
            LayerNorm2d(16),
            nn.GELU(),
            TubularResBlock(16),
        )

        # Dynamic HQ token hypernetwork + spatial residual head
        self.hq_mlp = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.GELU(),
            nn.Linear(transformer_dim, transformer_dim),
            nn.GELU(),
            nn.Linear(transformer_dim, 16 * num_masks),
        )
        self.res_head = nn.Conv2d(16, num_masks, kernel_size=1)

        # Zero-init residual scalar gate with straight-through gradient trick
        self.gamma = nn.Parameter(torch.zeros(1))

    def _align_batch_and_spatial(
        self, feat: torch.Tensor, target_b: int, target_hw: Tuple[int, int]
    ) -> torch.Tensor:
        if feat.shape[0] != target_b:
            feat = torch.repeat_interleave(
                feat, max(target_b // feat.shape[0], 1), dim=0
            )
        if feat.shape[-2:] != target_hw:
            feat = F.interpolate(
                feat, size=target_hw, mode="bilinear", align_corners=False
            )
        return feat

    def forward(
        self,
        coarse_masks: torch.Tensor,
        upscaled_embedding: torch.Tensor,
        hq_token_out: torch.Tensor,
        stem_feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        s0, s1, s2 = stem_feats
        b = coarse_masks.shape[0]
        full_hw = s0.shape[-2:]

        # 1. 1/4 scale refinement (148x148 -> 296x296)
        s2_aligned = self._align_batch_and_spatial(s2, b, upscaled_embedding.shape[-2:])
        x_1_4 = self.fuse_1_4(torch.cat([upscaled_embedding, s2_aligned], dim=1))
        x_1_2_up = self.up_1_4_to_1_2(x_1_4)

        # 2. 1/2 scale refinement (296x296 -> 592x592)
        s1_aligned = self._align_batch_and_spatial(s1, b, x_1_2_up.shape[-2:])
        x_1_2 = self.fuse_1_2(torch.cat([x_1_2_up, s1_aligned], dim=1))
        x_1_1_up = self.up_1_2_to_1_1(x_1_2)

        # 3. 1/1 scale refinement (592x592)
        if x_1_1_up.shape[-2:] != full_hw:
            x_1_1_up = F.interpolate(
                x_1_1_up, size=full_hw, mode="bilinear", align_corners=False
            )
        s0_aligned = self._align_batch_and_spatial(s0, b, full_hw)
        coarse_up = F.interpolate(
            coarse_masks, size=full_hw, mode="bilinear", align_corners=False
        )
        feat_1_1 = self.fuse_1_1(
            torch.cat([x_1_1_up, s0_aligned, coarse_up], dim=1)
        )

        # 4. Compute dynamic HQ residual logits delta_logits
        dyn_weights = self.hq_mlp(hq_token_out).view(b, self.num_masks, 16)
        h_full, w_full = full_hw
        dyn_logits = (dyn_weights @ feat_1_1.view(b, 16, h_full * w_full)).view(
            b, self.num_masks, h_full, w_full
        )
        conv_logits = self.res_head(feat_1_1)
        delta_logits = dyn_logits + conv_logits

        # 5. Zero-initialized residual logit fusion with straight-through step-0 gradient
        gate_scale = (
            self.gamma if self.gamma.abs().item() >= 1e-6 else (self.gamma + 1e-3)
        )
        residual = (
            self.gamma * delta_logits.detach()
            + gate_scale.detach() * (delta_logits - delta_logits.detach())
        )
        return coarse_up + residual


class HybridVascularPromptGenerator(nn.Module):
    """
    Pillar 4: Hybrid Learnable Vascular Prompt Generator.
    Generates image-conditioned sparse vascular query tokens via cross-attention
    over image embeddings and a dense spatial vascular prior via TubularResBlock + CBAM,
    both gated by zero-initialized scalars with straight-through step-0 gradient flow.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_queries: int = 4,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.num_heads = num_heads

        # 1. Sparse Vascular Query Generation
        self.vascular_queries = nn.Parameter(
            torch.randn(1, num_queries, embed_dim) * 0.02
        )
        self.q_norm = nn.LayerNorm(embed_dim)
        self.kv_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.sparse_gamma = nn.Parameter(torch.zeros(1))

        # 2. Dense Spatial Vascular Prior Generation
        self.dense_prior = nn.Sequential(
            TubularResBlock(embed_dim),
            CBAM(embed_dim),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        )
        self.dense_gamma = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        image_embeddings: torch.Tensor,
        sparse_embeddings: torch.Tensor,
        dense_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b_img, c, h, w = image_embeddings.shape

        # Sparse vascular queries via cross-attention pooling over flattened image_embeddings
        kv_in = image_embeddings.flatten(2).transpose(1, 2)  # (B_img, H*W, C)
        q_in = self.vascular_queries.expand(b_img, -1, -1)   # (B_img, num_queries, C)
        q = self.q_norm(q_in)
        kv = self.kv_norm(kv_in.to(dtype=q_in.dtype)).to(dtype=q.dtype)
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv, need_weights=False)
        q_res = q_in + attn_out
        learned_sparse = q_res + self.ffn(self.ffn_norm(q_res))

        if learned_sparse.shape[0] != sparse_embeddings.shape[0]:
            learned_sparse = learned_sparse.expand(sparse_embeddings.shape[0], -1, -1)

        eff_sparse_gamma = (
            self.sparse_gamma + (1e-3 - self.sparse_gamma).detach()
            if self.sparse_gamma.abs().item() < 1e-6
            else self.sparse_gamma
        )
        sparse_out = torch.cat(
            [sparse_embeddings, eff_sparse_gamma * learned_sparse], dim=1
        )

        # Dense spatial vascular prior
        prior_feat = self.dense_prior(image_embeddings)
        if prior_feat.shape[0] != dense_embeddings.shape[0]:
            prior_feat = prior_feat.expand(dense_embeddings.shape[0], -1, -1, -1)
        if prior_feat.shape[-2:] != dense_embeddings.shape[-2:]:
            prior_feat = F.interpolate(
                prior_feat,
                size=dense_embeddings.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        eff_dense_gamma = (
            self.dense_gamma + (1e-3 - self.dense_gamma).detach()
            if self.dense_gamma.abs().item() < 1e-6
            else self.dense_gamma
        )
        dense_out = dense_embeddings + eff_dense_gamma * prior_feat

        return sparse_out, dense_out
