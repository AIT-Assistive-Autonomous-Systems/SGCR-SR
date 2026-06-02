import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- vendor dirs (match your current pattern) ---
_THIS = Path(__file__).resolve()
ROOT = _THIS.parents[1]
sys.path.insert(0, str(ROOT / "OmniSR"))

from OmniSR.utils.grid_sample import grid_sample


# -------------------------
# Simple U-Net building blocks
# -------------------------
class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, groups: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        g = min(groups, out_ch)
        while out_ch % g != 0 and g > 1:
            g -= 1
        self.gn = nn.GroupNorm(g, out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(in_ch, out_ch),
            ConvGNAct(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downsample by stride-2 conv, then refine."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = ConvGNAct(in_ch, out_ch, k=3, s=2, p=1)
        self.refine = ConvGNAct(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return self.refine(x)


# -------------------------
# ShadowFormer-like U-Net model (NO DINO)
# -------------------------
class ShadowFormer_OmniSR_NoDINO(nn.Module):
    """
    ShadowFormer-like wrapper with:
      - 3-level U-Net backbone
    """

    def __init__(
        self,
        opt,
    ):
        super().__init__()

        e = int(getattr(opt, "embed_dim", 64))
        self.embed_dim = e

        # ---- Geometry injection (optional) ----
        self.geom0 = nn.Conv2d(6, e, kernel_size=1, bias=False)
        self.geom1 = nn.Conv2d(6, 2 * e, kernel_size=1, bias=False)
        self.geom2 = nn.Conv2d(6, 4 * e, kernel_size=1, bias=False)
        self.geom3 = nn.Conv2d(6, 8 * e, kernel_size=1, bias=False)
        self.geomb = nn.Conv2d(6, 16 * e, kernel_size=1, bias=False)

        # ---- U-Net backbone ----
        # Input: RGB + depth(z) if point is provided, else RGB + zeros
        in_ch = 4  # 3 rgb + 1 depth
        self.stem = DoubleConv(in_ch, e)        # H
        self.down1 = Down(e, 2 * e)             # H/2
        self.down2 = Down(2 * e, 4 * e)         # H/4
        self.down3 = Down(4 * e, 8 * e)         # H/8
        self.bot = DoubleConv(8 * e, 16 * e)    # H/8

        # ---- Projections for decoder upsampling ----
        self.proj_lr0 = nn.Conv2d(16 * e, 4 * e, kernel_size=1)  # 1/8 -> 1/4
        self.proj_lr1 = nn.Conv2d(8 * e, 2 * e, kernel_size=1)   # 1/4 -> 1/2
        self.proj_lr2 = nn.Conv2d(4 * e, 1 * e, kernel_size=1)   # 1/2 -> 1/1

        # ---- Decoder refine blocks (post fusion + skip concat) ----
        self.dec0 = DoubleConv(8 * e, 8 * e)  # (up0 4e + skip 4e)
        self.dec1 = DoubleConv(4 * e, 4 * e)  # (up1 2e + skip 2e)
        self.dec2 = DoubleConv(2 * e, 2 * e)  # (up2 1e + skip 1e)


        # ---- Output residual head ----
        self.out_res = nn.Conv2d(2 * e, 3, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        DINO_Mat_features: Optional[object] = None,  # <-- accept but ignore
        point: Optional[torch.Tensor] = None,
        normal: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        B, _, H, W = x.shape
        h2, w2 = H // 2, W // 2
        h4, w4 = H // 4, W // 4
        h8, w8 = H // 8, W // 8

        # ---- geometry pyramids ----
        if point is not None and normal is not None:
            p0, n0 = point, normal
            p1, n1 = grid_sample(point, (h2, w2)), grid_sample(normal, (h2, w2))
            p2, n2 = grid_sample(point, (h4, w4)), grid_sample(normal, (h4, w4))
            p3, n3 = grid_sample(point, (h8, w8)), grid_sample(normal, (h8, w8))
        else:
            p0 = n0 = p1 = n1 = p2 = n2 = p3 = n3 = None

        # ---- input: RGB + depth ----
        if point is not None:
            depth = point[:, 2:3, :, :]
        else:
            depth = torch.zeros((B, 1, H, W), device=x.device, dtype=x.dtype)
        xi = torch.cat([x, depth], dim=1)  # [B,4,H,W]

        # ---- encoder ----
        enc0 = self.stem(xi)               # [B,e,H,W]
        if p0 is not None:
            enc0 = enc0 + self.geom0(torch.cat([p0, n0], dim=1))

        enc1 = self.down1(enc0)            # [B,2e,H/2,W/2]
        if p1 is not None:
            enc1 = enc1 + self.geom1(torch.cat([p1, n1], dim=1))

        enc2 = self.down2(enc1)            # [B,4e,H/4,W/4]
        if p2 is not None:
            enc2 = enc2 + self.geom2(torch.cat([p2, n2], dim=1))

        enc3 = self.down3(enc2)            # [B,8e,H/8,W/8]
        if p3 is not None:
            enc3 = enc3 + self.geom3(torch.cat([p3, n3], dim=1))

        bot = self.bot(enc3)               # [B,16e,H/8,W/8]
        if p3 is not None:
            bot = bot + self.geomb(torch.cat([p3, n3], dim=1))


        # Stage 0: 1/8 -> 1/4
        lr0 = self.proj_lr0(bot)  # [B,4e,h8,w8]
        hr2 = enc2
        up0_map = F.interpolate(lr0, size=(h4, w4), mode="bilinear", align_corners=False)

        dec0 = self.dec0(torch.cat([up0_map, hr2], dim=1))  # [B,8e,h4,w4]

        # Stage 1: 1/4 -> 1/2
        lr1 = self.proj_lr1(dec0)  # [B,2e,h4,w4]
        hr1 = enc1
        up1_map = F.interpolate(lr1, size=(h2, w2), mode="bilinear", align_corners=False)

        dec1 = self.dec1(torch.cat([up1_map, hr1], dim=1))  # [B,4e,h2,w2]

        # Stage 2: 1/2 -> 1/1
        lr2 = self.proj_lr2(dec1)  # [B,e,h2,w2]
        hr0 = enc0
        up2_map = F.interpolate(lr2, size=(H, W), mode="bilinear", align_corners=False)

        dec2 = self.dec2(torch.cat([up2_map, hr0], dim=1))  # [B,2e,H,W]

        res = self.out_res(dec2)  # [B,3,H,W]
        y = x + res
        return y

