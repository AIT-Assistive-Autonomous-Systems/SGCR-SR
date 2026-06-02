import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from OmniSR.utils.grid_sample import grid_sample

# --- add vendor dirs to import path ---
_THIS = Path(__file__).resolve()
ROOT = _THIS.parents[1]
sys.path.insert(0, str(ROOT / "OmniSR"))

from model import ShadowFormer  # OmniSR model.py: ShadowFormer


def tokens_to_map(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    # x: [B, H*W, C] -> [B, C, H, W]
    b, n, c = x.shape
    assert n == h * w
    return x.transpose(1, 2).contiguous().view(b, c, h, w)

def map_to_tokens(x: torch.Tensor) -> torch.Tensor:
    # x: [B, C, H, W] -> [B, H*W, C]
    b, c, h, w = x.shape
    return x.flatten(2).transpose(1, 2).contiguous()

class ShadowFormer_OmniSR(ShadowFormer):
    """
    OmniSR ShadowFormer with upsampling/fusion at each decoder stage.
    Keeps everything else identical to OmniSR.
    """

    def __init__(self, opt, *args, **kwargs):
        kwargs.setdefault("img_size", opt.train_ps)
        kwargs.setdefault("embed_dim", opt.embed_dim)
        kwargs.setdefault("win_size", opt.win_size)
        kwargs.setdefault("token_projection", opt.token_projection)
        kwargs.setdefault("token_mlp", opt.token_mlp)
        super().__init__(*args, **kwargs)

        # ---- DINO channel adapter (registered, DDP-safe) ----
        dino_name = opt.dino_name
        expected = 1024  # model is built around vitl14-width DINO features
        actual = {"dinov2_vitl14": 1024, "dinov2_vitb14": 768, "dinov2_vits14": 384}[dino_name]
        self.dino_proj = nn.Identity() if actual == expected else nn.Conv2d(actual, expected, kernel_size=1, bias=False)

        # OmniSR embed_dim is self.embed_dim
        e = self.embed_dim

        # In OmniSR forward():
        # conv0: [B, HW, e] at full res
        # conv1: [B, HW/4, 2e] at 1/2 res
        # conv2: [B, HW/16, 4e] at 1/4 res
        # conv3: [B, HW/64, 16e] at 1/8 res (after cat with DINO features)

        # Project LR decoder features to match HR skip channels at each decoder stage
        self.proj_lr0 = nn.Conv2d(16 * e, 4 * e, kernel_size=1)   # 1/8 -> fuse into 1/4
        self.proj_lr1 = nn.Conv2d(8 * e,  2 * e, kernel_size=1)   # 1/4 -> fuse into 1/2
        self.proj_lr2 = nn.Conv2d(4 * e,  1 * e, kernel_size=1)   # 1/2 -> fuse into 1/1


    def forward(self, x, DINO_Mat_features=None, point=None, normal=None, mask=None, **_):
        point_feature=None
        dino_mat =None
        dino_mat1=None

        # --- base spatial sizes (keep as plain ints/tuples; avoid torch.tensor here) ---
        H, W = int(x.shape[2]), int(x.shape[3])
        self.img_size = (H, W)
        img1 = (H // 2, W // 2)
        img2 = (H // 4, W // 4)
        img3 = (H // 8, W // 8)

        point_feature1 = grid_sample(point, img1)
        point_feature2 = grid_sample(point, img2)
        point_feature3 = grid_sample(point, img3)
        normal1 = grid_sample(normal, img1)
        normal2 = grid_sample(normal, img2)
        normal3 = grid_sample(normal, img3)

        # DINO gives a global semantic token set (skipped when DINO_Mat_features is None)
        if DINO_Mat_features is not None:
            patch_features_0 = self.dino_proj(DINO_Mat_features[0])
            patch_features_1 = self.dino_proj(DINO_Mat_features[1])
            patch_features_2 = self.dino_proj(DINO_Mat_features[2])
            patch_features_3 = self.dino_proj(DINO_Mat_features[3])
            patch_feature_all = torch.cat((patch_features_0, patch_features_1,
                                        patch_features_2, patch_features_3), dim=1)
            dino_mat_cat = self.Conv(patch_feature_all)
            dino_mat_cat = self.relu(dino_mat_cat)
        else:
            # No DINO: inject zero conditioning so the bottleneck cat keeps the same shape
            dino_mat_cat = torch.zeros(
                x.shape[0], self.Conv.out_channels, *img3,
                device=x.device, dtype=x.dtype
            )

        #B, Cc, Hc, Wc = dino_mat_cat.shape
        #dino_mat_cat_flat = dino_mat_cat.view(B, Cc, Hc * Wc).permute(0, 2, 1)
            
        # RGBD: we are adding the depth information
        xi = torch.cat((x, point[:,2,:].unsqueeze(1)), dim=1)

        y = self.input_proj(xi)
        y = self.pos_drop(y)

        # Encoder
        self.img_size = (H, W)
        conv0 = self.encoderlayer_0(y, dino_mat, point_feature, normal, mask, img_size = self.img_size)
        pool0 = self.dowsample_0(conv0, img_size = self.img_size)

        self.img_size = (H // 2, W // 2)
        conv1 = self.encoderlayer_1(pool0, dino_mat1, point_feature1, normal1, img_size = self.img_size)
        pool1 = self.dowsample_1(conv1, img_size = self.img_size)

        # Stage 2 (1/4 res): resize DINO map to the *current* token grid
        self.img_size = (H // 4, W // 4)
        dino_mat2 = self.resize_to_stage(self.dino_proj(DINO_Mat_features[-1]), self.img_size) if DINO_Mat_features is not None else torch.ones(x.shape[0], 1, *img2, device=x.device, dtype=x.dtype)
        conv2 = self.encoderlayer_2(pool1, dino_mat2, point_feature2, normal2, mask, img_size = self.img_size)
        pool2 = self.dowsample_2(conv2, img_size = self.img_size)

        # Bottleneck
        # Stage 3 (1/8 res): resize DINO map to the *current* token grid
        self.img_size = (H // 8, W // 8)
        dino_mat3 = self.resize_to_stage(self.dino_proj(DINO_Mat_features[-1]), self.img_size) if DINO_Mat_features is not None else torch.ones(x.shape[0], 1, *img3, device=x.device, dtype=x.dtype)
        dino_mat_cat_rs = F.interpolate(dino_mat_cat, size=img3, mode="bilinear", align_corners=False)
        dino_mat_cat_flat = dino_mat_cat_rs.flatten(2).transpose(1, 2).contiguous()
        pool2 = torch.cat([pool2, dino_mat_cat_flat], -1)
        conv3 = self.conv(pool2, dino_mat3, point_feature3, normal3, mask, img_size = self.img_size)

        #Decoder
        # At this point in OmniSR forward:
        # self.img_size is the spatial size of conv3 tokens (the 1/8 resolution).
        h8, w8 = self.img_size

        # conv2 is at 1/4 resolution, so:
        h4, w4 = h8 * 2, w8 * 2
        h2, w2 = h4 * 2, w4 * 2
        h1, w1 = h2 * 2, w2 * 2

        # --- Stage 0: 1/8 -> 1/4 using conv2 skip (4e channels) ---
        conv2_map = tokens_to_map(conv2, h4, w4)            # [B, 4e, h4, w4]
        conv3_map = tokens_to_map(conv3, h8, w8)            # [B,16e, h8, w8]
        conv3_map = self.proj_lr0(conv3_map)                # [B, 4e, h8, w8]
        up0_map = F.interpolate(conv3_map, size=conv2_map.shape[-2:], mode="bilinear", align_corners=False)


        up0 = map_to_tokens(up0_map)
        conv2_ff = map_to_tokens(conv2_map)
        deconv0 = torch.cat([up0, conv2_ff], dim=-1)
        deconv0 = self.decoderlayer_0(deconv0, dino_mat2, point_feature2, normal2, mask, img_size=(h4, w4))

        # --- Stage 1: 1/4 -> 1/2 using conv1 skip (2e channels) ---
        conv1_map = tokens_to_map(conv1, h2, w2)            # [B,2e,h2,w2]
        deconv0_map = tokens_to_map(deconv0, h4, w4)        # [B,8e,h4,w4]
        deconv0_map = self.proj_lr1(deconv0_map)            # [B,2e,h4,w4]
        up1_map = F.interpolate(deconv0_map, size=conv1_map.shape[-2:], mode="bilinear", align_corners=False)


        up1 = map_to_tokens(up1_map)
        conv1_ff = map_to_tokens(conv1_map)
        deconv1 = torch.cat([up1, conv1_ff], dim=-1)
        deconv1 = self.decoderlayer_1(deconv1, dino_mat1, point_feature1, normal1, img_size=(h2, w2))
        # --- Stage 2: 1/2 -> 1/1 using conv0 skip (e channels) ---
        conv0_map = tokens_to_map(conv0, h1, w1)            # [B,e,h1,w1]
        deconv1_map = tokens_to_map(deconv1, h2, w2)        # [B,4e,h2,w2]
        deconv1_map = self.proj_lr2(deconv1_map)            # [B,e,h2,w2]
        up2_map = F.interpolate(deconv1_map, size=conv0_map.shape[-2:], mode="bilinear", align_corners=False)


        up2 = map_to_tokens(up2_map)
        conv0_ff = map_to_tokens(conv0_map)
        deconv2 = torch.cat([up2, conv0_ff], dim=-1)
        deconv2 = self.decoderlayer_2(deconv2, dino_mat, point_feature, normal, mask, img_size=(h1, w1))
        y = self.output_proj(deconv2, img_size=(h1, w1)) + x
        return y

    def resize_to_stage(self, d: Optional[torch.Tensor], img_size: Tuple[int, int]) -> Optional[torch.Tensor]:
        """Resize a BCHW feature map to match a stage token grid (h,w)."""
        if d is None:
            return None
        h, w = int(img_size[0]), int(img_size[1])
        return F.interpolate(d, size=(h, w), mode="bilinear", align_corners=False)
