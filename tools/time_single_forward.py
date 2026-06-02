"""
Estimate per-image forward pass runtime for a given config.

Usage:
    python tools/time_single_forward.py your/path/to/config

The script loads the model (+ DINO if enabled), loads one real image from val_dir,
and times N_WARMUP + N_RUNS full forward passes (including tiling if applicable),
reporting mean, std, and extrapolated estimate for the whole dataset.
"""

import sys
import os
import time

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "OmniSR"))
sys.path.insert(0, os.path.join(ROOT, "FreqFusion"))

# ── FreqFusion patch (same as main.py) ──────────────────────────────────────
import importlib, torch, torch.nn.functional as F
from torch.cuda.amp import autocast
_ff = importlib.import_module("FreqFusion")
_FF = _ff.FreqFusion

def _patched_kernel_normalizer(self, mask, kernel, scale_factor=None, hamming=1):
    if scale_factor is not None:
        mask = F.pixel_shuffle(mask, self.scale_factor)
    n, mask_c, h, w = mask.size()
    mask_channel = int(mask_c / float(kernel ** 2))
    mask = mask.view(n, mask_channel, -1, h, w)
    mask = F.softmax(mask, dim=2, dtype=mask.dtype)
    mask = mask.view(n, mask_channel, kernel, kernel, h, w)
    mask = mask.permute(0, 1, 4, 5, 2, 3).reshape(n, -1, kernel, kernel)
    mask = mask * hamming
    mask /= mask.sum(dim=(-1, -2), keepdims=True)
    mask = mask.view(n, mask_channel, h, w, -1)
    mask = mask.permute(0, 1, 4, 2, 3).reshape(n, -1, h, w).contiguous()
    return mask
_FF.kernel_normalizer = _patched_kernel_normalizer

# ── imports ──────────────────────────────────────────────────────────────────
import numpy as np
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP

from config.config import CVConfig
from sgcr.factory.arch_factory import get_arch_sgcr
from sgcr.utility import load_checkpoint
from sgcr.train import (
    maybe_imagenet_normalize_for_dino,
    validate_one_image_tiled,
)
from OmniSR.utils import load_img, load_depth, load_normal, depthToPoint, process_normal

N_WARMUP = 3
N_RUNS   = 10

def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/time_single_forward.py <config_path>")
        print("ERROR: No config path provided. Please set val_dir and pretrain_weights in a local config override.")
        sys.exit(1)
    cfg_path = sys.argv[1]
    print(f"Config: {cfg_path}")

    opt = CVConfig.build_from_configs(cfg_path, do_not_merge_command_line=True)

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    # ── build model ──────────────────────────────────────────────────────────
    print("Loading model...")
    model = get_arch_sgcr(opt, Path(ROOT)).to(device)
    ckpt = opt.pretrain_weights
    load_checkpoint(model, ckpt, strict=False)
    model.eval()
    print(f"  arch={opt.arch}, n_stage={getattr(opt,'n_stage',2)}, "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # ── DINO ─────────────────────────────────────────────────────────────────
    DINO_Net = None
    if opt.use_dino:
        print("Loading DINO...")
        DINO_Net = torch.hub.load(
            str(Path(ROOT) / "OmniSR" / "dinov2"),
            opt.dino_name, source="local"
        ).to(device).eval()
        for p in DINO_Net.parameters():
            p.requires_grad = False
        print(f"  dino={opt.dino_name}")

    # ── dataset paths ─────────────────────────────────────────────────────────
    data_dir   = opt.val_dir
    origin_dir = os.path.join(data_dir, "origin")
    depth_dir  = os.path.join(data_dir, "depth")
    normal_dir = os.path.join(data_dir, "normal")

    M         = 8 * opt.win_size
    img_files = sorted(os.listdir(origin_dir))
    n_images  = len(img_files)

    def load_and_forward(img_file):
        stem        = os.path.splitext(img_file)[0]
        noisy  = np.float32(load_img(os.path.join(origin_dir, img_file)))
        depth  = np.float32(load_depth(os.path.join(depth_dir,  f"depth_{stem}.npy")))
        normal = np.float32(load_normal(os.path.join(normal_dir, f"depth_{stem}.npy")))
        pt = depthToPoint(60, depth)
        nm = process_normal(normal)
        pt = pt / (2 * pt[:, :, 2].mean())
        n_t = torch.from_numpy(noisy).permute(2,0,1).unsqueeze(0).to(device)
        p_t = torch.from_numpy(pt   ).permute(2,0,1).unsqueeze(0).to(device)
        m_t = torch.from_numpy(nm   ).permute(2,0,1).unsqueeze(0).to(device)
        H_o, W_o = n_t.shape[2], n_t.shape[3]
        Hp = ((H_o + M - 1) // M) * M
        Wp = ((W_o + M - 1) // M) * M
        if Hp != H_o or Wp != W_o:
            n_t = F.pad(n_t, (0, Wp-W_o, 0, Hp-H_o), "replicate")
            p_t = F.pad(p_t, (0, Wp-W_o, 0, Hp-H_o), "replicate")
            m_t = F.pad(m_t, (0, Wp-W_o, 0, Hp-H_o), "replicate")
        uses_tiling = opt.tile_size is not None and max(Hp, Wp) > opt.tile_size
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            dino_feats = None
            if DINO_Net is not None:
                inp_dino = F.interpolate(n_t, scale_factor=(14/8), mode="bilinear", align_corners=False)
                inp_dino = maybe_imagenet_normalize_for_dino(inp_dino, opt.dino_imagenet_norm)
                dino_feats = DINO_Net.get_intermediate_layers(inp_dino, 4, True)
            if uses_tiling:
                out = validate_one_image_tiled(
                    opt, model, n_t,
                    DINO_Mat_features=dino_feats,
                    point=p_t, normal=m_t,
                    rgb2x_feat=None, rgb2x_kv_tokens=None,
                    tile=opt.tile_size, overlap=opt.tile_overlap,
                )
            else:
                out = model(n_t, dino_feats, p_t, m_t, rgb2x_feat=None, rgb2x_kv_tokens=None)
        torch.cuda.synchronize()
        return out

    # info from first image
    _s = np.float32(load_img(os.path.join(origin_dir, img_files[0])))
    _h, _w = _s.shape[:2]
    _Hp = ((_h + M - 1) // M) * M
    _Wp = ((_w + M - 1) // M) * M
    print(f"  sample: {img_files[0]}  raw={_h}x{_w}  padded={_Hp}x{_Wp}")
    print(f"  tile_size={opt.tile_size}, tiling={opt.tile_size is not None and max(_Hp,_Wp)>opt.tile_size}")

    # warmup
    print(f"\nWarming up ({N_WARMUP} runs on first image)...")
    for _ in range(N_WARMUP):
        load_and_forward(img_files[0])

    # timed full-dataset pass (reads from disk each time)
    print(f"Timing full dataset ({n_images} images, reading from disk)...")
    per_image_times = []
    torch.cuda.synchronize()
    t_total_start = time.perf_counter()
    for img_file in img_files:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        load_and_forward(img_file)
        t1 = time.perf_counter()
        per_image_times.append(t1 - t0)
    torch.cuda.synchronize()
    t_total = time.perf_counter() - t_total_start

    times = np.array(per_image_times)
    print(f"\n{'='*50}")
    print(f"  Images         : {n_images}")
    print(f"  Mean per image : {times.mean()*1000:.1f} ± {times.std()*1000:.1f} ms")
    print(f"  Min / Max      : {times.min()*1000:.1f} / {times.max()*1000:.1f} ms")
    print(f"  Actual total   : {t_total:.2f} s  ({t_total/60:.2f} min)")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
