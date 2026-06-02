# validation code, it can be used for both test and validation sets
import os
import sys
from pathlib import Path
from torch.utils.data import Dataset
from OmniSR.utils import load_normal, load_ssao, load_img, depthToPoint, process_normal, load_depth, Augment_RGB_torch
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
from torchmetrics.functional import multiscale_structural_similarity_index_measure as ms_ssim_fn
from torchmetrics.functional import peak_signal_noise_ratio as psnr_fn

try:
    from scipy import stats as _scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "OmniSR"))

import OmniSR.utils as utils
from utils.loader import get_validation_data

from sgcr.factory.arch_factory import get_arch_sgcr
from sgcr.ntire_metrics import NTIREMetrics
from sgcr.utility import to_hwc_rgb_uint8, load_checkpoint
from sgcr.tiled_inference import validate_one_image_tiled

import math
import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ski_ssim

def _gaussian_kernel_2d(window_size: int, sigma: float, device, dtype):
    # 1D gaussian
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    # outer product -> 2D
    k2d = (g[:, None] * g[None, :]).contiguous()
    return k2d

@torch.no_grad()
def ssim_map_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    K1: float = 0.01,
    K2: float = 0.03,
):
    """
    Compute an SSIM map (per-pixel) for tensors in [0,1].

    x, y: [1,C,H,W] float
    returns: ssim_map [H,W] float32 on CPU
    """
    assert x.shape == y.shape and x.ndim == 4, (x.shape, y.shape)
    B, C, H, W = x.shape
    assert B == 1, "This helper expects batch size 1 for diagnostics."

    # build gaussian kernel for depthwise conv
    device, dtype = x.device, x.dtype
    k2d = _gaussian_kernel_2d(window_size, sigma, device, dtype)
    kernel = k2d.view(1, 1, window_size, window_size).repeat(C, 1, 1, 1)  # [C,1,ws,ws]
    pad = window_size // 2

    # local means
    mu_x = F.conv2d(x, kernel, padding=pad, groups=C)
    mu_y = F.conv2d(y, kernel, padding=pad, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    # local variances/covariance
    sigma_x2 = F.conv2d(x * x, kernel, padding=pad, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=pad, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=pad, groups=C) - mu_xy

    # SSIM constants
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # per-channel SSIM map
    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_c = num / (den + 1e-12)  # [1,C,H,W]

    # average over channels -> [H,W]
    ssim = ssim_c.mean(dim=1, keepdim=True)  # [1,1,H,W]
    ssim_hw = ssim[0, 0].detach().float().clamp(0.0, 1.0).cpu().numpy()
    return ssim_hw


def _save_gray_png(path: str, x01: np.ndarray):
    """x01: HxW float in [0,1]"""
    x8 = (np.clip(x01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(x8, mode="L").save(path)

def _save_rgb_png(path: str, x_uint8: np.ndarray):
    """x_uint8: HxWx3 uint8"""
    Image.fromarray(x_uint8).save(path)

def _blend_overlay(rgb_uint8: np.ndarray, heat01: np.ndarray, alpha=0.5):
    """
    Cheap red overlay without matplotlib colormaps:
      overlay = rgb*(1-alpha) + red*alpha*heat
    heat01: HxW in [0,1]
    """
    rgb = rgb_uint8.astype(np.float32)
    heat = np.clip(heat01, 0.0, 1.0).astype(np.float32)
    red = np.zeros_like(rgb)
    red[..., 0] = 255.0
    a = alpha * heat[..., None]
    out = rgb * (1.0 - a) + red * a
    return np.clip(out, 0, 255).astype(np.uint8)

@torch.no_grad()
def make_error_maps(restored: torch.Tensor, target: torch.Tensor, eps=1e-3):
    """
    restored/target: [1,3,H,W] float in [0,1]
    returns (abs_map, log_map): each [H,W] float in [0,1] (normalized later)
    """
    # absolute error where we get the mean over channels
    abs_err = (restored - target).abs().mean(dim=1, keepdim=True)  # [1,1,H,W]

    # log error where we get the mean over channels
    log_err = (torch.log(restored.clamp_min(eps)) - torch.log(target.clamp_min(eps))).abs().mean(dim=1, keepdim=True)

    # squeeze to HxW on CPU float32
    abs_map = abs_err[0, 0].detach().float().cpu().numpy()
    log_map = log_err[0, 0].detach().float().cpu().numpy()
    return abs_map, log_map

def robust_norm01(x: np.ndarray, q=0.99):
    """
    Map x to [0,1] using a robust max (quantile) so outliers don't blow contrast.
    """
    x = np.maximum(x, 0.0)
    m = np.quantile(x, q)
    if m <= 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip(x / m, 0.0, 1.0).astype(np.float32)


def _compute_stats(values: list) -> dict:
    """
    Compute publication-quality statistics from a list of per-image scalar values.
    Returns mean, std (sample), median, min, max, Q1, Q3, and 95% CI.
    """
    v = np.array(values, dtype=np.float64)
    n = len(v)
    if n == 0:
        return {}
    mean   = float(v.mean())
    std    = float(v.std(ddof=1)) if n > 1 else 0.0
    median = float(np.median(v))
    vmin   = float(v.min())
    vmax   = float(v.max())
    q25    = float(np.percentile(v, 25))
    q75    = float(np.percentile(v, 75))
    # 95% CI via t-distribution (scipy) or normal approximation fallback
    if n > 1:
        sem = std / (n ** 0.5)
        if _HAS_SCIPY:
            lo, hi = _scipy_stats.t.interval(0.95, df=n - 1, loc=mean, scale=sem)
        else:
            lo = mean - 1.96 * sem
            hi = mean + 1.96 * sem
        ci95_lo, ci95_hi = float(lo), float(hi)
    else:
        ci95_lo = ci95_hi = mean
    return {
        "n": n, "mean": mean, "std": std, "median": median,
        "min": vmin, "max": vmax, "q25": q25, "q75": q75,
        "ci95_lo": ci95_lo, "ci95_hi": ci95_hi,
    }



from PIL import Image

def save_png_uint8(result_dir: str, filename, hwc_uint8):
    os.makedirs(result_dir, exist_ok=True)
    base = os.path.basename(str(filename))
    if not base.lower().endswith(".png"):
        base = base + ".png"
    Image.fromarray(hwc_uint8).save(os.path.join(result_dir, base))


ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def _infer_single(model, input_, dino_feats, point, normal, opt, tile, overlap, orig_hw=None):
    """Run one forward pass (tiled or direct) and return a clamped [B,3,H,W] tensor."""
    # Use original (pre-padding) dimensions for the tiling decision so that
    # images which fit within tile_size are not incorrectly routed to tiled
    # inference just because padding rounded them up to the next multiple-of.
    _oh, _ow = orig_hw if orig_hw is not None else input_.shape[-2:]
    _th, _tw = tile if isinstance(tile, (list, tuple)) else (tile, tile)
    if tile is not None and (_oh > _th or _ow > _tw):
        out = validate_one_image_tiled(
            opt, model, input_,
            DINO_Mat_features=dino_feats,
            point=point, normal=normal,
            tile=tile, overlap=overlap,
        )
    else:
        out = model(input_, dino_feats, point, normal)
    return out.clamp(0.0, 1.0)


@torch.no_grad()
def _infer_with_tta(model, input_, dino_feats, point, normal, opt, tile, overlap, orig_hw=None):
    """Horizontal-flip TTA: average forward pass + flipped forward pass."""
    out_orig = _infer_single(model, input_, dino_feats, point, normal, opt, tile, overlap, orig_hw=orig_hw)

    input_f  = torch.flip(input_, dims=[-1])

    # Spatial flip of point and normal, then negate the X channel (channel 0)
    # because the horizontal coordinate reverses sign under a left-right mirror.
    point_f  = torch.flip(point,  dims=[-1]).clone()
    point_f[:, 0, :, :] = -point_f[:, 0, :, :]

    normal_f = torch.flip(normal, dims=[-1]).clone()
    normal_f[:, 0, :, :] = -normal_f[:, 0, :, :]

    # DINO features are spatial [1, C, H, W] — flip along width to stay consistent
    # with the flipped input. Without this, the flipped pass receives geometrically
    # inconsistent inputs (flipped RGB but unflipped DINO), degrading quality.
    dino_feats_f = (
        [torch.flip(f, dims=[-1]) for f in dino_feats]
        if dino_feats is not None else None
    )

    # kv_tokens are spatially pooled — not safe to flip spatially, pass as-is
    out_flip = _infer_single(model, input_f, dino_feats_f, point_f, normal_f, opt, tile, overlap, orig_hw=orig_hw)
    out_flip = torch.flip(out_flip, dims=[-1])

    return ((out_orig + out_flip) * 0.5).clamp(0.0, 1.0)


@torch.no_grad()
def _infer_ensemble(models, weights, input_, dino_feats, point, normal, opt, tile, overlap, use_tta, orig_hw=None):
    """Average predictions from multiple models, with optional TTA per model."""
    infer_fn = _infer_with_tta if use_tta else _infer_single
    total_w = sum(weights)
    acc = None
    for model, w in zip(models, weights):
        out = infer_fn(model, input_, dino_feats, point, normal, opt, tile, overlap, orig_hw=orig_hw)
        acc = out * (w / total_w) if acc is None else acc + out * (w / total_w)
    return acc.clamp(0.0, 1.0)


@torch.no_grad()
def validate(opt, tracker, mode="val"):
    if not torch.cuda.is_available():
        raise RuntimeError("Please use a device with GPU acceleration.")
    device = torch.device("cuda:0")

    # --- logging dirs (minimal) ---
    dir_name = opt.save_dir
    suffix = "_test" if mode == "test" else "_val"
    log_dir = os.path.join(dir_name, "log", opt.arch + opt.env + suffix)

    print(log_dir)
    result_dir = os.path.join(log_dir, "results")
    model_dir = os.path.join(log_dir, "models")
    utils.mkdir(log_dir); utils.mkdir(result_dir); utils.mkdir(model_dir)

    # --- model ---
    model = get_arch_sgcr(opt, ROOT).to(device)

    # --- ensemble: load additional checkpoints if specified ---
    # ensemble_configs (cross-arch): list of config yaml paths, each with pretrain_weights set.
    # ensemble_weights (same-arch):  list of checkpoint paths, all use current opt's arch.
    # If neither is set, single-model mode.
    ensemble_ckpts = opt.ensemble_weights  # list[str] or null — same-arch
    ensemble_cfgs  = opt.ensemble_configs  # list[str] or null — cross-arch
    use_ensemble = (
        (ensemble_cfgs  is not None and len(ensemble_cfgs)  > 1) or
        (ensemble_ckpts is not None and len(ensemble_ckpts) > 1)
    )
    extra_models = []
    use_tta = bool(opt.tta_hflip)

    # DINO (optional)
    DINO_Net = None
    if opt.use_dino:
        dino_name = opt.dino_name
        DINO_Net = torch.hub.load(str(ROOT / "OmniSR" / "dinov2"), dino_name, source="local").to(device).eval()
        for p in DINO_Net.parameters():
            p.requires_grad = False

    # --- load checkpoint (expects opt.pretrain_weights) ---
    ckpt = opt.pretrain_weights
    if ckpt is None:
        raise ValueError("Validation requires opt.pretrain_weights to be set.")
    load_checkpoint(model, ckpt, strict=False)

    # Load extra ensemble models
    if ensemble_cfgs is not None and len(ensemble_cfgs) > 1:
        # Cross-architecture mode.
        # Each entry is either:
        #   - a string: path to a config yaml that already has pretrain_weights set
        #   - a two-element list [config_path, ckpt_path]: loads arch from
        #     config_path and overrides pretrain_weights with ckpt_path.
        from config.config import CVConfig as _CVConfig
        for entry in ensemble_cfgs:
            if isinstance(entry, (list, tuple)):
                if len(entry) != 2:
                    raise ValueError(f"ensemble_configs list entries must be [config_path, ckpt_path], got: {entry!r}")
                cfg_file, m_ckpt = str(entry[0]), str(entry[1])
                member_opt = _CVConfig.build_from_configs(
                    cfg_file, {"pretrain_weights": m_ckpt},
                    do_not_merge_command_line=True
                )
            else:
                cfg_file = str(entry)
                member_opt = _CVConfig.build_from_configs(
                    cfg_file, do_not_merge_command_line=True
                )
                m_ckpt = member_opt.pretrain_weights
                if m_ckpt is None:
                    raise ValueError(f"ensemble_configs entry {cfg_file!r} has no pretrain_weights set.")
            # Reuse the already-loaded primary model if checkpoint matches
            if m_ckpt == ckpt:
                extra_models.append(model)
                continue
            m = get_arch_sgcr(member_opt, ROOT).to(device)
            load_checkpoint(m, m_ckpt, strict=False)
            m.eval()
            extra_models.append(m)
        all_models = extra_models
        all_weights = [1.0] * len(all_models)
    elif ensemble_ckpts is not None and len(ensemble_ckpts) > 1:
        # Same-architecture mode (legacy): build all members from current opt
        for ckpt_path in ensemble_ckpts:
            if ckpt_path == ckpt:
                extra_models.append(model)  # reuse already-loaded primary model
                continue
            m = get_arch_sgcr(opt, ROOT).to(device)
            load_checkpoint(m, ckpt_path, strict=False)
            m.eval()
            extra_models.append(m)
        all_models = extra_models
        all_weights = [1.0] * len(all_models)
    else:
        all_models = [model]
        all_weights = [1.0]

    # --- dataloader ---
    if mode == "test":
        val_dataset = DataLoaderTest(opt.val_dir, debug=opt.debug)
    else:
        val_dataset = get_validation_data(opt.val_dir, opt.debug)

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=opt.validation_batch_size,
        num_workers=opt.eval_workers,
        pin_memory=False,
        drop_last=False,
    )

    metrics = None
    if mode != "test":
        metrics = NTIREMetrics(device=device, enable_lpips=True, lpips_net="alex")
        metrics.reset()

    model.eval()
    if DINO_Net is not None:
        DINO_Net.eval()
    for m in extra_models:
        if m is not model:
            m.eval()

    overlap = opt.tile_overlap
    # tile_size may be an int (square) or a [h, w] list from the config
    _ts = opt.tile_size
    tile = (int(_ts[0]), int(_ts[1])) if isinstance(_ts, (list, tuple)) else int(_ts)
    DINO_patch_size = 14
    img_multiple_of = 8 * opt.win_size

    ssim_y_records_local = []   # list of (ssim_y_float, filename_str)  — Y-channel skimage/MATLAB convention
    psnr_records_local  = []   # list of (psnr_float,  filename_str)
    lpips_records_local = []   # list of (lpips_float, filename_str)
    rmse_records_local  = []   # list of (rmse_float,  filename_str)  — [0,255] scale
    # --- global average MS-SSIM accumulators ---
    ms_ssim_sum = torch.zeros(1, device=device, dtype=torch.float64)
    ms_ssim_count = torch.zeros(1, device=device, dtype=torch.float64)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for idx, data_val in enumerate(val_loader):
            #if idx > 25:
            #    break
            if len(data_val) == 5:
                target = data_val[0].to(device, non_blocking=True)
                input_  = data_val[1].to(device, non_blocking=True)
                point   = data_val[2].to(device, non_blocking=True)
                normal  = data_val[3].to(device, non_blocking=True)
                filenames = data_val[4]
            elif len(data_val) == 4:
                target = None
                input_  = data_val[0].to(device, non_blocking=True)
                point   = data_val[1].to(device, non_blocking=True)
                normal  = data_val[2].to(device, non_blocking=True)
                filenames = data_val[3]
            else:
                target = data_val[0].to(device, non_blocking=True)
                input_  = data_val[1].to(device, non_blocking=True)
                point   = data_val[2].to(device, non_blocking=True)
                normal  = data_val[3].to(device, non_blocking=True)
                filenames = data_val[4]

            height, width = input_.shape[2], input_.shape[3]
            if not getattr(opt, 'use_depth', True):
                point  = torch.zeros_like(point)
            if not getattr(opt, 'use_normal', True):
                normal = torch.zeros_like(normal)

            # pad to multiple-of — use proper ceiling division (matches training)
            H = ((height + img_multiple_of - 1) // img_multiple_of) * img_multiple_of
            W = ((width  + img_multiple_of - 1) // img_multiple_of) * img_multiple_of
            padh = H - height
            padw = W - width
            if padh or padw:
                input_  = F.pad(input_,  (0, padw, 0, padh), "replicate")
                point   = F.pad(point,   (0, padw, 0, padh), "replicate")
                normal  = F.pad(normal,  (0, padw, 0, padh), "replicate")

            # DINO feats
            dino_feats = None
            if DINO_Net is not None:
                input_DINO = F.interpolate(input_, scale_factor=(DINO_patch_size / 8), mode="bilinear", align_corners=False)
                input_DINO = maybe_imagenet_normalize_for_dino(
                    input_DINO, opt.dino_imagenet_norm
                )
                dino_feats = DINO_Net.get_intermediate_layers(input_DINO, 4, True)

            # infer (tiled if big) — with TTA and/or ensemble
            height_orig, width_orig = height, width  # already saved above
            _orig_hw = (height, width)
            if use_ensemble or use_tta:
                restored = _infer_ensemble(
                    all_models, all_weights,
                    input_, dino_feats, point, normal,
                    opt, tile, overlap, use_tta=use_tta, orig_hw=_orig_hw,
                )
            else:
                restored = _infer_single(model, input_, dino_feats, point, normal, opt, tile, overlap, orig_hw=_orig_hw)

            restored = restored[:, :, :height, :width]

            # ---- per-image metrics — same parameters as NTIREMetrics exactly ----
            # Optional resize before metrics (e.g. eval_resize=[256,256] for Xu et al. ISTD+ protocol)
            _eval_resize = getattr(opt, 'eval_resize', None)
            if _eval_resize is not None:
                _rh, _rw = int(_eval_resize[0]), int(_eval_resize[1])
                restored_m = F.interpolate(restored.float(), size=(_rh, _rw), mode='bilinear', align_corners=False)
                target_m   = F.interpolate(target.float(),   size=(_rh, _rw), mode='bilinear', align_corners=False) if target is not None else None
            else:
                restored_m = restored.float()
                target_m   = target.float() if target is not None else None

            B = restored.shape[0]
            for b in range(B):
                if target_m is not None:
                    ms = ms_ssim_fn(
                        restored_m[b:b+1],
                        target_m[b:b+1],
                        data_range=1.0,
                    )

                    ms_ssim_sum += ms.detach().to(torch.float64)
                    ms_ssim_count += 1.0

                    # per-image Y-channel SSIM — skimage / MATLAB Wang et al. convention:
                    # Y = 16 + 65.481*R + 128.553*G + 24.966*B, giving Y in [16, 235].
                    # gaussian_weights + use_sample_covariance=False matches MATLAB ssim().
                    def _to_y_np(t: torch.Tensor) -> np.ndarray:
                        w = torch.tensor([65.481, 128.553, 24.966],
                                         device=t.device, dtype=t.dtype).view(1, 3, 1, 1)
                        return ((t * w).sum(dim=1, keepdim=True) + 16.0)[0, 0].cpu().numpy()

                    sy = ski_ssim(
                        _to_y_np(restored_m[b:b+1]),
                        _to_y_np(target_m[b:b+1]),
                        data_range=255.0,
                        gaussian_weights=True,
                        sigma=1.5,
                        use_sample_covariance=False,
                    )
                    ssim_y_records_local.append((float(sy), str(filenames[b])))
                    pv = psnr_fn(
                        restored_m[b:b+1],
                        target_m[b:b+1],
                        data_range=1.0,
                    )
                    psnr_records_local.append((float(pv.item()), str(filenames[b])))

                    # per-image LPIPS (only when the metric object has lpips_fn loaded)
                    if metrics is not None and getattr(metrics, "lpips_fn", None) is not None:
                        p_img = restored_m[b:b+1] * 2.0 - 1.0
                        t_img = target_m[b:b+1] * 2.0 - 1.0
                        lp = metrics.lpips_fn(p_img, t_img).mean()
                        lpips_records_local.append((float(lp.item()), str(filenames[b])))

                    # per-image RMSE on [0, 255] scale (comparable to baseline papers)
                    rmse_val = torch.sqrt(
                        ((restored_m[b:b+1] * 255.0 - target_m[b:b+1] * 255.0) ** 2).mean()
                    )
                    rmse_records_local.append((float(rmse_val.item()), str(filenames[b])))


            if target is not None:
                base = os.path.splitext(os.path.basename(str(filenames[0])))[0]
                out_dir = os.path.join(result_dir, "diagnostics")
                os.makedirs(out_dir, exist_ok=True)

                # diagnostics operate on single images — use the first in the batch
                restored_d = restored[0:1]
                target_d   = target[0:1]

                abs_map, log_map = make_error_maps(restored_d, target_d)

                abs01 = robust_norm01(abs_map, q=0.99)
                log01 = robust_norm01(log_map, q=0.99)

                _save_gray_png(os.path.join(out_dir, f"{base}_abs.png"), abs01)
                _save_gray_png(os.path.join(out_dir, f"{base}_log.png"), log01)

                restored_u8 = to_hwc_rgb_uint8(restored_d[0])
                overlay_u8 = _blend_overlay(restored_u8, log01, alpha=0.6)
                _save_rgb_png(os.path.join(out_dir, f"{base}_log_overlay.png"), overlay_u8)

                # --- SSIM map diagnostics ---
                ssim01 = ssim_map_torch(restored_d, target_d, window_size=11, sigma=1.5)
                ssim_err01 = 1.0 - ssim01  # high = bad

                # Save raw SSIM map (bright=good)
                _save_gray_png(os.path.join(out_dir, f"{base}_ssim.png"), ssim01)

                # Save SSIM error map (bright=bad); robust-normalize for visibility
                ssim_err_vis = robust_norm01(ssim_err01, q=0.99)
                _save_gray_png(os.path.join(out_dir, f"{base}_ssim_err.png"), ssim_err_vis)

                # Overlay SSIM error on restored
                restored_u8 = to_hwc_rgb_uint8(restored_d[0])
                ssim_overlay_u8 = _blend_overlay(restored_u8, ssim_err_vis, alpha=0.6)
                _save_rgb_png(os.path.join(out_dir, f"{base}_ssim_err_overlay.png"), ssim_overlay_u8)





            for name, img in zip(filenames, restored):
                save_png_uint8(result_dir, name, to_hwc_rgb_uint8(img))

            if metrics is not None and target_m is not None:
                metrics.update(restored_m, target_m)

            # optional: log images
            if tracker is not None:
                for name, image in zip(filenames, restored):
                    tracker.log_image(name=str(name), image=to_hwc_rgb_uint8(image), maximum_per_step=50)
    avg_ms_ssim = (ms_ssim_sum / ms_ssim_count.clamp_min(1.0)).item()
    print(f"Avg MS-SSIM: {avg_ms_ssim:.6f}")
    if tracker is not None:
        tracker.log_scalar("val/avg_ms_ssim", avg_ms_ssim)

    if tracker is not None and metrics is not None:
        result = metrics.log(tracker, prefix=("val" if mode == "val" else "test"))
        print("\n========== Validation Metrics ==========")
        for k, v in sorted(result.items()):
            print(f"  {k}: {v:.6f}")
        print("========================================\n")
    
    # ---- per-image records ----
    all_psnr_records   = psnr_records_local
    all_lpips_records  = lpips_records_local
    all_rmse_records   = rmse_records_local
    all_ssim_y_records = ssim_y_records_local

    # ---- publication-quality per-image statistics ----
    if True:
        metric_data = {
            "RMSE"      : [v for v, _ in all_rmse_records],
            "PSNR (dB)" : [v for v, _ in all_psnr_records],
            "SSIM-Y"    : [v for v, _ in all_ssim_y_records],
            "LPIPS"     : [v for v, _ in all_lpips_records],
        }
        pfx = "val" if mode == "val" else "test"
        header = f"{'Metric':<14} {'N':>5}  {'Mean':>8}  {'Std':>8}  {'Median':>8}  {'Min':>8}  {'Max':>8}  {'Q1':>8}  {'Q3':>8}  {'95% CI':>22}"
        print("\n========== Per-image Statistics (publication) ==========")
        print(header)
        print("-" * len(header))
        for name, vals in metric_data.items():
            if not vals:
                continue
            st = _compute_stats(vals)
            ci_str = f"[{st['ci95_lo']:+.4f}, {st['ci95_hi']:+.4f}]"
            print(
                f"{name:<14} {st['n']:>5}  "
                f"{st['mean']:>8.4f}  {st['std']:>8.4f}  {st['median']:>8.4f}  "
                f"{st['min']:>8.4f}  {st['max']:>8.4f}  "
                f"{st['q25']:>8.4f}  {st['q75']:>8.4f}  {ci_str:>22}"
            )
            if tracker is not None:
                safe = name.split()[0].lower().replace('-', '_')  # "psnr", "ssim", "ssim_y", "rmse", "lpips"
                for stat_name, stat_val in [
                    ("mean",   st["mean"]),
                    ("std",    st["std"]),
                    ("median", st["median"]),
                    ("min",    st["min"]),
                    ("max",    st["max"]),
                    ("q25",    st["q25"]),
                    ("q75",    st["q75"]),
                    ("ci95_lo", st["ci95_lo"]),
                    ("ci95_hi", st["ci95_hi"]),
                ]:
                    tracker.log_scalar(f"{pfx}/{safe}_{stat_name}", stat_val)
        print("=" * len(header) + "\n")




def maybe_imagenet_normalize_for_dino(x: torch.Tensor, enable: bool) -> torch.Tensor:
    """
    x: [B,3,H,W] float tensor in [0,1]
    Apply ImageNet mean/std normalization if enabled.
    """
    if not enable:
        return x
    assert x.ndim == 4 and x.shape[1] == 3, x.shape
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std

class DataLoaderTest(Dataset):
    """
    Test loader for splits with NO ground-truth (no shadow_free/).
    Expects:
      rgb_dir/origin/*.png
      rgb_dir/depth/*.npy or whatever load_depth expects
      rgb_dir/normal/*.png or whatever load_normal expects
    Returns:
      noisy, point, normal, noisy_filename
    """
    def __init__(self, rgb_dir, target_transform=None, debug=False):
        super(DataLoaderTest, self).__init__()

        self.target_transform = target_transform

        input_dir = 'origin'
        depth_dir = 'depth'
        normal_dir = 'normal'

        noisy_files  = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        depth_files  = sorted(os.listdir(os.path.join(rgb_dir, depth_dir)))
        normal_files = sorted(os.listdir(os.path.join(rgb_dir, normal_dir)))

        self.noisy_filenames  = [os.path.join(rgb_dir, input_dir,  x) for x in noisy_files]
        self.depth_filenames  = [os.path.join(rgb_dir, depth_dir,  x) for x in depth_files]
        self.normal_filenames = [os.path.join(rgb_dir, normal_dir, x) for x in normal_files]

        if debug:
            self.tar_size = min(10, len(self.noisy_filenames))
        else:
            self.tar_size = len(self.noisy_filenames)

        # Basic sanity: ensure we have matching counts
        if not (len(self.noisy_filenames) == len(self.depth_filenames) == len(self.normal_filenames)):
            raise ValueError(
                f"File count mismatch in {rgb_dir}: "
                f"origin={len(self.noisy_filenames)}, depth={len(self.depth_filenames)}, normal={len(self.normal_filenames)}"
            )

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size

        noisy  = np.float32(load_img(self.noisy_filenames[tar_index]))
        depth  = np.float32(load_depth(self.depth_filenames[tar_index]))
        normal = np.float32(load_normal(self.normal_filenames[tar_index]))

        point = depthToPoint(60, depth)
        normal = process_normal(normal)
        point = point / (2 * point[:, :, 2].mean())

        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        noisy  = torch.from_numpy(noisy).permute(2, 0, 1)
        point  = torch.from_numpy(point).permute(2, 0, 1)
        normal = torch.from_numpy(normal).permute(2, 0, 1)

        return noisy, point, normal, noisy_filename