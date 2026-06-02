from __future__ import annotations
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from sgcr.losses.grad_loss import gradL1Loss
from sgcr.losses.hessian_loss import hessianLoss


def grad_mag(x):
    # simple finite diffs; x is BCHW
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    # pad back to size
    dx = F.pad(dx, (0,1,0,0))
    dy = F.pad(dy, (0,0,0,1))
    return torch.abs(dx) + torch.abs(dy)


# ---------------------------------------------------------------------------
# Solution 2 helper: separable Gaussian blur for progressive LPF supervision
# ---------------------------------------------------------------------------
def _gaussian_kernel_1d(sigma: float, device, dtype) -> torch.Tensor:
    r = max(int(3.0 * sigma + 0.5), 1)
    ks = 2 * r + 1
    xs = torch.arange(ks, device=device, dtype=dtype) - r
    k = torch.exp(-0.5 * (xs / sigma) ** 2)
    return k / k.sum()


def _blur_target(target: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur on BCHW target (sigma in pixels). Returns same shape."""
    if sigma <= 0.0:
        return target
    k = _gaussian_kernel_1d(sigma, target.device, target.dtype)
    ks = k.shape[0]
    pad = ks // 2
    C = target.shape[1]
    # depthwise separable convolution: H then W
    k_h = k.view(1, 1, ks, 1).expand(C, 1, ks, 1).contiguous()
    k_w = k.view(1, 1, 1, ks).expand(C, 1, 1, ks).contiguous()
    t = F.pad(target, (0, 0, pad, pad), mode='reflect')
    t = F.conv2d(t, k_h, groups=C)
    t = F.pad(t, (pad, pad, 0, 0), mode='reflect')
    t = F.conv2d(t, k_w, groups=C)
    return t


@dataclass(frozen=True)
class LossPack:
    total: torch.Tensor
    terms: Dict[str, torch.Tensor]


def compute_training_losses(
    *,
    opt,
    model_restoration,
    restored: torch.Tensor,
    target: torch.Tensor,
    input_: torch.Tensor,
    criterion_restore,
    lpips_loss_fn,
    ssim_loss_fn,
) -> LossPack:
    """
    Required:
      - opt must provide all referenced lambdas (lam_tv, lam_pseudo,
        lambda_lpips, lambda_hessian, lambda_alpha, lambda_id, lambda_bgrad,
        lambda_grad, lambda_stage).
    """
    # ---- base restoration loss ----
    loss_restore = criterion_restore(restored, target)

    m = model_restoration.module if hasattr(model_restoration, "module") else model_restoration

    terms: Dict[str, torch.Tensor] = {}
    terms["restore"] = loss_restore

    terms["reg"] = restored.new_tensor(0.0)

    # Start with weighted restore + reg (your 0.9 factor)
    loss = 0.9 * loss_restore

    terms["pseudo"] = restored.new_tensor(0.0)

    # ---- LPIPS ----
    loss_lpips = restored.new_tensor(0.0)
    if opt.lambda_lpips > 0.0:
        loss_lpips = lpips_loss_fn(restored, target)
        loss = loss + opt.lambda_lpips * loss_lpips
    terms["lpips"] = loss_lpips
    

    # ---- SSIM (optional) ----
    loss_ssim = restored.new_tensor(0.0)
    if (opt.lambda_ssim or 0) > 0:
        loss_ssim = ssim_loss_fn(restored, target)
        loss = loss + opt.lambda_ssim * loss_ssim
    terms["ssim"] = loss_ssim

    # ---- Hessian (optional) ----
    loss_hess = restored.new_tensor(0.0)
    if (opt.lambda_hessian or 0) > 0:
        loss_hess = hessianLoss(restored, target)
        loss = loss + opt.lambda_hessian * loss_hess
    terms["hessian"] = loss_hess

    # ---- Gradient alignment (optional) ----
    loss_grad = restored.new_tensor(0.0)
    if (opt.lambda_grad or 0) > 0:
        loss_grad = gradL1Loss(restored, target)
        loss = loss + opt.lambda_grad * loss_grad
    terms["grad"] = loss_grad

    # ---- multi-stage supervision (apply to each intermediate stage) ----
    # lambda_stage takes precedence; fall back to legacy lambda_stage1 for old configs.
    lam_stage = opt.lambda_stage
    if lam_stage is None:
        lam_stage = opt.lambda_stage1
    lam_stage = float(lam_stage) if lam_stage is not None else 0.0

    loss_stage_sum = restored.new_tensor(0.0)

    if lam_stage > 0.0:
        ys = None
        if hasattr(m, "ys") and isinstance(m.ys, (list, tuple)) and len(m.ys) > 0:
            ys = list(m.ys)
        elif hasattr(m, "y1") and m.y1 is not None:
            # Fallback for legacy 2-stage models that only expose y1/y2
            ys = [m.y1]

        if ys is not None:
            # We supervise intermediate outputs only; final output is already supervised via `restored`.
            # If the wrapper stores all stage outputs in m.ys (length == n_stage),
            # then m.ys[-1] is the final output.
            ys_inter = ys[:-1] if len(ys) > 1 else ys
            n_inter = len(ys_inter)

            # Per-stage loss weights.
            # Priority: dynamic_stage_weights > lambda_stage_weights > uniform 1.0.
            if opt.dynamic_stage_weights and n_inter > 1:
                # Compute each intermediate stage's loss against full-frequency GT (no LPF)
                # then downweight stages with higher loss so they don't drown backbone gradients.
                with torch.no_grad():
                    raw = torch.stack([
                        criterion_restore(y_si, target)
                        for y_si in ys_inter if y_si is not None
                    ])
                    # mean / loss_i: harder stage gets lower weight
                    stage_weights = (raw.mean() / (raw + 1e-8)).tolist()
            else:
                stage_weights_cfg = opt.lambda_stage_weights
                if (stage_weights_cfg is not None
                        and isinstance(stage_weights_cfg, (list, tuple))
                        and len(stage_weights_cfg) == n_inter):
                    stage_weights = [float(w) for w in stage_weights_cfg]
                else:
                    stage_weights = [1.0] * n_inter

            # Solution 2: progressive LPF supervision.
            # sigma_max > 0 blurs the supervision target for early intermediate stages,
            # forcing them to produce coarse illumination corrections while later stages
            # are supervised against the full-frequency GT.
            # sigma decreases linearly: sigma_max (stage 0) -> 0 (last intermediate).
            sigma_max = float(opt.lambda_stage_lpf_sigma_max)

            for si, y_si in enumerate(ys_inter):
                if y_si is None:
                    continue
                if sigma_max > 0.0:
                    # Linear schedule: full sigma at si=0, zero at si=n_inter-1
                    sigma_k = sigma_max * (1.0 - si / max(n_inter - 1, 1)) if n_inter > 1 else sigma_max
                    tgt_k = _blur_target(target, sigma_k)
                else:
                    tgt_k = target
                li = criterion_restore(y_si, tgt_k)
                terms[f"stage{si}"] = li
                loss_stage_sum = loss_stage_sum + stage_weights[si] * li

            # Only add if we actually accumulated something
            loss = loss + lam_stage * loss_stage_sum

    terms["stage"] = loss_stage_sum

    # ---- Solution 4: Contraction constraint loss ----
    # Penalises any stage k that is *further* from GT than stage k-1 (hinge on L1).
    # Forces monotone improvement without enforcing an exact schedule.
    lambda_contraction = float(opt.lambda_contraction)
    loss_contraction = restored.new_tensor(0.0)
    if lambda_contraction > 0.0:
        ys_all = None
        if hasattr(m, "ys") and isinstance(m.ys, (list, tuple)) and len(m.ys) > 1:
            ys_all = list(m.ys)
        if ys_all is not None:
            for k in range(1, len(ys_all)):
                yk = ys_all[k]
                yk_prev = ys_all[k - 1]
                if yk is None or yk_prev is None:
                    continue
                dist_k    = F.l1_loss(yk, target)
                dist_prev = F.l1_loss(yk_prev.detach(), target)
                # Hinge: only penalise regression (positive = worse than previous)
                violation = torch.clamp(dist_k - dist_prev, min=0.0)
                loss_contraction = loss_contraction + violation
        loss = loss + lambda_contraction * loss_contraction
    terms["contraction"] = loss_contraction

    terms["total"] = loss

    return LossPack(total=loss, terms=terms)

