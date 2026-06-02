# ntire_metrics.py
# Helper utilities to compute & TensorBoard-log NTIRE-style metrics.
#
# Metrics:
#   - PSNR (torchmetrics)
#   - SSIM (torchmetrics)
#   - LPIPS (Alex/VGG/Squeeze via lpips; NTIRE commonly uses "alex")
#   - Param count (M) as an efficiency proxy
#
# Notes:
#   - MOS/user-study cannot be computed offline.
#   - Works in single-GPU mode.
#
#
# Example usage (DDP):
#   from ntire_metrics import NTIREMetrics
#   metrics = NTIREMetrics(device=device, enable_lpips=True, lpips_net="alex")
#   metrics.reset()
#   for ...:
#       metrics.update(restored, target)  # both in [0,1], BCHW

#  add any new metrics below
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.distributed as dist

try:
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
except Exception as e:
    raise ImportError(
        "torchmetrics is required. Install with: pip install torchmetrics"
    ) from e

try:
    import lpips  # type: ignore
except Exception:
    lpips = None  # optional


def _ddp_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


@dataclass
class NTIREMetrics:
    device: torch.device
    enable_lpips: bool = True
    lpips_net: str = "alex"  # "alex" / "vgg" / "squeeze"
    data_range: float = 1.0

    def __post_init__(self) -> None:
        self.psnr_fn = PeakSignalNoiseRatio(data_range=self.data_range).to(self.device)
        self.ssim_fn = StructuralSimilarityIndexMeasure(data_range=self.data_range).to(self.device)

        self.lpips_fn = None
        if self.enable_lpips:
            if lpips is None:
                raise ImportError("lpips is required for LPIPS. Install with: pip install lpips")
            self.lpips_fn = lpips.LPIPS(net=self.lpips_net).to(self.device).eval()

        self.reset()

    def reset(self) -> None:
        # running sums (on device for easy all_reduce)
        self._sum_psnr = torch.zeros((), device=self.device)
        self._sum_ssim = torch.zeros((), device=self.device)
        self._sum_lpips = torch.zeros((), device=self.device) if self.lpips_fn is not None else None
        self._count = torch.zeros((), device=self.device)  # number of images accumulated
        # RMSE accumulators (pixel-level, [0,255] scale)
        self._sum_sq_err = torch.zeros((), device=self.device)
        self._sum_pixels = torch.zeros((), device=self.device)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """
        pred, target: BCHW tensors in [0, 1].
        Accumulates per-image metrics into running sums.
        """
        if pred.ndim != 4 or target.ndim != 4:
            raise ValueError(f"Expected BCHW tensors, got pred {pred.shape}, target {target.shape}")
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

        # Convert to float32 for numerical stability in metrics
        p = pred.detach().float()
        t = target.detach().float()

        # Compute metrics (torchmetrics returns batch-reduced scalar by default)
        psnr_val = self.psnr_fn(p, t)
        ssim_val = self.ssim_fn(p, t)

        self._sum_psnr += psnr_val
        self._sum_ssim += ssim_val

        if self.lpips_fn is not None:
            # LPIPS expects inputs in [-1, 1]
            lp = self.lpips_fn(p * 2.0 - 1.0, t * 2.0 - 1.0)
            # lpips returns shape [B,1,1,1] or [B,1]; take mean over batch
            lp = lp.mean()
            self._sum_lpips += lp

        # RMSE: accumulate squared error and pixel count (scale to [0, 255])
        self._sum_sq_err += ((p * 255.0 - t * 255.0) ** 2).sum()
        self._sum_pixels += torch.tensor(float(p.numel()), device=self.device)

        # count images (batch size)
        self._count += torch.tensor(float(p.shape[0]), device=self.device)

    @torch.no_grad()
    def compute(self) -> Dict[str, float]:
        """Compute (DDP-averaged) mean metrics over all updates."""
        # Reduce sums and counts across ranks
        sum_psnr = self._sum_psnr.clone()
        sum_ssim = self._sum_ssim.clone()
        count = self._count.clone()
        sum_sq_err = self._sum_sq_err.clone()
        sum_pixels = self._sum_pixels.clone()

        if _ddp_is_initialized():
            dist.all_reduce(sum_psnr, op=dist.ReduceOp.SUM)
            dist.all_reduce(sum_ssim, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
            dist.all_reduce(sum_sq_err, op=dist.ReduceOp.SUM)
            dist.all_reduce(sum_pixels, op=dist.ReduceOp.SUM)

        count_val = max(count.item(), 1.0)

        out = {
            "psnr": (sum_psnr / count_val).item(),
            "ssim": (sum_ssim / count_val).item(),
            "rmse": (sum_sq_err / sum_pixels.clamp(min=1.0)).sqrt().item(),
        }

        if self._sum_lpips is not None:
            sum_lpips = self._sum_lpips.clone()
            if _ddp_is_initialized():
                dist.all_reduce(sum_lpips, op=dist.ReduceOp.SUM)
            out[f"lpips_{self.lpips_net}"] = (sum_lpips / count_val).item()

        return out

    @torch.no_grad()
    def log(self, tracker, prefix: str = "val") -> Dict[str, float]:
        """
        Log computed metrics to TensorBoard.
        """
        metrics = self.compute()
        tracker.log_scalars(metrics, sub_logger=prefix)
        return metrics

    @staticmethod
    def param_count_m(model: torch.nn.Module) -> float:
        """Return number of parameters in millions."""
        return sum(p.numel() for p in model.parameters()) / 1e6

    @staticmethod
    def log_param_count(tracker, model: torch.nn.Module, tag: str = "efficiency/params_M") -> float:
        """Log parameter count (M) once to TensorBoard."""
        n_params_m = NTIREMetrics.param_count_m(model)
        tracker.log_scalar(tag, n_params_m)
        return n_params_m
