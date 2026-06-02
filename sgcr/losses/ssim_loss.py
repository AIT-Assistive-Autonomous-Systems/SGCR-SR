from __future__ import annotations

import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure


def _to_luma(x: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB (B,3,H,W) -> luma (B,1,H,W).
    If already single-channel, returns as-is.
    Assumes x is linear in the same space as your training targets.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected NCHW tensor, got shape {tuple(x.shape)}")
    if x.shape[1] == 3:
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        return 0.2989 * r + 0.5870 * g + 0.1140 * b
    return x


class SSIMLoss(nn.Module):
    """
    Uses torchmetrics StructuralSimilarityIndexMeasure so it's the same SSIM
    as in NTIREMetrics.
    Loss = 1 - SSIM.
    """
    def __init__(self, data_range: float = 1.0):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # torchmetrics is stateful-ish; ensure it's on same device
        self.ssim = self.ssim.to(pred.device)
        pred = pred.float()
        target = target.float()
        val = self.ssim(pred, target)   # scalar (batch-reduced)
        return 1.0 - val


class MSSSIMLoss(nn.Module):
    """
    MS-SSIM loss using torchmetrics, optionally computed on luma only.
    Loss = 1 - MS-SSIM.
    This is often a good lever to increase SSIM without fighting valid chroma shifts
    in relighting/shadow-removal datasets.
    """
    def __init__(self, data_range: float = 1.0, on_luma: bool = True):
        super().__init__()
        self.on_luma = on_luma
        self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=data_range)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.ms_ssim = self.ms_ssim.to(pred.device)
        pred = pred.float()
        target = target.float()

        if self.on_luma:
            pred = _to_luma(pred)
            target = _to_luma(target)

        val = self.ms_ssim(pred, target)  # scalar (batch-reduced)
        return 1.0 - val