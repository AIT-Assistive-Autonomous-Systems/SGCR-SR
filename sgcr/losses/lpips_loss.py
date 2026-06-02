from __future__ import annotations
import torch
import torch.nn as nn
import lpips


class LPIPSLoss(nn.Module):
    """
    LPIPS perceptual loss wrapper.
    - Expects inputs in [0,1] by default (will map to [-1,1] for LPIPS)
    - Returns a scalar tensor (mean over batch).
    """
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        if opt.lambda_lpips <= 0.0:
            # Disabled mode: keep a tiny module that returns 0
            self.lpips = None
            return

        self.lpips = lpips.LPIPS(net=opt.net or "vgg")
        self.lpips.eval()
        for p in self.lpips.parameters():
            p.requires_grad_(False)

    @property
    def enabled(self) -> bool:
        return self.lpips is not None and self.opt.lambda_lpips > 0.0

    def forward(self, pred_01: torch.Tensor, tgt_01: torch.Tensor) -> torch.Tensor:
        """
        pred_01, tgt_01: [B,3,H,W] in [0,1] (recommended)
        """
        if not self.enabled:
            # preserve device/dtype; return scalar 0
            return pred_01.new_zeros(())

        if self.opt.lpips_clamp_01:
            pred_01 = pred_01.clamp(0.0, 1.0)
            tgt_01 = tgt_01.clamp(0.0, 1.0)

        # LPIPS expects [-1, 1]
        pred = pred_01 * 2.0 - 1.0
        tgt  = tgt_01 * 2.0 - 1.0

        if self.opt.lpips_compute_in_fp32:
            # Run LPIPS in fp32 for stability (safe under AMP)
            with torch.autocast(device_type="cuda", enabled=False):
                return self.lpips(pred.float(), tgt.float()).mean()
        else:
            return self.lpips(pred, tgt).mean()
