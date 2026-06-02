import torch
import torch.nn.functional as F

def _to_luma(x: torch.Tensor) -> torch.Tensor:
    """Convert BCHW RGB to single-channel luma (B1HW). If not RGB, return as-is."""
    if x.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got {tuple(x.shape)}")
    if x.shape[1] == 3:
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        return 0.2989 * r + 0.5870 * g + 0.1140 * b
    return x


def hessianLoss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Hessian loss (second derivatives) between recon and target, computed on luma."""
    eps = 1e-3  # Charbonnier epsilon (fixed)
    recon = _to_luma(recon)
    target = _to_luma(target)

    # second differences (discrete d²/dx² and d²/dy²)
    ddx_r = recon[..., :, 2:] - 2.0 * recon[..., :, 1:-1] + recon[..., :, :-2]
    ddx_t = target[..., :, 2:] - 2.0 * target[..., :, 1:-1] + target[..., :, :-2]
    ddy_r = recon[..., 2:, :] - 2.0 * recon[..., 1:-1, :] + recon[..., :-2, :]
    ddy_t = target[..., 2:, :] - 2.0 * target[..., 1:-1, :] + target[..., :-2, :]

    # pad back to original spatial size
    ddx_r = F.pad(ddx_r, (1, 1, 0, 0))
    ddx_t = F.pad(ddx_t, (1, 1, 0, 0))
    ddy_r = F.pad(ddy_r, (0, 0, 1, 1))
    ddy_t = F.pad(ddy_t, (0, 0, 1, 1))

    # Charbonnier penalty for robustness
    loss_x = torch.sqrt((ddx_r - ddx_t) ** 2 + eps * eps).mean()
    loss_y = torch.sqrt((ddy_r - ddy_t) ** 2 + eps * eps).mean()
    return loss_x + loss_y