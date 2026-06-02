import torch
import torch.nn.functional as F

def _to_luma(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 3:
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        return 0.2989 * r + 0.5870 * g + 0.1140 * b
    return x

def gradL1Loss(pred: torch.Tensor, tgt: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
    """
    First-derivative alignment loss. If w is provided (B1HW), it weights spatially.
    Computes on luma for stability.
    """
    pred = _to_luma(pred)
    tgt  = _to_luma(tgt)

    gx_p = pred[..., :, 1:] - pred[..., :, :-1]
    gy_p = pred[..., 1:, :] - pred[..., :-1, :]
    gx_t = tgt[..., :, 1:] - tgt[..., :, :-1]
    gy_t = tgt[..., 1:, :] - tgt[..., :-1, :]

    if w is not None:
        if w.ndim == 3:
            w = w.unsqueeze(1)
        # match shapes
        wx = w[..., :, 1:]
        wy = w[..., 1:, :]
        loss_x = (wx * (gx_p - gx_t).abs()).mean()
        loss_y = (wy * (gy_p - gy_t).abs()).mean()
        return loss_x + loss_y

    return (gx_p - gx_t).abs().mean() + (gy_p - gy_t).abs().mean()
