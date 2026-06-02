import math
import torch
import torch.nn.functional as F

def _cosine_window_2d(h: int, w: int, device, dtype):
    if h == 1:
        wy = torch.ones((1,), device=device, dtype=dtype)
    else:
        y = torch.arange(h, device=device, dtype=dtype)
        wy = 0.5 - 0.5 * torch.cos(2.0 * math.pi * y / (h - 1))

    if w == 1:
        wx = torch.ones((1,), device=device, dtype=dtype)
    else:
        x = torch.arange(w, device=device, dtype=dtype)
        wx = 0.5 - 0.5 * torch.cos(2.0 * math.pi * x / (w - 1))

    win = wy[:, None] * wx[None, :]
    return win.clamp_min(1e-6)


@torch.no_grad()
def infer_tiled_ctx(
    x: torch.Tensor,
    model,
    forward_fn,              # forward_fn(model, patch, y0, y1, x0, x1, ctx) -> pred [B, C, ph, pw]
    ctx: dict,               # context object (precomputed pads/features/strides/etc.)
    tile: "int | tuple[int, int]" = 512,
    overlap: int = 128,
    out_channels: int = 3,
    use_amp: bool = True,
    pad_if_smaller: bool = True,
):
    """
    Sliding-window inference with overlap + cosine blending, using an explicit context dict.
    No nested functions required.

    Args:
        x: [B,C,H,W], assumes B==1.
        model: model callable.
        forward_fn: top-level callable(model, patch, y0, y1, x0, x1, ctx) -> pred [B,out_channels,ph,pw]
        ctx: dict with any auxiliary tensors/flags needed by forward_fn.
        tile: int (square) or (tile_h, tile_w) tuple for rectangular tiles.
        overlap/out_channels/use_amp/pad_if_smaller: same meaning as before.

    Returns:
        out: [B,out_channels,H,W]
    """
    assert x.ndim == 4, f"expected BCHW, got {x.shape}"
    B, _, H, W = x.shape
    assert B == 1, "This tiled inference assumes B==1 (typical for validation)."

    if isinstance(tile, (list, tuple)):
        tile_h, tile_w = int(tile[0]), int(tile[1])
    else:
        tile_h = tile_w = int(tile)
    overlap = int(overlap)
    stride_h = tile_h - overlap
    stride_w = tile_w - overlap
    if stride_h <= 0 or stride_w <= 0:
        raise ValueError("tile dimensions must be > overlap")

    device = x.device
    dtype = x.dtype

    out = torch.zeros((B, out_channels, H, W), device=device, dtype=dtype)
    wgt = torch.zeros((B, 1, H, W), device=device, dtype=dtype)

    full_win = _cosine_window_2d(tile_h, tile_w, device=device, dtype=dtype)[None, None, :, :]

    y_starts = list(range(0, H, stride_h))
    x_starts = list(range(0, W, stride_w))

    if y_starts and (y_starts[-1] + tile_h < H):
        y_starts.append(H - tile_h)
    if x_starts and (x_starts[-1] + tile_w < W):
        x_starts.append(W - tile_w)

    y_starts = [max(0, min(y, H - tile_h)) for y in y_starts] if H >= tile_h else [0]
    x_starts = [max(0, min(x0, W - tile_w)) for x0 in x_starts] if W >= tile_w else [0]

    pad_bottom = max(0, tile_h - H)
    pad_right = max(0, tile_w - W)
    if pad_if_smaller and (pad_bottom > 0 or pad_right > 0):
        x_pad = F.pad(x, (0, pad_right, 0, pad_bottom), mode="reflect")
    else:
        x_pad = x

    for y0 in y_starts:
        for x0 in x_starts:
            y1 = y0 + tile_h
            x1 = x0 + tile_w

            patch = x_pad[:, :, y0:y1, x0:x1]

            valid_y1 = min(y1, H)
            valid_x1 = min(x1, W)

            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    pred = forward_fn(model, patch, y0, valid_y1, x0, valid_x1, ctx)
            else:
                pred = forward_fn(model, patch, y0, valid_y1, x0, valid_x1, ctx)

            # allow model to return (pred, aux) etc.
            if isinstance(pred, (tuple, list)) and len(pred) > 0:
                pred = pred[0]

            pred_h = min(tile_h, H - y0)
            pred_w = min(tile_w, W - x0)
            pred = pred[:, :, :pred_h, :pred_w]

            win = full_win[:, :, :pred_h, :pred_w]

            out[:, :, y0:y0 + pred_h, x0:x0 + pred_w] += pred * win
            wgt[:, :, y0:y0 + pred_h, x0:x0 + pred_w] += win

    out = out / wgt.clamp_min(1e-6)
    return out


def forward_tiled_shadowformer_ctx(model, patch, y0, y1, x0, x1, ctx):
    """
    model: your ShadowFormer-like model
    patch: [1,C,tile,tile] taken from padded input
    y0/y1/x0/x1: valid region coordinates in original (unpadded) image space
    ctx: dict containing pre-padded geom tensors, dino features, strides, latents, etc.
    """
    patch_rgb = patch[:, :3, :, :]

    ph, pw = patch.shape[-2], patch.shape[-1]

    point_pad = ctx["point_pad"]
    normal_pad = ctx["normal_pad"]

    p_patch = point_pad[:, :, y0:y0 + ph, x0:x0 + pw] if point_pad is not None else None
    n_patch = normal_pad[:, :, y0:y0 + ph, x0:x0 + pw] if normal_pad is not None else None

    use_dino = ctx["use_dino"]
    dino_full = ctx["dino_full"]
    dino_stride = ctx["dino_stride"]

    if use_dino:
        dino_patch = [_crop_dino_feat(f, y0, y1, x0, x1, dino_stride) for f in dino_full]
    else:
        dino_patch = None

    pred = model(
        patch_rgb,
        DINO_Mat_features=dino_patch,
        point=p_patch,
        normal=n_patch,
    )
    if isinstance(pred, (tuple, list)) and len(pred) > 0:
        pred = pred[0]
    return pred


@torch.no_grad()
def validate_one_image_tiled(
    opt,
    model,
    input_,
    DINO_Mat_features,
    point,
    normal,
    tile=512,
    overlap=128,
):
    """
    Clean tiled validation. No inner functions. No getattr.
    Assumes batch size = 1.
    """
    assert input_.shape[0] == 1, "tiled validation expects batch=1"

    tile = int(tile)
    overlap = int(overlap)
    if overlap >= tile:
        overlap = max(0, tile // 4)

    # If you want these configurable later, add them to opt and set them explicitly in config.
    dino_stride = 8

    input_pad  = _pad_to_tile(input_, input_, tile)
    point_pad  = _pad_to_tile(point,  input_, tile)
    normal_pad = _pad_to_tile(normal, input_, tile)

    ctx = {
        "point_pad": point_pad,
        "normal_pad": normal_pad,
        "dino_full": DINO_Mat_features,
        "use_dino": (DINO_Mat_features is not None),
        "dino_stride": dino_stride,
    }

    # IMPORTANT: input_pad is already reflect-padded to at least (tile,tile) if needed.
    # Therefore pad_if_smaller should remain False here to avoid double padding.
    return infer_tiled_ctx(
        x=input_pad,
        model=model,
        forward_fn=forward_tiled_shadowformer_ctx,
        ctx=ctx,
        tile=tile,
        overlap=overlap,
        out_channels=3,
        use_amp=True,
        pad_if_smaller=False,
    )


def _pad_to_tile(x, input, tile_size):
    if x is None:
        return None
    H, W = x.shape[-2], x.shape[-1]     # <-- use x, not input
    pad_bottom = max(0, tile_size - H)
    pad_right  = max(0, tile_size - W)
    if pad_bottom > 0 or pad_right > 0:
        return F.pad(x, (0, pad_right, 0, pad_bottom), mode="replicate")
    return x

def _crop_dino_feat(fm, y0, y1, x0, x1, dino_stride):
    """
    fm: [1,C,h,w] at DINO grid
    crop to cover pixel region [y0:y1, x0:x1] in input_ coordinates.
    """
    if fm is None:
        return None
    assert fm.ndim == 4, fm.shape
    yy0 = y0 // dino_stride
    xx0 = x0 // dino_stride
    yy1 = (y1 + dino_stride - 1) // dino_stride  # ceil
    xx1 = (x1 + dino_stride - 1) // dino_stride  # ceil
    yy0 = max(0, yy0); xx0 = max(0, xx0)
    yy1 = min(fm.shape[-2], yy1); xx1 = min(fm.shape[-1], xx1)
    return fm[:, :, yy0:yy1, xx0:xx1]