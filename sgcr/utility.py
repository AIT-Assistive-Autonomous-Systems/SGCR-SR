from __future__ import annotations
import os
from typing import Callable, Optional, Sequence, Tuple, Union
from collections import OrderedDict
import numpy as np
import logging
import torch
import datetime
import OmniSR.utils as utils
import random
import torch.optim as optim

def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def make_epoch_results_dir(results_root: str, epoch: int) -> str:
    """
    Returns a directory like: <results_root>/epoch_0005
    """
    epoch_dir = os.path.join(results_root, f"epoch_{epoch:04d}")
    ensure_dir(epoch_dir)
    return epoch_dir


def to_hwc_rgb_uint8(img_chw: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Converts a CHW float image in [0,1] to HWC uint8 in [0,255].
    Accepts torch.Tensor or np.ndarray. Output is np.uint8 HWC.
    """
    if isinstance(img_chw, torch.Tensor):
        img = img_chw.detach().float().cpu().numpy()
    else:
        img = img_chw.astype(np.float32, copy=False)

    # Expect CHW
    if img.ndim != 3 or img.shape[0] not in (1, 3, 4):
        raise ValueError(f"Expected CHW with C in (1,3,4), got shape {img.shape}")

    # If RGBA, drop A
    if img.shape[0] == 4:
        img = img[:3, :, :]

    # If grayscale, expand to 3 channels (optional; remove if you don't want this)
    if img.shape[0] == 1:
        img = np.repeat(img, 3, axis=0)

    img = np.clip(img, 0.0, 1.0)
    img_hwc = np.transpose(img, (1, 2, 0))
    img_u8 = (img_hwc * 255.0).round().astype(np.uint8)
    return img_u8


def load_checkpoint(model, weights, strict=True):
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    logger = logging.getLogger(__name__)

    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    # 0) load into underlying module if wrapped (DDP)
    target = model.module if hasattr(model, "module") else model

    # 1) strip DDP/DataParallel "module." prefix from checkpoint keys
    base_state = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        base_state[name] = v

    # 2) detect stage prefix scheme on the TARGET module
    stage_prefixes = None
    if hasattr(target, "stage1") and hasattr(target, "stage2"):
        stage_prefixes = ["stage1.", "stage2."]
    elif hasattr(target, "stages"):
        try:
            n = len(target.stages)
            if n >= 1:
                stage_prefixes = [f"stages.{i}." for i in range(n)]
        except TypeError:
            stage_prefixes = None
    elif hasattr(target, "stage"):
        stage_prefixes = ["stage."]

    def _remap_stage_keys_for_target(base_sd, target_stage_prefixes):
        """
        Remap between common multi-stage checkpoint naming schemes.

        Supported conversions:
          - stage1.* / stage2.*     -> stages.0.* / stages.1.*
          - stages.0.* / stages.1.* -> stage1.* / stage2.*
        """
        if not target_stage_prefixes:
            return base_sd, False

        keys = list(base_sd.keys())

        # Case A:
        #   checkpoint: stage1.*, stage2.*
        #   target:     stages.0.*, stages.1.*
        if all(p.startswith("stages.") for p in target_stage_prefixes):
            legacy_prefixes = [f"stage{i+1}." for i in range(len(target_stage_prefixes))]
            if any(k.startswith(p) for p in legacy_prefixes for k in keys):
                remapped = OrderedDict()
                for k, v in base_sd.items():
                    matched = False
                    for i, old_p in enumerate(legacy_prefixes):
                        if k.startswith(old_p):
                            remapped[f"stages.{i}." + k[len(old_p):]] = v
                            matched = True
                            break
                    if not matched:
                        remapped[k] = v
                return remapped, True

        # Case B:
        #   checkpoint: stages.0.*, stages.1.*
        #   target:     stage1.*, stage2.*
        legacy_target_prefixes = [f"stage{i+1}." for i in range(len(target_stage_prefixes))]
        if target_stage_prefixes == legacy_target_prefixes:
            ckpt_prefixes = [f"stages.{i}." for i in range(len(target_stage_prefixes))]
            if any(k.startswith(p) for p in ckpt_prefixes for k in keys):
                remapped = OrderedDict()
                for k, v in base_sd.items():
                    matched = False
                    for i, old_p in enumerate(ckpt_prefixes):
                        if k.startswith(old_p):
                            remapped[f"stage{i+1}." + k[len(old_p):]] = v
                            matched = True
                            break
                    if not matched:
                        remapped[k] = v
                return remapped, True

        return base_sd, False

    # 3) first remap known multi-stage naming mismatches
    new_state_dict = base_state
    new_state_dict, remapped = _remap_stage_keys_for_target(base_state, stage_prefixes)
    if remapped:
        logger.info("Remapped checkpoint stage prefixes to match target module")

    # 4) expand single-stage ckpt into stage-prefixed keys if needed
    if stage_prefixes is not None:
        has_any_prefixed_key = any(
            any(k.startswith(p) for p in stage_prefixes) for k in base_state.keys()
        )
        if (not remapped) and (not has_any_prefixed_key):
            expanded = OrderedDict()
            for k, v in new_state_dict.items():
                for p in stage_prefixes:
                    expanded[p + k] = v
            new_state_dict = expanded
            logger.info(
                "Expanded single-stage checkpoint into %d stages: %s",
                len(stage_prefixes),
                ", ".join([p.rstrip(".") for p in stage_prefixes]),
            )

    # 5) drop size-mismatched keys so strict=False can proceed without crashing.
    #    This most commonly hits relative_position_bias_table when train_ps changes
    #    and the bottleneck conv block's effective win_size was clamped differently.
    model_state = target.state_dict()
    size_mismatch_keys = [
        k for k, v in new_state_dict.items()
        if k in model_state and v.shape != model_state[k].shape
    ]
    if size_mismatch_keys:
        logger.warning(
            "Dropping %d size-mismatched key(s) from checkpoint (will use model init): %s",
            len(size_mismatch_keys),
            size_mismatch_keys[:20],
        )
        new_state_dict = OrderedDict(
            (k, v) for k, v in new_state_dict.items() if k not in size_mismatch_keys
        )

    # 6) load
    try:
        incompatible = target.load_state_dict(new_state_dict, strict=strict)
    except RuntimeError as e:
        if strict:
            logger.warning("Strict load failed; retrying strict=False. Error: %s", e)
            incompatible = target.load_state_dict(new_state_dict, strict=False)
        else:
            raise

    # 7) log mismatches
    if (incompatible.missing_keys or incompatible.unexpected_keys):
        logger.warning(
            "Checkpoint load: %d missing keys, %d unexpected keys",
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )
        if incompatible.missing_keys:
            logger.warning("Missing keys (first 50): %s", incompatible.missing_keys[:50])
        if incompatible.unexpected_keys:
            logger.warning("Unexpected keys (first 50): %s", incompatible.unexpected_keys[:50])

    # Return the checkpoint key names (post-remap) so callers can inspect what was loaded.
    return list(new_state_dict.keys())
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


def get_dir_names(opt):
    # creates the log folders and an unique path for each experiment
    dir_name = opt.save_dir
    log_dir = os.path.join(dir_name, 'log', opt.arch+opt.env+datetime.datetime.now().isoformat(timespec='minutes'))
    logname = os.path.join(log_dir, datetime.datetime.now().isoformat(timespec='minutes')+'.txt') 

    result_dir = os.path.join(log_dir, 'results')
    model_dir  = os.path.join(log_dir, 'models')
    tensorlog_dir  = os.path.join(log_dir, 'tensorlog')
    utils.mkdir(log_dir)
    utils.mkdir(result_dir)
    utils.mkdir(model_dir)
    utils.mkdir(tensorlog_dir)
    utils.mknod(logname)

    return logname, result_dir, model_dir


def worker_init_fn(worker_id):
    random.seed(1234 + worker_id)

def seed_everything_by_rank(base_seed: int = 1234, **_kwargs):
    seed = int(base_seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # IMPORTANT: DataLoader expects a CPU generator
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    return seed, g


def resume_checkpoint_and_scheduler(
    *,
    opt,
    model_restoration,
    optimizer: optim.Optimizer,
    scheduler_tmax: int = 10,
    scheduler_eta_min: float = 5e-5,
    strict: bool = False,
) -> Tuple[int, float, optim.lr_scheduler._LRScheduler]:
    """
    Args:
      - ckpt path from opt.pretrain_weights
      - start_epoch = load_start_epoch + 1
      - load model weights (all ranks)
      - try load optimizer state (optional; warns on ValueError)
      - optional arch-specific stage2 zero init
      - rebuild CosineAnnealingLR scheduler
      - log/print LR

    Returns:
      (start_epoch, lr, scheduler)
    """
    ckpt_path = opt.pretrain_weights
    start_epoch = int(utils.load_start_epoch(ckpt_path)) + 1

    load_checkpoint(model_restoration, ckpt_path, strict=strict)

    # Optimizer state is optional; skip if incompatible
    lr: float
    try:
        lr = float(utils.load_optim(optimizer, ckpt_path))
    except ValueError as e:
        print(f"[WARN] Skipping optimizer state load: {e}")
        lr = float(optimizer.param_groups[0]["lr"])

    # Arch-specific post-load init
    if opt.multi_stage_mode == "residual":
        # two-stage legacy
        if hasattr(model_restoration, "_zero_init_stage2_head"):
            if opt.arch == "twoStageShadowFormer":
                model_restoration._zero_init_stage2_head()
        # n-stage wrapper (if you expose a helper; optional)
        if hasattr(model_restoration, "_zero_init_heads"):
            try:
                model_restoration._zero_init_heads(stage_indices=range(1, model_restoration.n_stage))
            except Exception:
                pass

    # Recreate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=scheduler_tmax,
        eta_min=scheduler_eta_min,
    )

    logging.info("------------------------------------------------------------------------------")
    logging.info(f"==> Resuming Training with learning rate:{lr}")
    logging.info("------------------------------------------------------------------------------")

    return start_epoch, lr, scheduler





def build_optimizer(opt, model, *, params=None) -> optim.Optimizer:
    """
    Optimizer factory.
    Required opt fields:
      - optimizer: str ("adam" or "adamw")
      - lr_initial: float
      - weight_decay: float
    """
    # No fallbacks: these must exist or Python will raise AttributeError
    name = opt.optimizer.lower().strip()
    lr = opt.lr_initial
    wd = opt.weight_decay

    if params is None:
        params = model.parameters()

    kwargs = dict(lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=wd)

    if name == "adam":
        return optim.Adam(params, **kwargs)
    elif name == "adamw":
        return optim.AdamW(params, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer '{opt.optimizer}'. Use 'adam' or 'adamw'.")
    

def handle_val_metrics_and_checkpoint(
    *,
    metrics,
    tracker,
    model_restoration,
    optimizer,
    model_dir: str,
    epoch: int,
    i: int,
    best_psnr: float,
    best_epoch: int,
    best_iter: int,
    epoch_loss: float,
    train_loader,
    opt,
    rank0: bool,
    save_best_name: str = "model_best.pth",
    metrics_prefix: str = "val",
) -> Tuple[float, int, int, float, float, Optional[float]]:
    """
    Consolidates end-of-validation bookkeeping:
      - log metrics to tracker (rank0 only)
      - parse psnr/ssim/lpips from logged output
      - update and save best checkpoint by PSNR (rank0 only)
      - log epoch train loss to tracker (rank0 only)
      - print summary lines
      - restore train mode and reset img_size if present

    Returns:
      (best_psnr, best_epoch, best_iter, metrics_out["psnr"], metrics_out["ssim"], metrics_out.get("lpips_alex", None))
    """

    metrics_out = None
    if rank0 and tracker is not None:
        metrics_out = metrics.log(tracker, prefix=metrics_prefix)

    # save the best checkpoint
    if metrics_out["psnr"] > best_psnr:
        best_psnr = metrics_out["psnr"]
        best_epoch = epoch
        best_iter = i
        if rank0:
            ckpt = {
                "epoch": epoch,
                "state_dict": model_restoration.module.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(ckpt, os.path.join(model_dir, save_best_name))

    avg_epoch_loss = epoch_loss / max(1, len(train_loader))
    if rank0:
        tracker.log_scalar("train/loss_epoch", avg_epoch_loss)

    logging.info(
        "[Ep %d it %d\t PSNR: %.4f  SSIM: %.4f  LPIPS(Alex): %.4f\t] ---- "
        "[best_Ep %d best_it %d Best_PSNR %.4f]"
        % (epoch, i, metrics_out["psnr"], metrics_out["ssim"], metrics_out.get("lpips_alex", None), best_epoch, best_iter, best_psnr)
    )
    logging.info("Now time is : {}".format(datetime.datetime.now().isoformat()))

    model_restoration.train()
    if hasattr(model_restoration.module, "img_size"):
        model_restoration.module.img_size = (opt.train_ps, opt.train_ps)

    return best_psnr, best_epoch, best_iter