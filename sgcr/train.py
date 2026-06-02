# Main training function that inherits the configuration
import os
import sys
from pathlib import Path
import torch
import time
import datetime
import logging
from tqdm import tqdm
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "OmniSR"))
import torch.nn as nn
from timm.utils import NativeScaler as TimmNativeScaler
from sgcr.losses.lpips_loss import LPIPSLoss
from sgcr.losses.ssim_loss import MSSSIMLoss
from sgcr.losses.loss_utility import compute_training_losses
import torch.nn.functional as F
from sgcr.ntire_metrics import NTIREMetrics
from sgcr.factory.criterion import get_criterion
from sgcr.factory.dataset import build_train_val_loaders
from sgcr.utility import handle_val_metrics_and_checkpoint, build_optimizer, resume_checkpoint_and_scheduler, to_hwc_rgb_uint8, maybe_imagenet_normalize_for_dino, get_dir_names, seed_everything_by_rank, load_checkpoint
from sgcr.scheduler import build_lr_scheduler
from sgcr.factory.arch_factory import get_arch_sgcr
from sgcr.tiled_inference import validate_one_image_tiled
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingWarmRestarts


def _maybe_set_dataset_epoch(loader, epoch: int) -> None:
    """Propagate epoch into datasets that implement set_epoch().

    This is useful for augmentation curricula (e.g., enable shadow transfer for
    the first N epochs, then disable) without rebuilding the DataLoader.
    """
    ds = getattr(loader, "dataset", None)
    # unwrap common dataset wrappers (Subset, etc.) a few times
    for _ in range(4):
        if ds is None:
            return
        if hasattr(ds, "set_epoch"):
            try:
                ds.set_epoch(epoch)
            except TypeError:
                ds.set_epoch(int(epoch))
            return
        ds = getattr(ds, "dataset", None)

def train(opt, tracker):
    if opt.debug: opt.eval_now = 2
    if not torch.cuda.is_available():
        raise RuntimeError("Please use a device with GPU acceleration.")
    device = torch.device("cuda:0")

    lpips_loss_fn = None
    if opt.lambda_lpips > 0.0:
        lpips_loss_fn = LPIPSLoss(opt).to(device)
    # MS-SSIM computed on luma only (more robust for relighting / small chroma shifts)
    # Loss returned is (1 - MS-SSIM), same convention as SSIMLoss in your pipeline.
    ssim_loss_fn = None
    if opt.ssim_loss_luma:
        ssim_loss_fn = MSSSIMLoss(data_range=1.0, on_luma=opt.ssim_loss_luma).to(device)

    ######### Logs dir ###########
    logname, _, model_dir = get_dir_names(opt)

    logging.basicConfig(filename=logname, level=logging.INFO)

    logging.info(opt.details(shorten=150))
    logging.info(f"Now time is : {datetime.datetime.now().isoformat()}")
    ########### Set Seeds ###########     
    _, g = seed_everything_by_rank(1234)
    torch.backends.cudnn.benchmark = True

    ######### Model ###########
    model_restoration = get_arch_sgcr(opt, ROOT)
    model_restoration.to(device)
    # for the finetuning run
    if (opt.pretrain_weights is not None) and (not opt.resume):
        ckpt_keys = load_checkpoint(model_restoration, opt.pretrain_weights, strict=False)
        # Warm-start all extra stages (index >= 2) from the learned stage[1] weights
        # rather than random init, so they begin as trained refiners.
        # Skip when the checkpoint already contains multi-stage weights (fine-tuning
        # from an already-converged N-stage run) — unless force_copy_stages is set.
        if opt.copy_stage1_to_stage2:
            if hasattr(model_restoration, "stages") and len(model_restoration.stages) >= 3:
                # Detect whether checkpoint already populated stages[2]+
                ckpt_has_multistage = any(
                    k.startswith("stages.2.") for k in (ckpt_keys or [])
                )
                force_copy = bool(getattr(opt, "force_copy_stages", False))
                if ckpt_has_multistage and not force_copy:
                    logging.info("[init] Checkpoint already contains multi-stage weights — "
                                 "skipping copy_stage1_to_stage2 (set force_copy_stages=true to override)")
                else:
                    src_state = model_restoration.stages[1].state_dict()
                    for idx in range(2, len(model_restoration.stages)):
                        model_restoration.stages[idx].load_state_dict(src_state)
                        logging.info(f"[init] Copied stage[1] weights -> stage[{idx}]")
    logging.info(f"Model type: {type(model_restoration)}")
    logging.info(f"Params (M): {sum(p.numel() for p in model_restoration.parameters()) / 1e6:.2f}")


    metrics = NTIREMetrics(device=device, enable_lpips=True, lpips_net="alex")
    NTIREMetrics.log_param_count(tracker, model_restoration)

    DINO_Net = None
    if opt.use_dino:
        DINO_Net = torch.hub.load(str(ROOT / "OmniSR" / "dinov2"), opt.dino_name, source="local")
        DINO_Net.to(device)
        DINO_Net.eval()
        for p in DINO_Net.parameters():
            p.requires_grad = False

    logging.info(str(model_restoration) + '\n')

    ######### Optimizer ###########
    start_epoch = 1
    optimizer = build_optimizer(opt, model_restoration)

    ######### Resume ###########
    if opt.resume:
        start_epoch, _, scheduler = resume_checkpoint_and_scheduler(
            opt=opt, model_restoration=model_restoration, optimizer=optimizer,
            scheduler_tmax=10, scheduler_eta_min=5e-5, strict=False)

    ######### Loss ###########
    criterion_restore = get_criterion(opt, device)  # either charbonnier or l2

    ######### DataLoader ###########
    logging.info('===> Loading datasets')
    train_loader, _, val_loader = build_train_val_loaders(opt=opt, g=g)

    # --------- Scheduler ----------
    # Build once, then call scheduler.step() once per epoch (at end of epoch), as you already do.
    steps_per_epoch = len(train_loader)  # available after train_loader is created
    scheduler = build_lr_scheduler(optimizer, opt, steps_per_epoch=steps_per_epoch)

    ######### train ###########
    logging.info("===> Start Epoch {} End Epoch {}".format(start_epoch,opt.nepoch))
    best_psnr = 0
    best_epoch = 0
    best_iter = 0
    logging.info("\nEvaluation after every {} Epochs !!!\n".format(opt.eval_now))
    if opt.train_only:
        logging.info("Training-only mode enabled: validation is disabled.")
    tracker.step(start_epoch, False, False)
    loss_scaler = TimmNativeScaler()

    img_multiple_of = 8 * opt.win_size

    # the train_ps must be the multiple of win_size
    UpSample = None
    if opt.use_dino:
        DINO_patch_size = 14
        UpSample = nn.UpsamplingBilinear2d(
            size=(int(opt.train_ps * DINO_patch_size / 8),
                int(opt.train_ps * DINO_patch_size / 8))
        )


    for epoch in range(start_epoch, opt.nepoch + 1):
        epoch_start_time = time.time()
        epoch_loss = 0

        # dataset-side epoch hook (augmentation scheduling)
        _maybe_set_dataset_epoch(train_loader, epoch)
        iterator = tqdm(train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True)

        for i, data in enumerate(iterator, 0):
            optimizer.zero_grad(set_to_none=True)

            target = data[0].to(device, non_blocking=True)
            input_  = data[1].to(device, non_blocking=True)
            point   = data[2].to(device, non_blocking=True)
            normal  = data[3].to(device, non_blocking=True)
            if not getattr(opt, 'use_depth', True):
                point  = torch.zeros_like(point)
            if not getattr(opt, 'use_normal', True):
                normal = torch.zeros_like(normal)

            with torch.autocast("cuda", dtype=torch.float16):
                dino_mat_features = None
                if opt.use_dino:
                    with torch.no_grad():
                        input_DINO = UpSample(input_)
                        dino_norm = bool(getattr(opt, "dino_imagenet_norm", False))
                        input_DINO = maybe_imagenet_normalize_for_dino(input_DINO, dino_norm)
                        dino_mat_features = DINO_Net.get_intermediate_layers(input_DINO, 4, True)

                # reconstruction
                restored = model_restoration(
                    input_, dino_mat_features, point, normal,
                )
                losses = compute_training_losses(
                    opt=opt,
                    model_restoration=model_restoration,
                    restored=restored,
                    target=target,
                    input_=input_,
                    criterion_restore=criterion_restore,
                    lpips_loss_fn=lpips_loss_fn,
                    ssim_loss_fn=ssim_loss_fn,
                )

                loss = losses.total

            # AMP step
            loss_scaler(
                loss,
                optimizer,
                parameters=model_restoration.parameters(),
                clip_grad=opt.tresh_gradient_clipping)

            epoch_loss += loss.item()
            iterator.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        # log lr
        tracker.log_scalar("train/lr", optimizer.param_groups[0]["lr"])
        avg_epoch_loss = epoch_loss / len(train_loader)
        tracker.log_scalar("train/loss", avg_epoch_loss)
        ################# Evaluation ########################
        if val_loader:
            if epoch % opt.eval_now == 0:
                model_restoration.eval()
                if DINO_Net:
                    DINO_Net.eval()
                metrics.reset()
                _ts = opt.tile_size
                tile = (int(_ts[0]), int(_ts[1])) if isinstance(_ts, (list, tuple)) else int(_ts)
    
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    for _, data_val in enumerate(val_loader, 0):
                        target = data_val[0].to(device, non_blocking=True)
                        input_  = data_val[1].to(device, non_blocking=True)
                        point   = data_val[2].to(device, non_blocking=True)
                        normal  = data_val[3].to(device, non_blocking=True)
                        filenames = data_val[4]
    
                        # pad
                        height, width = input_.shape[2], input_.shape[3]
    
                        # Correct ceil-to-multiple
                        H = ((height + img_multiple_of - 1) // img_multiple_of) * img_multiple_of
                        W = ((width  + img_multiple_of - 1) // img_multiple_of) * img_multiple_of
                        padh = H - height
                        padw = W - width
    
                        if padh or padw:
                            input_  = F.pad(input_,  (0, padw, 0, padh), "replicate")
                            point   = F.pad(point,   (0, padw, 0, padh), "replicate")
                            normal  = F.pad(normal,  (0, padw, 0, padh), "replicate")
    
                        dino_mat_features = None
                        if opt.use_dino:
                            input_DINO = F.interpolate(
                                input_,
                                scale_factor=(DINO_patch_size / 8),
                                mode="bilinear",
                                align_corners=False,
                            )
                            dino_norm = opt.dino_imagenet_norm
                            input_DINO = maybe_imagenet_normalize_for_dino(input_DINO, dino_norm)
                            dino_mat_features = DINO_Net.get_intermediate_layers(input_DINO, 4, True)
    
                        _tile_max = max(tile) if isinstance(tile, (list, tuple)) else tile
                        if max(H, W) > _tile_max:
                            logging.info(f"[VAL] {filenames[0]} H,W={(H,W)} tile={tile} -> tiled inference")
                            restored = validate_one_image_tiled(
                                opt, model_restoration, input_,
                                DINO_Mat_features=dino_mat_features,
                                point=point, normal=normal,
                                tile=tile, overlap=opt.tile_overlap,
                            )
                        else:
                            restored = model_restoration(
                                input_, dino_mat_features, point, normal,
                            )
                        restored = restored.clamp_(0.0, 1.0)[:, :, :height, :width]
                        metrics.update(restored, target)
    
    
                        for name, image in zip(data_val[4], restored):
                            tracker.log_image(name=name, image=to_hwc_rgb_uint8(image), maximum_per_step=10)
    
                best_psnr, best_epoch, best_iter = handle_val_metrics_and_checkpoint(
                    metrics=metrics,
                    tracker=tracker,
                    model_restoration=model_restoration,
                    optimizer=optimizer,
                    model_dir=model_dir,
                    epoch=epoch,
                    i=i,
                    best_psnr=best_psnr,
                    best_epoch=best_epoch,
                    best_iter=best_iter,
                    epoch_loss=epoch_loss,
                    train_loader=train_loader,
                    opt=opt,
                    rank0=True,
                )
                model_restoration.train()
        
        # Step scheduler
        if isinstance(scheduler, SequentialLR):
            # SequentialLR cannot take a "t" argument
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            # Warm restarts benefits from fractional epoch stepping
            scheduler.step(epoch - 1 + (i + 1) / max(1, len(train_loader)))
        else:
            scheduler.step()
        tracker.step()
        
        logging.info(
            "Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(
                epoch, time.time() - epoch_start_time, epoch_loss, optimizer.param_groups[0]["lr"]
            )
        )

        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,"model_latest.pth"))

        if epoch%opt.checkpoint == 0:
            torch.save({'epoch': epoch,
                        'state_dict': model_restoration.state_dict(),
                        'optimizer' : optimizer.state_dict()
                        }, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch)))
    logging.info("Now time is : {}".format(datetime.datetime.now().isoformat()))


