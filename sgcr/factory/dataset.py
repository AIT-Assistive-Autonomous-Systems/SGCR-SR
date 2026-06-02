from __future__ import annotations
from typing import Optional, Tuple
import logging
import torch
from torch.utils.data import DataLoader, ConcatDataset
from sgcr.utility import worker_init_fn
from OmniSR.utils.loader import get_training_data, get_validation_data
 

def build_train_val_loaders(
    *,
    opt,
    g: torch.Generator,
    world_size=1,  # kept for backward compat; always treated as 1 in this version
    worker_init_fn=worker_init_fn,
) -> Tuple[DataLoader, None, Optional[DataLoader]]:
    """
    Build train/val datasets, samplers, and dataloaders (no fallbacks).

    Requires opt fields:
      - train_ps, train_dir, debug
      - batch_size, train_workers
      - val_dir, validation_batch_size, eval_workers
      - extra_train_dirs (optional list[str]): additional directories with the
        same structure as train_dir to append to the training set via ConcatDataset.
    """
    train_only = opt.train_only
    
    # ---- Train ----
    img_options_train = {
        "patch_size": opt.train_ps,
        "aug_all_rotations": getattr(opt, "aug_all_rotations", False),
        "aug_hflip": getattr(opt, "aug_hflip", False),
        "aug_multiscale": getattr(opt, "aug_multiscale", False),
    }
    def _build_one_train_dataset(rgb_dir):
        """Instantiate a single training dataset for the given directory."""
        return get_training_data(rgb_dir, img_options_train, opt.debug)

    train_dataset = get_training_data(opt.train_dir, img_options_train, opt.debug)

    # Append any extra training directories (e.g. val split for final fine-tuning)
    extra_train_dirs = opt.extra_train_dirs  # list[str] or None
    if extra_train_dirs:
        extra_datasets = [train_dataset]
        for extra_dir in extra_train_dirs:
            extra_ds = _build_one_train_dataset(extra_dir)            
            extra_datasets.append(extra_ds)
            logging.info(f"[extra_train_dirs] Added {len(extra_ds)} samples from {extra_dir}")
        train_dataset = ConcatDataset(extra_datasets)
        logging.info(f"[extra_train_dirs] Combined train set size: {len(train_dataset)}")
    train_sampler = None

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,
        persistent_workers=opt.train_workers > 0,
        num_workers=opt.train_workers,
        prefetch_factor=2 if opt.train_workers > 0 else None,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        generator=g,
    )

    # ---- Val ----
    val_loader=None
    len_valset=0
    if not train_only:
        val_dataset = get_validation_data(opt.val_dir, debug=opt.debug)
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=opt.validation_batch_size,
            num_workers=opt.eval_workers,
            persistent_workers=opt.eval_workers > 0,
            prefetch_factor=2 if opt.eval_workers > 0 else None,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=worker_init_fn,
            generator=g,
        )
        len_valset = len(val_dataset)
        
    len_trainset = len(train_dataset)
    
    logging.info(f"Sizeof training set: {len_trainset} sizeof validation set: {len_valset}")

    return train_loader, train_sampler, val_loader
