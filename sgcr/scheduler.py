from torch.optim import lr_scheduler


def build_lr_scheduler(optimizer, opt, steps_per_epoch: int):
    """
    Supported:
      - "constant"
      - "step"
      - "cosine"
      - "linear"

    Optional warmup:
      - opt.warmup: bool
      - opt.warmup_epochs: int
      - opt.warmup_start_factor: float
    """
    # Accept either lr_scheduler or lr_schedule
    schedule = opt.lr_scheduler
    total_epochs = opt.nepoch
    lr_min = opt.lr_min
    step_size = opt.lr_step_size
    gamma = opt.lr_gamma

    # Warmup knobs
    warmup_epochs = opt.warmup_epochs
    warmup_start_factor = opt.warmup_start_factor

    # ---- base scheduler (epoch-based) ----
    if schedule == "constant":
        base_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

    elif schedule == "step":
        base_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif schedule == "cosine":
        # Use lr_T_max if set (e.g. when resuming mid-run); otherwise decay over full nepoch.
        t_max = int(opt.lr_T_max) if opt.lr_T_max is not None else total_epochs
        base_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, t_max), eta_min=lr_min
        )
    elif schedule in ("cosine_warm_restarts", "cosine_restart", "cosine_restarts"):
        # CosineAnnealingWarmRestarts: cycles of length T_0 epochs (optionally multiplied by T_mult)
        T_0 = opt.lr_T_max  # interpret config lr_T_max as first cycle length (epochs)
        T_mult = opt.lr_T_mult
        base_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, T_0), T_mult=max(1, T_mult), eta_min=lr_min
        )
    elif schedule == "linear":
        # Linear decay multiplier from 1.0 -> 0.0 over total_epochs
        # (If you need linear-to-lr_min, set lr_min by hand in your optimizer or
        # implement a custom schedule; this matches typical "linear to zero".)
        def linear_lambda(epoch: int):
            if total_epochs <= 1:
                return 1.0
            t = min(max(epoch, 0), total_epochs) / float(total_epochs)
            return 1.0 - t

        base_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_lambda)
    else:
        raise ValueError(
            f"Unknown lr scheduler '{schedule}'. Use one of: constant, step, cosine, linear."
        )

    # ---- optional warmup ----
    if warmup_epochs > 0:
        warmup = lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            total_iters=warmup_epochs,
        )
        scheduler = lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, base_scheduler],
            milestones=[warmup_epochs],
        )
        return scheduler

    return base_scheduler