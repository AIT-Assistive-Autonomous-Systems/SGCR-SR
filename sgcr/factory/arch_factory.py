import sys
import types
from pathlib import Path


def _ensure_vendor_paths(project_root: Path) -> None:
    """
    Make OmniSR importable exactly like your training script does.
    project_root is the directory that contains the 'OmniSR' folder.
    """
    project_root = Path(project_root)
    sys.path.insert(0, str(project_root / "OmniSR"))


def _install_omnisr_grid_sample_shim() -> None:
    """
    sr_model.py contains: `from OmniSR.utils.grid_sample import grid_sample`
    But OmniSR is not a Python package in this repo layout.

    This shim creates a fake module path:
        OmniSR.utils.grid_sample.grid_sample -> utils.image_utils.grid_sample

    No changes to OmniSR are required.
    """
    import utils.image_utils as iu
    # Create fake packages/modules
    if "OmniSR" not in sys.modules:
        sys.modules["OmniSR"] = types.ModuleType("OmniSR")
    if "OmniSR.utils" not in sys.modules:
        sys.modules["OmniSR.utils"] = types.ModuleType("OmniSR.utils")

    mod = types.ModuleType("OmniSR.utils.grid_sample")
    mod.grid_sample = iu.grid_sample
    sys.modules["OmniSR.utils.grid_sample"] = mod


def get_arch_sgcr(opt, project_root: Path):
    _ensure_vendor_paths(project_root)
    _install_omnisr_grid_sample_shim()

    arch = opt.arch
    if arch in ("unet_no_dino", "unet"):
        import sgcr.model_unet_OmniSR_no_dino as m
        model = m.ShadowFormer_OmniSR_NoDINO(
            opt=opt,
        )
    elif arch in ("twoStageShadowFormer", "NStageShadowFormer"):
        import sgcr.sgcr as m
        n_stage = 2 if arch == "twoStageShadowFormer" else int(getattr(opt, "n_stage", 2))
        model = m.NStageShadowFormer(
            opt=opt,
            n_stage=n_stage,
            tie_weights=opt.tie_stage_weights
        )
    else:
        import sgcr.sr_model as m
        if not hasattr(m, "F"):
            import torch.nn.functional as F
            m.F = F
        model = m.ShadowFormer_OmniSR(
            opt=opt,
        )

    return model

