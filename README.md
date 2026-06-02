# Private General Deshadowing

This repository contains the training and evaluation code for the SGCR / multi-stage ShadowFormer deshadowing experiments. Experiments are configured with [YAECS](https://github.com/valentingol/yaecs): the default configuration lives in `config/default/_root_default.yaml`, and the experiment configurations used for training and ablations live under `config/training/`.

> **Checkpoints.** The released repository does not currently include model checkpoints. The checkpoints required for exact reproduction will be added during CVPRW and will be freely available. Until then, set `pretrain_weights` to a local checkpoint if you have one, or train the models from the provided configs.

## Repository layout

```text
.
├── main.py                         # entry point: train / val / test
├── config/
│   ├── config.py                   # CVConfig, the YAECS configuration class
│   ├── default/_root_default.yaml  # typed default values
│   └── training/                   # experiment configs and ablations
├── sgcr/                           # model wrappers, training, validation, losses, utilities
├── tools/time_single_forward.py    # inference speed utility
├── OmniSR/                         # git submodule dependency
├── requirements.txt                # Python package list
├── environment.yml                 # conda environment export
└── spec-file.txt                   # explicit conda Linux spec export
```

## Requirements

The code requires a CUDA-capable GPU. Both training and validation explicitly check for CUDA and will stop if no GPU is available.

Tested environment assumptions from the exported files:

- Linux
- Python 3.11
- PyTorch / torchvision with CUDA support
- YAECS
- OmniSR submodule, including the local DINOv2 code used through `torch.hub.load(..., source="local")`

## Installation

Clone the repository with submodules, or initialise the `OmniSR` submodule after extracting the archive:

```bash
git clone --recurse-submodules <REPO_URL>
cd Private_general_deshadowing-public

# If the repository was cloned without submodules, run:
git submodule update --init --recursive
```

The uploaded archive contains an empty `OmniSR/` directory, so the submodule step is required before running the code.

Create and activate a Python environment. The most portable option is to create a clean Python 3.11 environment and then install the packages:

```bash
conda create -n sgcr python=3.11 -y
conda activate sgcr
python -m pip install --upgrade pip

# GDAL is often easier to install with conda than pip.
conda install -c conda-forge gdal -y

# Install the remaining Python packages.
pip install -r requirements.txt
```

Alternatively, recreate the exported conda environment and then install the pip requirements:

```bash
conda env create -f environment.yml
conda activate myenv
pip install -r requirements.txt
```

If your CUDA driver or cluster image is incompatible with the pinned `torch`, `torchvision`, `xformers`, or CUDA wheel versions in `requirements.txt`, install the PyTorch stack that matches your CUDA setup first, then install the remaining requirements.

A minimal import check is:

```bash
python - <<'PY'
import torch
import yaecs
import timm
import lpips
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("yaecs import ok")
PY
```

## Dataset format

The training and validation loaders are provided by the `OmniSR` submodule. The expected split structure is:

```text
<DATA_ROOT>/
├── train/
│   ├── origin/       # shadow input images
│   ├── shadow_free/  # target images for supervised training
│   ├── depth/        # depth maps, typically .npy
│   └── normal/       # normal maps, typically .npy
├── val/
│   ├── origin/
│   ├── shadow_free/
│   ├── depth/
│   └── normal/
└── test/
    ├── origin/       # test images; no shadow_free required
    ├── depth/
    └── normal/
```

For test-time inference, `DataLoaderTest` reads `origin/`, `depth/`, and `normal/` from the directory specified by `val_dir`. The test split does not require ground-truth images.

Depth and normal preprocessing helpers are included:

```bash
# Edit sgcr/paths.py first so DATASET_DIR points to your dataset root.
python sgcr/prepare_depth.py
python sgcr/prepare_normal.py --split-name train
python sgcr/prepare_normal.py --split-name val
python sgcr/prepare_normal.py --split-name test
```

`prepare_depth.py` currently loops over `train` and `test`; edit the split list in that file if you also want to process `val` automatically.

## Configuring experiments with YAECS

The entry point is:

```bash
python main.py --config <CONFIG_PATH>
```

`main.py` builds a `CVConfig` with YAECS and dispatches according to `run_mode`:

- `run_mode: train` calls `sgcr.train.train`
- `run_mode: val`, `valid`, `validate`, or `validation` calls validation with metrics
- `run_mode: test` calls test inference without requiring ground truth

Most configs in `config/training/` use placeholder paths (e.g. `/path/to/dataset/...`). Before running, copy `config/local_paths.example.yaml` to `config/local_paths.yaml`, fill in your local paths, and merge it as an override config.

Example local override file (see `config/local_paths.example.yaml`):

```yaml
# config/local_paths.yaml
save_dir: /path/to/output/experiments/n3_s4
train_dir: /path/to/dataset/train
val_dir: /path/to/dataset/val
pretrain_weights: /path/to/checkpoints/model_epoch_1990.pth
train_workers: 8
eval_workers: 4
```

Run the base experiment with an override config:

```bash
python main.py --config "[config/training/base/config.yaml,config/local_paths.yaml]"
```

You can also edit the relevant YAML directly. The most important fields are:

| Field | Purpose |
| --- | --- |
| `save_dir` | Root directory for logs, model checkpoints, validation images, and metrics |
| `train_dir` | Training split directory |
| `val_dir` | Validation or test split directory |
| `pretrain_weights` | Checkpoint used for warm-starting, validation, or test inference |
| `run_mode` | `train`, `val`, or `test` |
| `arch` | Model family, for example `ShadowFormer`, `twoStageShadowFormer`, or `NStageShadowFormer` |
| `n_stage` | Number of stages for `NStageShadowFormer` |
| `use_dino` | Enables local DINOv2 guidance |
| `use_depth`, `use_normal` | Enables or zeroes geometric cues |
| `tile_size`, `tile_overlap` | Tiled inference settings for large images |
| `batch_size`, `validation_batch_size` | Train and validation batch sizes |
| `nepoch`, `lr_initial`, `lr_scheduler` | Main optimisation schedule |

## Reproducing the provided experiments

The main experiment config is:

```bash
python main.py --config "[config/training/base/config.yaml,config/local_paths.yaml]"
```

This corresponds to the 3-stage `NStageShadowFormer` setting with S4 contraction loss and stage warm-starting.

To reproduce the ablations, replace the first config path with one of the following:

```text
config/training/Nstage_ablation/1/config.yaml
config/training/Nstage_ablation/2/config.yaml
config/training/Nstage_ablation/4/config.yaml
config/training/Nstage_ablation/5/config.yaml
config/training/geometry_ablation/no_contraction/config.yaml
config/training/geometry_ablation/no_depth/config.yaml
config/training/geometry_ablation/no_depth_no_normals/config.yaml
config/training/geometry_ablation/no_dino/config.yaml
config/training/ensemble/n2_dynamic_s4/config.yaml
config/training/ensemble/n3_dynamic_s4/config.yaml
config/training/ensemble/n3_s4/config.yaml
config/training/ensemble/s1_s2/config.yaml
```

For example:

```bash
python main.py --config "[config/training/geometry_ablation/no_dino/config.yaml,config/local_paths.yaml]"
```

The `config/training/progression_history/` directory contains earlier training schedules used during development and warm-start progression. These are useful for tracing the training history, but the current main reproduction config is `config/training/base/config.yaml`.

## Validation and test inference

Validation requires a checkpoint in `pretrain_weights` and a validation split with ground truth:

```yaml
# config/local_val.yaml
run_mode: val
save_dir: /path/to/output/eval_n3_s4
val_dir: /path/to/dataset/val
pretrain_weights: /path/to/checkpoints/model_best.pth
```

```bash
python main.py --config "[config/training/base/config.yaml,config/local_val.yaml]"
```

Validation writes restored images and diagnostics under:

```text
<save_dir>/log/<arch><env>_val/results/
```

Test inference uses `run_mode: test` and reads images from `val_dir`, but does not require `shadow_free/`:

```yaml
# config/local_test.yaml
run_mode: test
save_dir: /path/to/output/test_n3_s4
val_dir: /path/to/dataset/test
pretrain_weights: /path/to/checkpoints/model_best.pth
```

```bash
python main.py --config "[config/training/base/config.yaml,config/local_test.yaml]"
```

Test predictions are written under:

```text
<save_dir>/log/<arch><env>_test/results/
```

## Checkpoints and logs

During training, outputs are created inside `save_dir`:

```text
<save_dir>/log/<arch><env><timestamp>/
├── <timestamp>.txt       # training log
├── models/
│   ├── model_latest.pth
│   ├── model_epoch_<N>.pth
│   └── model_best.pth
├── results/
└── tensorlog/
```

The `checkpoint` config value controls how often `model_epoch_<N>.pth` files are saved. `model_latest.pth` is updated every epoch. `model_best.pth` is saved according to the best validation PSNR.

## Ensemble evaluation

The validation code supports two ensemble modes:

1. Same-architecture ensembles through `ensemble_weights`.
2. Cross-architecture ensembles through `ensemble_configs`.

Example same-architecture ensemble override:

```yaml
run_mode: val
pretrain_weights: /path/to/checkpoints/member_1.pth
ensemble_weights:
  - /path/to/checkpoints/member_1.pth
  - /path/to/checkpoints/member_2.pth
  - /path/to/checkpoints/member_3.pth
tta_hflip: true
```

Example cross-architecture ensemble override:

```yaml
run_mode: val
ensemble_configs:
  - [config/training/base/config.yaml, /path/to/checkpoints/n3_s4.pth]
  - [config/training/Nstage_ablation/2/config.yaml, /path/to/checkpoints/n2_s4.pth]
```

## Timing a model

To estimate forward-pass time for a config and checkpoint:

```bash
python tools/time_single_forward.py config/training/base/config.yaml
```

The timing script loads one real image from `val_dir`, applies the configured tiling path if needed, and reports per-image runtime statistics.

## Common issues

### `ModuleNotFoundError` for `OmniSR`, `utils`, or `model`

Initialise the submodule and run commands from the repository root:

```bash
git submodule update --init --recursive
python main.py --config config/training/base/config.yaml
```

### DINOv2 cannot be loaded

The code loads DINOv2 locally from `OmniSR/dinov2` using `torch.hub.load(..., source="local")`. Make sure the `OmniSR` submodule contains the expected `dinov2` directory and model definition.

### Out-of-memory during training or evaluation

Reduce `batch_size` for training, or reduce `tile_size` / increase tiled inference usage during validation and testing. The provided configs generally use `tile_size: 896` and `tile_overlap: 128`.

### Paths use placeholders

The checked-in configs use placeholder paths like `/path/to/dataset/...` and `/path/to/output/...`. Copy `config/local_paths.example.yaml` to `config/local_paths.yaml`, set your local `save_dir`, `train_dir`, `val_dir`, and `pretrain_weights`, and pass it as an override config.
