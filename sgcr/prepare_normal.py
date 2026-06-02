import argparse
from pathlib import Path

import numpy as np

from paths import DATASET_DIR

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def depth_to_normals(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)

    dzdx = np.gradient(depth, axis=1)
    dzdy = np.gradient(depth, axis=0)

    nx = -dzdx
    ny = -dzdy
    nz = np.ones_like(depth, dtype=np.float32)

    n = np.stack([nx, ny, nz], axis=-1)
    n = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8)
    return n.astype(np.float32)


def run_split(split_dir: Path):
    depth_dir = split_dir / "depth"
    normal_dir = split_dir / "normal"
    normal_dir.mkdir(parents=True, exist_ok=True)

    if not depth_dir.exists():
        raise FileNotFoundError(f"Depth folder not found: {depth_dir}")

    depth_files = sorted(depth_dir.glob("*.npy"))
    if not depth_files:
        raise FileNotFoundError(f"No .npy depth files found in: {depth_dir}")

    iterator = depth_files
    if tqdm is not None:
        iterator = tqdm(depth_files, desc=f"Normals {split_dir.name}", unit="file")

    for dp in iterator:
        depth = np.load(dp)

        # handle common shapes: (H,W), (1,H,W), (H,W,1)
        if depth.ndim == 3:
            if depth.shape[0] == 1:
                depth = depth[0]
            elif depth.shape[-1] == 1:
                depth = depth[..., 0]
            else:
                depth = depth[..., 0]

        normals = depth_to_normals(depth)
        normals = normals.transpose(2, 0, 1)  # HWC to CHW
        np.save(normal_dir / dp.name, normals.astype(np.float32))


    print(f"[ok] wrote {len(depth_files)} normal maps to {normal_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-name", default=None, help="folder name (e.g. validation, train)")
    args = ap.parse_args()

    if args.split_name:
        splits = [args.split_name]
    else:
        splits = ["train","test"]

    for s in splits:
        split_dir = DATASET_DIR / s
        if split_dir.exists():
            run_split(split_dir)
        else:
            print(f"[skip] {split_dir} not found")


if __name__ == "__main__":
    main()
