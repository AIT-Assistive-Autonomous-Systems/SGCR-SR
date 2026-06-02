import os
import sys
import subprocess
from pathlib import Path

from paths import OMNISR_DIR, DATASET_DIR


def run_split(split_dir: Path, encoder="vits", input_size=512):
    img_dir = split_dir / "origin"
    out_dir = split_dir / "depth"
    out_dir.mkdir(parents=True, exist_ok=True)

    script = OMNISR_DIR / "Depth-Anything-V2_run.py"

    if not img_dir.exists():
        print(f"[skip] {img_dir} does not exist")
        return

    cmd = [
        sys.executable, str(script),
        "--img-path", str(img_dir),
        "--outdir", str(out_dir),
        "--encoder", encoder,
        "--input-size", str(input_size),
    ]
    print("Running:", " ".join(cmd))

    env = os.environ.copy()

    # This is the key path that contains the `depth_anything_v2/` package:
    dav2_root = OMNISR_DIR / "Depth-Anything-V2"

    # Keep OmniSR parent too (harmless, sometimes needed for other imports)
    extra_paths = [str(dav2_root), str(OMNISR_DIR), str(OMNISR_DIR.parent)]

    env["PYTHONPATH"] = os.pathsep.join(
        extra_paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )

    subprocess.check_call(cmd, env=env)



def main():
    # If your dataset is not exactly DATASET_DIR/train|val|test, you can edit this list.
    for split in ["train","test"]:    # adapt it!!
        split_dir = DATASET_DIR / split
        if split_dir.exists():
            run_split(split_dir)
        else:
            print(f"[skip] {split_dir} not found")


if __name__ == "__main__":
    main()
