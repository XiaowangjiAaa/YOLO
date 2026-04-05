"""Train YOLO11 with SAVSS-style blocks.

Workflow inspired by SCSegamba's `main.py` style:
1) Parse centralized runtime args.
2) Build/prepare dataset.
3) Train and validate in one entrypoint.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO11-SAVSS on crack dataset")
    p.add_argument("--dataset-root", type=Path, required=True, help="Source root with train_img/train_lab etc.")
    p.add_argument("--workdir", type=Path, default=Path("runs/savss"), help="Run output directory")
    p.add_argument("--prepared-dataset", type=Path, default=Path("datasets/crack_yolo"))
    p.add_argument("--model-cfg", type=Path, default=Path("ultralytics/cfg/models/11/yolo11-scgb.yaml"))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--project", type=str, default="runs/train")
    p.add_argument("--name", type=str, default="yolo11_savss")
    p.add_argument("--prepare-only", action="store_true")
    return p.parse_args()


def prepare_dataset(src_root: Path, dst_root: Path) -> Path:
    cmd = [
        sys.executable,
        "scripts/prepare_crack_dataset.py",
        "--src",
        str(src_root),
        "--dst",
        str(dst_root),
        "--class-id",
        "0",
        "--class-name",
        "crack",
    ]
    subprocess.run(cmd, check=True)
    return dst_root / "data.yaml"


def main() -> None:
    args = parse_args()
    data_yaml = prepare_dataset(args.dataset_root, args.prepared_dataset)

    if args.prepare_only:
        print(f"[train] dataset prepared at: {data_yaml}")
        return

    try:
        from ultralytics import YOLO
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Ultralytics is required. Install with: pip install ultralytics"
        ) from e

    model = YOLO(str(args.model_cfg))
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
