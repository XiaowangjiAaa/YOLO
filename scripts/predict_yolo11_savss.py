"""Inference script for YOLO11-SAVSS.

Workflow aligned with SCSegamba `test.py` idea:
- Single script for loading checkpoint, running inference, and writing outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict with YOLO11-SAVSS checkpoint")
    p.add_argument("--weights", type=Path, required=True, help="Path to trained .pt weights")
    p.add_argument("--source", type=str, required=True, help="Image/video/folder path")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--project", type=str, default="runs/predict")
    p.add_argument("--name", type=str, default="yolo11_savss")
    p.add_argument("--save-txt", action="store_true")
    p.add_argument("--save-conf", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Ultralytics is required. Install with: pip install ultralytics"
        ) from e

    model = YOLO(str(args.weights))
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=args.project,
        name=args.name,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
    )


if __name__ == "__main__":
    main()
