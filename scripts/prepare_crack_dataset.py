"""Convert SCSegamba-style crack dataset layout to YOLO segmentation format.

Expected source layout (same style as SCSegamba):
  <dataset_root>/train_img
  <dataset_root>/train_lab
  <dataset_root>/val_img
  <dataset_root>/val_lab
  <dataset_root>/test_img  (optional)
  <dataset_root>/test_lab  (optional)

Mask convention:
- Crack pixels > 0 are foreground (class id 0 by default).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _iter_images(folder: Path):
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in IMG_EXTS and p.is_file():
            yield p


def _normalize_polygon(contour: np.ndarray, w: int, h: int) -> list[float]:
    pts = contour.reshape(-1, 2).astype(np.float32)
    pts[:, 0] /= max(w, 1)
    pts[:, 1] /= max(h, 1)
    return pts.reshape(-1).tolist()


def mask_to_yolo_segments(mask_path: Path, min_area: float = 8.0) -> list[list[float]]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {mask_path}")

    _, bw = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = bw.shape[:2]

    polys = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or len(c) < 3:
            continue
        poly = _normalize_polygon(c, w, h)
        if len(poly) >= 6:
            polys.append(poly)
    return polys


def convert_split(src_root: Path, dst_root: Path, split: str, class_id: int = 0, min_area: float = 8.0) -> int:
    img_dir = src_root / f"{split}_img"
    lab_dir = src_root / f"{split}_lab"
    if not img_dir.exists() or not lab_dir.exists():
        return 0

    out_img = dst_root / "images" / split
    out_lab = dst_root / "labels" / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in _iter_images(img_dir):
        stem = img_path.stem
        mask_candidates = [lab_dir / f"{stem}.png", lab_dir / f"{stem}.jpg", lab_dir / f"{stem}.bmp"]
        mask_path = next((p for p in mask_candidates if p.exists()), None)
        if mask_path is None:
            continue

        shutil.copy2(img_path, out_img / img_path.name)
        polys = mask_to_yolo_segments(mask_path, min_area=min_area)

        label_path = out_lab / f"{stem}.txt"
        with label_path.open("w", encoding="utf-8") as f:
            for poly in polys:
                line = f"{class_id} " + " ".join(f"{v:.6f}" for v in poly)
                f.write(line + "\n")
        count += 1
    return count


def write_data_yaml(dst_root: Path, class_name: str = "crack") -> Path:
    data = {
        "path": str(dst_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": [class_name],
    }
    yaml_path = dst_root / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return yaml_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert crack mask dataset to YOLO segmentation labels")
    p.add_argument("--src", type=Path, required=True, help="SCSegamba-style dataset root")
    p.add_argument("--dst", type=Path, required=True, help="Output YOLO dataset root")
    p.add_argument("--class-id", type=int, default=0)
    p.add_argument("--class-name", type=str, default="crack")
    p.add_argument("--min-area", type=float, default=8.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.dst.mkdir(parents=True, exist_ok=True)

    stats = {}
    for split in ("train", "val", "test"):
        stats[split] = convert_split(args.src, args.dst, split, class_id=args.class_id, min_area=args.min_area)

    yaml_path = write_data_yaml(args.dst, class_name=args.class_name)
    print(f"[prepare] done. data yaml: {yaml_path}")
    for k, v in stats.items():
        print(f"[prepare] {k}: {v} samples")


if __name__ == "__main__":
    main()
