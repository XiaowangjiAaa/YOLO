"""Run SAVSS ablation experiments from a JSON grid."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SAVSS ablation grid")
    p.add_argument("--config", type=Path, default=Path("configs/savss_ablation.json"))
    p.add_argument("--project", type=Path, default=Path("runs/ablation_savss"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    base = cfg.get("base", {})
    grid = cfg.get("grid", {})

    stage_list = grid.get("savss_stages", ["enc2,enc3,up3"])
    n_list = grid.get("savss_n", [1])
    scan_list = grid.get("scan_impl", ["fast"])

    args.project.mkdir(parents=True, exist_ok=True)

    idx = 0
    for stages, n, scan_impl in itertools.product(stage_list, n_list, scan_list):
        idx += 1
        run_dir = args.project / f"exp_{idx:03d}_{scan_impl}_n{n}"
        cmd = [
            sys.executable,
            "scripts/train_yolo11_savss.py",
            "--dataroot",
            str(base.get("dataroot", "")),
            "--epochs",
            str(base.get("epochs", 80)),
            "--batch_size",
            str(base.get("batch_size", 8)),
            "--lr",
            str(base.get("lr", 1e-3)),
            "--BCELoss_ratio",
            str(base.get("BCELoss_ratio", 0.5)),
            "--DiceLoss_ratio",
            str(base.get("DiceLoss_ratio", 0.5)),
            "--lr_scheduler",
            str(base.get("lr_scheduler", "PolyLR")),
            "--base-ch",
            str(base.get("base_ch", 16)),
            "--savss-stages",
            stages,
            "--savss-n",
            str(n),
            "--scan-impl",
            scan_impl,
            "--model_path",
            str(run_dir),
        ]

        if bool(base.get("amp", True)):
            cmd.append("--amp")

        if args.dry_run:
            print("[dry-run]", " ".join(cmd))
        else:
            print("[run]", " ".join(cmd))
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
