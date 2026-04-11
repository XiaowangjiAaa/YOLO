"""Run SAVSS ablation experiments (edge-focused default plan)."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SAVSS ablation plan")
    p.add_argument("--config", type=Path, default=Path("configs/savss_ablation.json"))
    p.add_argument("--project", type=Path, default=Path("runs/ablation_savss"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_from_experiments(base: dict, experiments: list[dict], project: Path):
    jobs = []
    for i, exp in enumerate(experiments, start=1):
        name = exp.get("name", f"exp_{i:03d}")
        jobs.append(
            {
                "name": name,
                "run_dir": project / f"{i:03d}_{name}",
                "savss_stages": str(exp.get("savss_stages", "enc3,up3")),
                "savss_n": int(exp.get("savss_n", 1)),
                "scan_impl": str(exp.get("scan_impl", base.get("scan_impl", "fast"))),
            }
        )
    return jobs


def build_from_grid(base: dict, grid: dict, project: Path):
    stage_list = grid.get("savss_stages", ["enc3,up3"])
    n_list = grid.get("savss_n", [1])
    scan_list = grid.get("scan_impl", [base.get("scan_impl", "fast")])
    jobs = []
    idx = 0
    for stages, n, scan_impl in itertools.product(stage_list, n_list, scan_list):
        idx += 1
        jobs.append(
            {
                "name": f"exp_{idx:03d}_{scan_impl}_n{n}",
                "run_dir": project / f"exp_{idx:03d}_{scan_impl}_n{n}",
                "savss_stages": str(stages),
                "savss_n": int(n),
                "scan_impl": str(scan_impl),
            }
        )
    return jobs


def main() -> None:
    args = parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    base = cfg.get("base", {})

    args.project.mkdir(parents=True, exist_ok=True)

    if "experiments" in cfg:
        jobs = build_from_experiments(base, cfg["experiments"], args.project)
    else:
        jobs = build_from_grid(base, cfg.get("grid", {}), args.project)

    rows = []

    for job in jobs:
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
            job["savss_stages"],
            "--savss-n",
            str(job["savss_n"]),
            "--scan-impl",
            job["scan_impl"],
            "--model_path",
            str(job["run_dir"]),
        ]

        if bool(base.get("amp", True)):
            cmd.append("--amp")

        prefix = "[dry-run]" if args.dry_run else "[run]"
        print(prefix, job["name"], "::", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)
            summary_file = job["run_dir"] / "run_summary.csv"
            if summary_file.exists():
                with summary_file.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        r["experiment"] = job["name"]
                        rows.append(r)
            else:
                rows.append({
                    "experiment": job["name"],
                    "best_miou": "",
                    "best_epoch": "",
                    "params": "",
                    "est_flops": "",
                    "scan_impl": job["scan_impl"],
                    "savss_stages": job["savss_stages"],
                    "savss_n": job["savss_n"],
                    "deep_supervision": "",
                })

    if not args.dry_run and rows:
        out_csv = args.project / "ablation_summary.csv"
        keys = ["experiment", "best_miou", "best_epoch", "params", "est_flops", "scan_impl", "savss_stages", "savss_n", "deep_supervision"]
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[summary] wrote {out_csv}")


if __name__ == "__main__":
    main()
