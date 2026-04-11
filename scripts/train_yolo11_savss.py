"""Strict-aligned training entrypoint (SCSegamba-style argument/strategy compatible)."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

# 将项目根目录加入系统路径
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO11-SAVSS (strict aligned mode)")

    # ---- SCSegamba-compatible names (primary) ----
    p.add_argument("--dataroot", type=Path, default=None, help="SCSegamba-style dataset root")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--BCELoss_ratio", type=float, default=0.5)
    p.add_argument("--DiceLoss_ratio", type=float, default=0.5)
    p.add_argument("--lr_scheduler", type=str, default="PolyLR", choices=["PolyLR", "Cosine"])
    p.add_argument("--model_path", type=Path, default=Path("runs/local_yolo11_savss"))

    # ---- Aliases from previous local versions ----
    p.add_argument("--dataset-root", dest="dataset_root_alias", type=Path, default=None)
    p.add_argument("--batch-size", dest="batch_size_alias", type=int, default=None)
    p.add_argument("--BCELoss-ratio", dest="bce_ratio_alias", type=float, default=None)
    p.add_argument("--DiceLoss-ratio", dest="dice_ratio_alias", type=float, default=None)
    p.add_argument("--save-dir", dest="save_dir_alias", type=Path, default=None)

    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--base-ch", type=int, default=16)
    p.add_argument("--poly-power", type=float, default=0.9)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--optim", type=str, default="adamw", choices=["adamw", "sgd"])
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)

    group = p.add_mutually_exclusive_group()
    group.add_argument("--deep-supervision", dest="deep_supervision", action="store_true")
    group.add_argument("--no-deep-supervision", dest="deep_supervision", action="store_false")
    p.set_defaults(deep_supervision=False)
    p.add_argument("--aux-weight", type=float, default=0.2)

    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--vis-interval", type=int, default=5)
    p.add_argument("--num-vis-samples", type=int, default=4)
    p.add_argument("--early-stop-patience", type=int, default=50, help="stop if no val_iou improvement for N epochs")
    p.add_argument("--amp", action="store_true", help="Enable torch AMP on CUDA")
    p.add_argument("--scan-impl", type=str, default="fast", choices=["fast", "ssm"], help="fast=for speed, ssm=for fidelity")
    p.add_argument("--savss-stages", type=str, default="enc3,up3", help="comma list: enc1,enc2,enc3,enc4,up1,up2,up3")
    p.add_argument("--savss-n", type=int, default=1, help="number of SAVSS blocks inside each replaced stage")

    args = p.parse_args()

    if args.dataset_root_alias is not None:
        args.dataroot = args.dataset_root_alias
    if args.dataroot is None:
        raise ValueError("Please provide --dataroot or --dataset-root")

    if args.batch_size_alias is not None:
        args.batch_size = args.batch_size_alias
    if args.bce_ratio_alias is not None:
        args.BCELoss_ratio = args.bce_ratio_alias
    if args.dice_ratio_alias is not None:
        args.DiceLoss_ratio = args.dice_ratio_alias
    if args.save_dir_alias is not None:
        args.model_path = args.save_dir_alias

    return args


def set_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_loss_from_logits(logits, target, smooth: float = 1e-6):
    import torch

    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * inter + smooth) / (denom + smooth)
    return 1.0 - dice.mean()


def hybrid_loss(logits, target, bce_ratio: float, dice_ratio: float):
    import torch

    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    dloss = dice_loss_from_logits(logits, target)
    total = bce_ratio * bce + dice_ratio * dloss
    return total, bce.detach(), dloss.detach()


def compute_total_loss(outputs, target, bce_ratio: float, dice_ratio: float, aux_weight: float = 0.4):
    if isinstance(outputs, tuple):
        main, *aux = outputs
    else:
        main, aux = outputs, []

    total, bce, dloss = hybrid_loss(main, target, bce_ratio, dice_ratio)
    if aux:
        aux_total = 0.0
        for out in aux:
            l_aux, _, _ = hybrid_loss(out, target, bce_ratio, dice_ratio)
            aux_total = aux_total + l_aux
        total = total + aux_weight * (aux_total / len(aux))
    return total, bce, dloss, main


def poly_lr(base_lr: float, cur_iter: int, max_iter: int, power: float, min_lr: float = 0.0) -> float:
    if max_iter <= 0:
        return base_lr
    factor = (1.0 - float(cur_iter) / float(max_iter)) ** power
    return max(min_lr, base_lr * factor)


def _dice_iou_from_logits(logits, target):
    import torch

    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    pred_sum = pred.sum(dim=(1, 2, 3))
    tgt_sum = target.sum(dim=(1, 2, 3))
    dice = (2 * inter + 1e-6) / (pred_sum + tgt_sum + 1e-6)
    union = pred_sum + tgt_sum - inter
    iou = (inter + 1e-6) / (union + 1e-6)
    return dice.mean().item(), iou.mean().item()


def _save_val_visualizations(model, loader, device, save_dir: Path, epoch: int, num_samples: int = 4):
    import numpy as np
    import torch
    from PIL import Image

    model.eval()
    vis_dir = save_dir / "val_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            outputs = model(x)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            pred = (torch.sigmoid(logits) > 0.5).float()
            x_np = (x.detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
            y_np = (y.detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
            p_np = (pred.detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
            for i in range(x_np.shape[0]):
                panel = np.concatenate(
                    [
                        np.transpose(x_np[i], (1, 2, 0)),
                        np.stack([y_np[i, 0]] * 3, axis=-1),
                        np.stack([p_np[i, 0]] * 3, axis=-1),
                    ],
                    axis=1,
                )
                Image.fromarray(panel).save(vis_dir / f"epoch{epoch:03d}_sample{saved:03d}.png")
                saved += 1
                if saved >= num_samples:
                    return


def _build_progress_iter(loader, epoch: int, epochs: int):
    try:
        from tqdm import tqdm

        return tqdm(loader, desc=f"Epoch {epoch}/{epochs}", dynamic_ncols=True)
    except Exception:
        return loader


def _append_csv_log(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def estimate_conv_flops(model, image_size: int, device):
    import torch
    from torch import nn

    flops = 0
    hooks = []

    def hook_fn(module, inp, out):
        nonlocal flops
        if not isinstance(module, nn.Conv2d):
            return
        x = inp[0]
        batch = x.shape[0]
        out_h, out_w = out.shape[-2:]
        k_h, k_w = module.kernel_size
        in_c = module.in_channels
        out_c = module.out_channels
        groups = module.groups
        conv_per_position = (in_c // groups) * k_h * k_w * out_c
        flops += batch * out_h * out_w * conv_per_position

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(hook_fn))

    with torch.no_grad():
        dummy = torch.randn(1, 3, image_size, image_size, device=device)
        _ = model(dummy)

    for h in hooks:
        h.remove()
    return float(flops)


def main() -> None:
    args = parse_args()
    import torch
    from torch.utils.data import DataLoader

    from yolo11_savss.data import CrackSegDataset
    from yolo11_savss.model import YOLO11SAVSSSeg

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.model_path.mkdir(parents=True, exist_ok=True)

    train_ds = CrackSegDataset(args.dataroot, split="train", image_size=args.image_size)
    val_ds = CrackSegDataset(args.dataroot, split="val", image_size=args.image_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = YOLO11SAVSSSeg(
        base_ch=args.base_ch,
        deep_supervision=args.deep_supervision,
        scan_impl=args.scan_impl,
        savss_stages=args.savss_stages,
        savss_n=args.savss_n,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    est_flops = estimate_conv_flops(model, args.image_size, device)
    print(f"[model] params={n_params/1e6:.3f}M, est_flops={est_flops/1e9:.3f}G, scan_impl={args.scan_impl}, stages={args.savss_stages}, savss_n={args.savss_n}, deep_supervision={args.deep_supervision}")

    if args.optim == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    total_iters = args.epochs * max(1, len(train_loader))
    cur_iter = 0
    best_iou = -1.0
    best_epoch = 0
    best_path = args.model_path / "best.pt"
    last_path = args.model_path / "last.pt"
    log_csv = args.model_path / "train_log.csv"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = train_bce = train_dice_loss = 0.0
        progress_iter = _build_progress_iter(train_loader, epoch, args.epochs)

        for step, (x, y) in enumerate(progress_iter, start=1):
            cur_iter += 1
            if args.lr_scheduler == "PolyLR":
                lr = poly_lr(args.lr, cur_iter, total_iters, args.poly_power, args.min_lr)
                for g in optimizer.param_groups:
                    g["lr"] = lr

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                outputs = model(x)
                loss, bce_v, dloss_v, logits = compute_total_loss(
                    outputs,
                    y,
                    bce_ratio=args.BCELoss_ratio,
                    dice_ratio=args.DiceLoss_ratio,
                    aux_weight=args.aux_weight,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = x.size(0)
            train_loss += loss.item() * bs
            train_bce += bce_v.item() * bs
            train_dice_loss += dloss_v.item() * bs

            if hasattr(progress_iter, "set_postfix"):
                progress_iter.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            elif step % max(1, args.log_interval) == 0:
                print(f"[epoch {epoch:03d} step {step:04d}/{len(train_loader):04d}] loss={loss.item():.4f}")

        n_train = len(train_loader.dataset)
        train_loss /= n_train
        train_bce /= n_train
        train_dice_loss /= n_train

        # Optional non-Poly schedule (epoch-level)
        if args.lr_scheduler == "Cosine":
            cos_factor = 0.5 * (1 + torch.cos(torch.tensor(epoch / args.epochs * 3.1415926535))).item()
            lr = max(args.min_lr, args.lr * cos_factor)
            for g in optimizer.param_groups:
                g["lr"] = lr

        model.eval()
        val_loss = val_bce = val_dice_loss = val_dice = val_iou = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                outputs = model(x)
                loss, bce_v, dloss_v, logits = compute_total_loss(
                    outputs,
                    y,
                    bce_ratio=args.BCELoss_ratio,
                    dice_ratio=args.DiceLoss_ratio,
                    aux_weight=args.aux_weight,
                )
                dice, iou = _dice_iou_from_logits(logits, y)

                bs = x.size(0)
                val_loss += loss.item() * bs
                val_bce += bce_v.item() * bs
                val_dice_loss += dloss_v.item() * bs
                val_dice += dice * bs
                val_iou += iou * bs

        n_val = len(val_loader.dataset)
        val_loss /= n_val
        val_bce /= n_val
        val_dice_loss /= n_val
        val_dice /= n_val
        val_iou /= n_val

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_iou": best_iou,
            "args": vars(args),
        }
        torch.save(ckpt, last_path)
        if val_iou > best_iou:
            best_iou = val_iou
            best_epoch = epoch
            ckpt["best_iou"] = best_iou
            torch.save(ckpt, best_path)

        if args.vis_interval > 0 and (epoch % args.vis_interval == 0 or epoch == 1 or epoch == args.epochs):
            _save_val_visualizations(model, val_loader, device, args.model_path, epoch, args.num_vis_samples)

        elapsed = time.time() - t0
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_bce": train_bce,
            "train_dice_loss": train_dice_loss,
            "val_loss": val_loss,
            "val_bce": val_bce,
            "val_dice_loss": val_dice_loss,
            "val_dice": val_dice,
            "val_iou": val_iou,
            "elapsed_sec": elapsed,
        }
        _append_csv_log(log_csv, row)

        print(
            f"[epoch {epoch:03d}] time={elapsed:.1f}s lr={optimizer.param_groups[0]['lr']:.6g} "
            f"train={train_loss:.4f} val={val_loss:.4f} val_dice={val_dice:.4f} val_iou={val_iou:.4f}"
        )

        no_improve = epoch - best_epoch
        if no_improve >= args.early_stop_patience:
            print(f"[early-stop] no val_iou improvement for {no_improve} epochs (patience={args.early_stop_patience}), stop at epoch {epoch}.")
            break

    summary_path = args.model_path / "run_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["best_miou", "best_epoch", "params", "est_flops", "scan_impl", "savss_stages", "savss_n", "deep_supervision"])
        w.writeheader()
        w.writerow({
            "best_miou": f"{best_iou:.6f}",
            "best_epoch": best_epoch,
            "params": n_params,
            "est_flops": int(est_flops),
            "scan_impl": args.scan_impl,
            "savss_stages": args.savss_stages,
            "savss_n": args.savss_n,
            "deep_supervision": args.deep_supervision,
        })

    print(f"[done] best_iou={best_iou:.4f} best_ckpt={best_path} best_epoch={best_epoch} summary={summary_path}")


if __name__ == "__main__":
    main()
