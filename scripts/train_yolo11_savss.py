"""Train local YOLO11-SAVSS model without external ultralytics package."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train local YOLO11-SAVSS segmentation model")
    p.add_argument("--dataset-root", type=Path, required=True, help="Root with train_img/train_lab and val_img/val_lab")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--save-dir", type=Path, default=Path("runs/local_yolo11_savss"))
    return p.parse_args()


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


def main() -> None:
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader

    from yolo11_savss.data import CrackSegDataset
    from yolo11_savss.model import YOLO11SAVSSSeg

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.save_dir.mkdir(parents=True, exist_ok=True)

    train_ds = CrackSegDataset(args.dataset_root, split="train", image_size=args.image_size)
    val_ds = CrackSegDataset(args.dataset_root, split="val", image_size=args.image_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = YOLO11SAVSSSeg(base_ch=args.base_ch).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_iou = -1.0
    best_path = args.save_dir / "best.pt"
    last_path = args.save_dir / "last.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                logits = model(x)
                loss = criterion(logits, y)
                dice, iou = _dice_iou_from_logits(logits, y)

                val_loss += loss.item() * x.size(0)
                val_dice += dice * x.size(0)
                val_iou += iou * x.size(0)

        val_loss /= len(val_loader.dataset)
        val_dice /= len(val_loader.dataset)
        val_iou /= len(val_loader.dataset)

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
            ckpt["best_iou"] = best_iou
            torch.save(ckpt, best_path)

        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} val_iou={val_iou:.4f}"
        )

    print(f"[done] best_iou={best_iou:.4f} best_ckpt={best_path}")


if __name__ == "__main__":
    main()
