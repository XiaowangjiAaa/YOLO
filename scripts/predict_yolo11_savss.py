"""Run inference with locally trained YOLO11-SAVSS checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path



IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict with local YOLO11-SAVSS checkpoint")
    p.add_argument("--weights", type=Path, required=True, help="Path to best.pt/last.pt")
    p.add_argument("--source", type=Path, required=True, help="Image path or directory")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save-dir", type=Path, default=Path("runs/local_yolo11_savss_predict"))
    return p.parse_args()


def iter_images(source: Path):
    if source.is_file():
        yield source
        return
    for p in sorted(source.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def main() -> None:
    args = parse_args()

    import numpy as np
    import torch
    from PIL import Image

    from yolo11_savss.model import YOLO11SAVSSSeg

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.save_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.weights, map_location="cpu")
    saved_args = ckpt.get("args", {})
    base_ch = int(saved_args.get("base_ch", 16))
    scan_impl = str(saved_args.get("scan_impl", "fast"))

    model = YOLO11SAVSSSeg(base_ch=base_ch, deep_supervision=False, scan_impl=scan_impl).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    with torch.no_grad():
        for img_path in iter_images(args.source):
            img = Image.open(img_path).convert("RGB")
            raw_w, raw_h = img.size
            inp = img.resize((args.image_size, args.image_size), Image.BILINEAR)
            arr = np.asarray(inp, dtype=np.float32) / 255.0
            x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)

            logits = model(x)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            mask = (prob > args.threshold).astype("uint8") * 255

            mask_img = Image.fromarray(mask).resize((raw_w, raw_h), Image.NEAREST)
            out_path = args.save_dir / f"{img_path.stem}_mask.png"
            mask_img.save(out_path)
            print(f"[predict] {img_path} -> {out_path}")


if __name__ == "__main__":
    main()
