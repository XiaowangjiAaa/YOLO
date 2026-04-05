from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class CrackSegDataset(Dataset):
    """Read SCSegamba-style split folders: `<split>_img` + `<split>_lab`."""

    def __init__(self, root: str | Path, split: str, image_size: int = 512) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size

        self.img_dir = self.root / f"{split}_img"
        self.lab_dir = self.root / f"{split}_lab"

        if not self.img_dir.exists() or not self.lab_dir.exists():
            raise FileNotFoundError(f"Missing split folders: {self.img_dir} or {self.lab_dir}")

        self.items: list[tuple[Path, Path]] = []
        for p in sorted(self.img_dir.iterdir()):
            if p.suffix.lower() not in IMG_EXTS or not p.is_file():
                continue
            stem = p.stem
            mask = self._find_mask(stem)
            if mask is not None:
                self.items.append((p, mask))

        if not self.items:
            raise RuntimeError(f"No valid image-mask pairs found in {self.img_dir} and {self.lab_dir}")

    def _find_mask(self, stem: str) -> Path | None:
        for ext in (".png", ".jpg", ".bmp", ".tif", ".tiff"):
            p = self.lab_dir / f"{stem}{ext}"
            if p.exists():
                return p
        return None

    def __len__(self) -> int:
        return len(self.items)

    def _read_image(self, p: Path) -> np.ndarray:
        img = Image.open(p).convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return np.transpose(arr, (2, 0, 1))

    def _read_mask(self, p: Path) -> np.ndarray:
        m = Image.open(p).convert("L").resize((self.image_size, self.image_size), Image.NEAREST)
        arr = (np.asarray(m, dtype=np.float32) > 0).astype(np.float32)
        return arr[None, ...]

    def __getitem__(self, idx: int):
        import torch

        img_p, mask_p = self.items[idx]
        x = torch.from_numpy(self._read_image(img_p))
        y = torch.from_numpy(self._read_mask(mask_p))
        return x, y
