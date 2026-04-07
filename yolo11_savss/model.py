from __future__ import annotations

import torch
from torch import nn

from ultralytics.nn.modules import C2fSAVSS


class ConvBNAct(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p: int | None = None) -> None:
        super().__init__()
        if p is None:
            p = k // 2
        self.block = nn.Sequential(nn.Conv2d(c1, c2, k, s, p, bias=False), nn.BatchNorm2d(c2), nn.SiLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class C2fLite(nn.Module):
    """Cheap fallback when a stage does not use SAVSS."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(c, c, 3, 1),
            ConvBNAct(c, c, 3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + x


class UpFuse(nn.Module):
    def __init__(self, c1: int, c_skip: int, c_out: int, use_savss: bool = True, scan_impl: str = "fast") -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.fuse = ConvBNAct(c1 + c_skip, c_out, 3, 1)
        self.mix = C2fSAVSS(c_out, c_out, n=1, e=0.5, scan_impl=scan_impl) if use_savss else C2fLite(c_out)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.mix(self.fuse(x))


class YOLO11SAVSSSeg(nn.Module):
    """YOLO11n-oriented SAVSS segmentation model.

    Uses SAVSS on selected stages only to keep parameter/FPS budget small.
    """

    def __init__(
        self,
        in_ch: int = 3,
        base_ch: int = 16,
        num_classes: int = 1,
        deep_supervision: bool = False,
        scan_impl: str = "fast",
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8
        self.deep_supervision = deep_supervision

        self.stem = ConvBNAct(in_ch, c1, 3, 2)
        self.enc1 = C2fLite(c1)  # keep cheap

        self.down2 = ConvBNAct(c1, c2, 3, 2)
        self.enc2 = C2fSAVSS(c2, c2, n=1, scan_impl=scan_impl)  # replace mid stage

        self.down3 = ConvBNAct(c2, c3, 3, 2)
        self.enc3 = C2fSAVSS(c3, c3, n=1, scan_impl=scan_impl)  # replace deep stage

        self.down4 = ConvBNAct(c3, c4, 3, 2)
        self.enc4 = C2fLite(c4)  # keep head speed

        self.up3 = UpFuse(c4, c3, c3, use_savss=True, scan_impl=scan_impl)
        self.up2 = UpFuse(c3, c2, c2, use_savss=False, scan_impl=scan_impl)
        self.up1 = UpFuse(c2, c1, c1, use_savss=False, scan_impl=scan_impl)

        self.head = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(c1, num_classes, 1))
        self.aux3 = nn.Conv2d(c3, num_classes, 1)

    def forward(self, x: torch.Tensor):
        x1 = self.enc1(self.stem(x))
        x2 = self.enc2(self.down2(x1))
        x3 = self.enc3(self.down3(x2))
        x4 = self.enc4(self.down4(x3))

        y3 = self.up3(x4, x3)
        y2 = self.up2(y3, x2)
        y1 = self.up1(y2, x1)
        out = self.head(y1)

        if not self.deep_supervision:
            return out

        aux3 = nn.functional.interpolate(self.aux3(y3), size=out.shape[-2:], mode="bilinear", align_corners=False)
        return out, aux3
