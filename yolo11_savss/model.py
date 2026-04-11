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
    def __init__(self, c: int) -> None:
        super().__init__()
        self.block = nn.Sequential(ConvBNAct(c, c, 3, 1), ConvBNAct(c, c, 3, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + x


def _make_stage(c: int, use_savss: bool, savss_n: int, scan_impl: str):
    if use_savss:
        return C2fSAVSS(c, c, n=savss_n, e=0.5, scan_impl=scan_impl)
    return C2fLite(c)


class UpFuse(nn.Module):
    def __init__(self, c1: int, c_skip: int, c_out: int, use_savss: bool, savss_n: int, scan_impl: str) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.fuse = ConvBNAct(c1 + c_skip, c_out, 3, 1)
        self.mix = _make_stage(c_out, use_savss=use_savss, savss_n=savss_n, scan_impl=scan_impl)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.mix(self.fuse(x))


class YOLO11SAVSSSeg(nn.Module):
    """YOLO11n-oriented SAVSS segmentation model with ablation knobs.

    savss_stages supports subset of: {enc1,enc2,enc3,enc4,up1,up2,up3}
    """

    def __init__(
        self,
        in_ch: int = 3,
        base_ch: int = 16,
        num_classes: int = 1,
        deep_supervision: bool = False,
        scan_impl: str = "fast",
        savss_stages: str = "enc3,up3",
        savss_n: int = 1,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8
        self.deep_supervision = deep_supervision
        stage_set = {s.strip() for s in savss_stages.split(",") if s.strip()}

        self.stem = ConvBNAct(in_ch, c1, 3, 2)
        self.enc1 = _make_stage(c1, "enc1" in stage_set, savss_n, scan_impl)

        self.down2 = ConvBNAct(c1, c2, 3, 2)
        self.enc2 = _make_stage(c2, "enc2" in stage_set, savss_n, scan_impl)

        self.down3 = ConvBNAct(c2, c3, 3, 2)
        self.enc3 = _make_stage(c3, "enc3" in stage_set, savss_n, scan_impl)

        self.down4 = ConvBNAct(c3, c4, 3, 2)
        self.enc4 = _make_stage(c4, "enc4" in stage_set, savss_n, scan_impl)

        self.up3 = UpFuse(c4, c3, c3, use_savss=("up3" in stage_set), savss_n=savss_n, scan_impl=scan_impl)
        self.up2 = UpFuse(c3, c2, c2, use_savss=("up2" in stage_set), savss_n=savss_n, scan_impl=scan_impl)
        self.up1 = UpFuse(c2, c1, c1, use_savss=("up1" in stage_set), savss_n=savss_n, scan_impl=scan_impl)

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
