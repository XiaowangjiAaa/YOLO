"""SCSegamba-inspired modules for YOLO11.

Core idea integration:
- GBC: gated bottleneck convolution for morphology-aware local modeling.
- SASS: structure-aware directional scanning to propagate long-range continuity.
- SAVSSBlock: combines GBC + SASS in one residual block.
"""

from __future__ import annotations

import torch
from torch import nn


class BottConv(nn.Module):
    """Pointwise -> depthwise -> pointwise bottleneck convolution."""

    def __init__(
        self,
        c1: int,
        c2: int,
        cm: int,
        k: int,
        s: int = 1,
        p: int | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if p is None:
            p = k // 2

        self.pointwise_1 = nn.Conv2d(c1, cm, 1, bias=bias)
        self.depthwise = nn.Conv2d(cm, cm, k, s, p, groups=cm, bias=False)
        self.pointwise_2 = nn.Conv2d(cm, c2, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pointwise_1(x)
        x = self.depthwise(x)
        return self.pointwise_2(x)


class GBC(nn.Module):
    """Gated Bottleneck Convolution from SCSegamba."""

    def __init__(self, c: int, mid_ratio: float = 0.125, groups: int = 16, act: nn.Module | None = None) -> None:
        super().__init__()
        cm = max(8, int(c * mid_ratio))
        g = max(1, min(groups, c))
        act = act if act is not None else nn.SiLU(inplace=True)

        self.block1 = nn.Sequential(BottConv(c, c, cm, 3), nn.GroupNorm(g, c), act)
        self.block2 = nn.Sequential(BottConv(c, c, cm, 3), nn.GroupNorm(g, c), act)
        self.block3 = nn.Sequential(BottConv(c, c, cm, 1), nn.GroupNorm(g, c), act)
        self.block4 = nn.Sequential(BottConv(c, c, cm, 1), nn.GroupNorm(g, c), act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x1 = self.block2(self.block1(x))
        x2 = self.block3(x)
        x = self.block4(x1 * x2)
        return x + residual


class SASS2D(nn.Module):
    """Structure-Aware Scanning Strategy (lightweight 2D adaptation).

    We approximate directional selective scanning using cumulative context in
    four directions (left->right, right->left, top->down, bottom->up), then
    gate and fuse the aggregated structural cues.
    """

    def __init__(self, c: int, groups: int = 16) -> None:
        super().__init__()
        g = max(1, min(groups, c))
        self.pre = nn.Sequential(
            nn.Conv2d(c, c, 1, 1, 0, bias=False),
            nn.GroupNorm(g, c),
            nn.SiLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(c, c, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False),
            nn.Conv2d(c, c, 1, 1, 0, bias=False),
            nn.GroupNorm(g, c),
            nn.SiLU(inplace=True),
        )

    @staticmethod
    def _scan(x: torch.Tensor, dim: int, reverse: bool = False) -> torch.Tensor:
        if reverse:
            x = torch.flip(x, dims=[dim])
        y = torch.cumsum(x, dim=dim)
        if reverse:
            y = torch.flip(y, dims=[dim])
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        lr = self._scan(x, dim=3, reverse=False)
        rl = self._scan(x, dim=3, reverse=True)
        tb = self._scan(x, dim=2, reverse=False)
        bt = self._scan(x, dim=2, reverse=True)

        fused = (lr + rl + tb + bt) * 0.25
        gated = fused * self.gate(x)
        return self.mix(gated)


class SAVSSBlock(nn.Module):
    """SCSegamba-style SAVSS block: GBC + SASS with residual connection."""

    def __init__(self, c: int, groups: int = 16) -> None:
        super().__init__()
        self.gbc = GBC(c, groups=groups)
        self.sass = SASS2D(c, groups=groups)
        self.proj = nn.Conv2d(c, c, 1, 1, 0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gbc(x)
        y = self.sass(y)
        return self.proj(y) + x


class C2fSAVSS(nn.Module):
    """YOLO C2f-style wrapper that uses SAVSS blocks."""

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5) -> None:
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1)
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(SAVSSBlock(self.c) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# Backward compatibility with earlier prototype naming.
C2fGBC = C2fSAVSS
