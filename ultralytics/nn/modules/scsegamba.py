"""High-fidelity SCSegamba-inspired modules for YOLO-style models.

This version upgrades the previous lightweight approximation by replacing
simple cumulative scans with a learnable state-space directional scan.
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


class SAVSS2D(nn.Module):
    """Learnable 2D directional state-space scan.

    For each direction we run a per-channel recurrence:
      h_t = a * h_{t-1} + b * x_t
      y_t = c * h_t + d * x_t
    where a,b,c,d are learnable vectors.
    """

    def __init__(self, c: int, groups: int = 16) -> None:
        super().__init__()
        g = max(1, min(groups, c))
        self.in_proj = nn.Sequential(
            nn.Conv2d(c, c, 1, 1, 0, bias=False),
            nn.GroupNorm(g, c),
            nn.SiLU(inplace=True),
        )

        self.a = nn.Parameter(torch.zeros(4, c))
        self.b = nn.Parameter(torch.ones(4, c))
        self.c = nn.Parameter(torch.ones(4, c))
        self.d = nn.Parameter(torch.ones(4, c))

        self.gate = nn.Sequential(
            nn.Conv2d(c, c, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False),
            nn.Conv2d(c, c, 1, 1, 0, bias=False),
            nn.GroupNorm(g, c),
            nn.SiLU(inplace=True),
        )

    def _scan_sequence(self, seq: torch.Tensor, idx: int) -> torch.Tensor:
        # seq: (B, C, L)
        bsz, channels, length = seq.shape
        a = torch.sigmoid(self.a[idx]).view(1, channels, 1)
        b = self.b[idx].view(1, channels, 1)
        c = self.c[idx].view(1, channels, 1)
        d = self.d[idx].view(1, channels, 1)

        h = torch.zeros((bsz, channels), device=seq.device, dtype=seq.dtype)
        outs = []
        for t in range(length):
            x_t = seq[:, :, t]
            h = a.squeeze(-1) * h + b.squeeze(-1) * x_t
            y_t = c.squeeze(-1) * h + d.squeeze(-1) * x_t
            outs.append(y_t.unsqueeze(-1))
        return torch.cat(outs, dim=-1)

    def _scan_w(self, x: torch.Tensor, reverse: bool, idx: int) -> torch.Tensor:
        # (B,C,H,W) -> flatten rows as sequences
        if reverse:
            x = torch.flip(x, dims=[3])
        bsz, channels, height, width = x.shape
        seq = x.permute(0, 2, 1, 3).reshape(bsz * height, channels, width)
        y = self._scan_sequence(seq, idx).reshape(bsz, height, channels, width).permute(0, 2, 1, 3)
        if reverse:
            y = torch.flip(y, dims=[3])
        return y

    def _scan_h(self, x: torch.Tensor, reverse: bool, idx: int) -> torch.Tensor:
        # (B,C,H,W) -> flatten cols as sequences
        if reverse:
            x = torch.flip(x, dims=[2])
        bsz, channels, height, width = x.shape
        seq = x.permute(0, 3, 1, 2).reshape(bsz * width, channels, height)
        y = self._scan_sequence(seq, idx).reshape(bsz, width, channels, height).permute(0, 2, 3, 1)
        if reverse:
            y = torch.flip(y, dims=[2])
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)

        lr = self._scan_w(x, reverse=False, idx=0)
        rl = self._scan_w(x, reverse=True, idx=1)
        tb = self._scan_h(x, reverse=False, idx=2)
        bt = self._scan_h(x, reverse=True, idx=3)

        fused = 0.25 * (lr + rl + tb + bt)
        fused = fused * self.gate(x)
        return self.out_proj(fused)


class FFN2D(nn.Module):
    def __init__(self, c: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        hidden = int(c * mlp_ratio)
        self.net = nn.Sequential(
            nn.Conv2d(c, hidden, 1, 1, 0),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c, 1, 1, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SAVSSBlock(nn.Module):
    """High-fidelity SAVSS-style block: GBC + directional SSM scan + FFN."""

    def __init__(self, c: int, groups: int = 16, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.gbc = GBC(c, groups=groups)
        self.savss = SAVSS2D(c, groups=groups)
        self.ffn = FFN2D(c, mlp_ratio=mlp_ratio)
        self.norm = nn.GroupNorm(max(1, min(groups, c)), c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gbc(x)
        y = self.savss(y) + y
        y = self.ffn(self.norm(y)) + y
        return y


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


# Compatibility aliases
SASS2D = SAVSS2D
C2fGBC = C2fSAVSS
