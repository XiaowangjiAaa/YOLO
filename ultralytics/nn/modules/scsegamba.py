"""SCSegamba-inspired modules with selective/dynamic SSM options.

- `scan_impl='fast'`: vectorized directional aggregation (fast)
- `scan_impl='ssm'`: directional state-space scan
  - `ssm_mode='static'` : fixed a,b,c,d parameters
  - `ssm_mode='dynamic'`: input-conditioned a,b,c,d (Mamba-like selective)
"""

from __future__ import annotations

import torch
from torch import nn


class BottConv(nn.Module):
    def __init__(self, c1: int, c2: int, cm: int, k: int, s: int = 1, p: int | None = None, bias: bool = True) -> None:
        super().__init__()
        if p is None:
            p = k // 2
        self.pointwise_1 = nn.Conv2d(c1, cm, 1, bias=bias)
        self.depthwise = nn.Conv2d(cm, cm, k, s, p, groups=cm, bias=False)
        self.pointwise_2 = nn.Conv2d(cm, c2, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise_2(self.depthwise(self.pointwise_1(x)))


class GBC(nn.Module):
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
        x1 = self.block2(self.block1(x))
        x2 = self.block3(x)
        return self.block4(x1 * x2) + x


class SAVSS2D(nn.Module):
    """Directional scanner with selectable implementation + SSM mode."""

    def __init__(self, c: int, groups: int = 16, scan_impl: str = "fast", ssm_mode: str = "dynamic") -> None:
        super().__init__()
        self.scan_impl = scan_impl
        self.ssm_mode = ssm_mode
        g = max(1, min(groups, c))
        self.in_proj = nn.Sequential(
            nn.Conv2d(c, c, 1, 1, 0, bias=False),
            nn.GroupNorm(g, c),
            nn.SiLU(inplace=True),
        )
        self.gate = nn.Sequential(nn.Conv2d(c, c, 1, 1, 0, bias=True), nn.Sigmoid())
        self.out_proj = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False),
            nn.Conv2d(c, c, 1, 1, 0, bias=False),
            nn.GroupNorm(g, c),
            nn.SiLU(inplace=True),
        )

        # static parameters
        self.a = nn.Parameter(torch.zeros(4, c))
        self.b = nn.Parameter(torch.ones(4, c))
        self.c = nn.Parameter(torch.ones(4, c))
        self.d = nn.Parameter(torch.ones(4, c))

        # dynamic projections (per direction) for selective scan
        self.dynamic_proj = nn.ModuleList(nn.Conv1d(c, 4 * c, 1, 1, 0) for _ in range(4))

    def _scan_sequence_static(self, seq: torch.Tensor, idx: int) -> torch.Tensor:
        bsz, channels, length = seq.shape
        a = torch.sigmoid(self.a[idx]).view(1, channels)
        b = self.b[idx].view(1, channels)
        c = self.c[idx].view(1, channels)
        d = self.d[idx].view(1, channels)
        h = torch.zeros((bsz, channels), device=seq.device, dtype=seq.dtype)
        outs = []
        for t in range(length):
            x_t = seq[:, :, t]
            h = a * h + b * x_t
            y_t = c * h + d * x_t
            outs.append(y_t.unsqueeze(-1))
        return torch.cat(outs, dim=-1)

    def _scan_sequence_dynamic(self, seq: torch.Tensor, idx: int) -> torch.Tensor:
        # seq: (B, C, L); coefficients are input-conditioned per time step
        coeff = self.dynamic_proj[idx](seq)  # (B, 4C, L)
        a_t, b_t, c_t, d_t = torch.chunk(coeff, 4, dim=1)
        a_t = torch.sigmoid(a_t)

        bsz, channels, length = seq.shape
        h = torch.zeros((bsz, channels), device=seq.device, dtype=seq.dtype)
        outs = []
        for t in range(length):
            x_t = seq[:, :, t]
            h = a_t[:, :, t] * h + b_t[:, :, t] * x_t
            y_t = c_t[:, :, t] * h + d_t[:, :, t] * x_t
            outs.append(y_t.unsqueeze(-1))
        return torch.cat(outs, dim=-1)

    def _scan_sequence(self, seq: torch.Tensor, idx: int) -> torch.Tensor:
        if self.ssm_mode == "dynamic":
            return self._scan_sequence_dynamic(seq, idx)
        return self._scan_sequence_static(seq, idx)

    def _scan_fast(self, x: torch.Tensor) -> torch.Tensor:
        lr = torch.cumsum(x, dim=3)
        rl = torch.flip(torch.cumsum(torch.flip(x, dims=[3]), dim=3), dims=[3])
        tb = torch.cumsum(x, dim=2)
        bt = torch.flip(torch.cumsum(torch.flip(x, dims=[2]), dim=2), dims=[2])
        return 0.25 * (lr + rl + tb + bt)

    def _scan_ssm(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = x.shape
        seq_lr = x.permute(0, 2, 1, 3).reshape(bsz * height, channels, width)
        y_lr = self._scan_sequence(seq_lr, 0).reshape(bsz, height, channels, width).permute(0, 2, 1, 3)

        x_rl = torch.flip(x, dims=[3])
        seq_rl = x_rl.permute(0, 2, 1, 3).reshape(bsz * height, channels, width)
        y_rl = self._scan_sequence(seq_rl, 1).reshape(bsz, height, channels, width).permute(0, 2, 1, 3)
        y_rl = torch.flip(y_rl, dims=[3])

        seq_tb = x.permute(0, 3, 1, 2).reshape(bsz * width, channels, height)
        y_tb = self._scan_sequence(seq_tb, 2).reshape(bsz, width, channels, height).permute(0, 2, 3, 1)

        x_bt = torch.flip(x, dims=[2])
        seq_bt = x_bt.permute(0, 3, 1, 2).reshape(bsz * width, channels, height)
        y_bt = self._scan_sequence(seq_bt, 3).reshape(bsz, width, channels, height).permute(0, 2, 3, 1)
        y_bt = torch.flip(y_bt, dims=[2])

        return 0.25 * (y_lr + y_rl + y_tb + y_bt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        fused = self._scan_fast(x) if self.scan_impl == "fast" else self._scan_ssm(x)
        return self.out_proj(fused * self.gate(x))


class FFN2D(nn.Module):
    def __init__(self, c: int, mlp_ratio: float = 1.5) -> None:
        super().__init__()
        hidden = int(c * mlp_ratio)
        self.net = nn.Sequential(nn.Conv2d(c, hidden, 1), nn.SiLU(inplace=True), nn.Conv2d(hidden, c, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SAVSSBlock(nn.Module):
    def __init__(self, c: int, groups: int = 16, mlp_ratio: float = 1.5, scan_impl: str = "fast", ssm_mode: str = "dynamic") -> None:
        super().__init__()
        self.gbc = GBC(c, groups=groups)
        self.savss = SAVSS2D(c, groups=groups, scan_impl=scan_impl, ssm_mode=ssm_mode)
        self.ffn = FFN2D(c, mlp_ratio=mlp_ratio)
        self.norm = nn.GroupNorm(max(1, min(groups, c)), c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gbc(x)
        y = self.savss(y) + y
        return self.ffn(self.norm(y)) + y


class C2fSAVSS(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5, scan_impl: str = "fast", ssm_mode: str = "dynamic") -> None:
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1)
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(SAVSSBlock(self.c, scan_impl=scan_impl, ssm_mode=ssm_mode) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


SASS2D = SAVSS2D
C2fGBC = C2fSAVSS
