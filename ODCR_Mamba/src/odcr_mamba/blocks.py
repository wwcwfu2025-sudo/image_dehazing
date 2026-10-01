from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F

from .scan import (
    diagonal_selective_scan,
    pack_horizontal,
    pack_vertical,
    unpack_horizontal,
    unpack_vertical,
)


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        y = (x - mean) * torch.rsqrt(var + self.eps)
        return y * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

class OpticalDepthConditionedSS2D(nn.Module):
    """Four-route selective state-space layer with OD-conditioned step size."""

    def __init__(
        self,
        channels: int,
        state_dim: int,
        conditioned: bool,
        alpha_init: float,
        delta_min: float,
        delta_max: float,
        depth_modulation: bool = False,
        adaptive_gate: bool = False,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.state_dim = state_dim
        self.conditioned = conditioned
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.depth_modulation = depth_modulation
        self.adaptive_gate = adaptive_gate
        self.norm = LayerNorm2d(channels)
        self.in_proj = nn.Conv2d(channels, channels * 2, 1)
        self.dwconv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.delta_proj = nn.Conv2d(channels, channels, 1)
        self.bc_proj = nn.Conv2d(channels, state_dim * 2, 1)
        self.route_bias = nn.Parameter(torch.zeros(4, channels))
        self.a_log = nn.Parameter(torch.log(torch.arange(1, state_dim + 1).float()).repeat(channels, 1))
        self.skip = nn.Parameter(torch.ones(channels))
        self.alpha_raw = nn.Parameter(torch.tensor(math.log(math.expm1(alpha_init))))
        if depth_modulation:
            self.depth_gain_raw = nn.Parameter(torch.tensor(math.log(math.expm1(0.3))))
        if adaptive_gate:
            self.condition_gate = nn.Sequential(nn.Linear(2, 8), nn.SiLU(), nn.Linear(8, 1))
        self.local = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        self.collect_debug = False
        self.last_debug = {}

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.alpha_raw)

    def _route(
        self,
        u: torch.Tensor,
        raw_delta: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        depth: torch.Tensor,
        route: int,
        vertical: bool,
        reverse: bool,
    ) -> torch.Tensor:
        pack = pack_vertical if vertical else pack_horizontal
        unpack = unpack_vertical if vertical else unpack_horizontal
        seq_u, shape = pack(u)
        seq_delta, _ = pack(raw_delta)
        seq_b, _ = pack(b)
        seq_c, _ = pack(c)
        seq_depth, _ = pack(depth)
        if reverse:
            seq_u = seq_u.flip(-1)
            seq_delta = seq_delta.flip(-1)
            seq_b = seq_b.flip(-1)
            seq_c = seq_c.flip(-1)
            seq_depth = seq_depth.flip(-1)
        delta = self.delta_min + (self.delta_max - self.delta_min) * torch.sigmoid(
            seq_delta + self.route_bias[route].view(1, -1, 1)
        )
        if self.conditioned:
            condition = seq_depth
            if self.depth_modulation:
                mean_depth = condition.mean(dim=-1, keepdim=True)
                std_depth = condition.std(dim=-1, keepdim=True).clamp_min(1.0e-3)
                relative = (condition - mean_depth) / (std_depth + 0.1)
                if self.adaptive_gate:
                    statistics = torch.cat([mean_depth, std_depth], dim=1).squeeze(-1)
                    condition_gate = torch.sigmoid(self.condition_gate(statistics)).view(-1, 1, 1)
                    if self.collect_debug:
                        self._debug_route_gates.append(float(condition_gate.mean().detach()))
                else:
                    condition_gate = 1.0
                    condition = condition / mean_depth.clamp_min(1.0e-4)
                seq_u = seq_u * (
                    1.0 + F.softplus(self.depth_gain_raw) * condition_gate * torch.tanh(relative)
                )
            else:
                condition_gate = 1.0
            delta = delta / (1.0 + self.alpha * condition_gate * condition)
        a = -torch.exp(self.a_log.float())
        scanned = diagonal_selective_scan(
            seq_u.float(), delta.float(), a, seq_b.float(), seq_c.float(), self.skip.float()
        ).to(u.dtype)
        if reverse:
            scanned = scanned.flip(-1)
        return unpack(scanned, shape)

    def forward(self, x: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        shortcut = x
        if self.collect_debug:
            self._debug_route_gates = []
        x = self.norm(x)
        u, gate = self.in_proj(x).chunk(2, dim=1)
        u = F.silu(self.dwconv(u))
        raw_delta = self.delta_proj(u)
        b, c = self.bc_proj(u).chunk(2, dim=1)
        b = torch.tanh(b)
        c = torch.tanh(c)
        depth = F.interpolate(depth, size=u.shape[-2:], mode="bilinear", align_corners=False)
        routes = [
            self._route(u, raw_delta, b, c, depth, 0, False, False),
            self._route(u, raw_delta, b, c, depth, 1, False, True),
            self._route(u, raw_delta, b, c, depth, 2, True, False),
            self._route(u, raw_delta, b, c, depth, 3, True, True),
        ]
        global_context = sum(routes) * 0.25
        local_context = F.silu(self.local(u))
        if self.collect_debug:
            self.last_debug = {
                "global_rms": float(global_context.float().pow(2).mean().sqrt().detach()),
                "local_rms": float(local_context.float().pow(2).mean().sqrt().detach()),
                "depth_std": float(depth.float().std().detach()),
                "alpha": float(self.alpha.detach()),
            }
            if self._debug_route_gates:
                self.last_debug["condition_gate"] = float(sum(self._debug_route_gates) / len(self._debug_route_gates))
        mixed = global_context + local_context
        return shortcut + self.out_proj(mixed * torch.sigmoid(gate))

class ConvFFN(nn.Module):
    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.norm = LayerNorm2d(channels)
        self.in_proj = nn.Conv2d(channels, hidden * 2, 1)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.out_proj = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, gate = self.in_proj(self.norm(x)).chunk(2, dim=1)
        return x + self.out_proj(F.silu(self.dwconv(a)) * torch.sigmoid(gate))

class ODCBlock(nn.Module):
    def __init__(self, channels: int, state_dim: int, conditioned: bool, **kwargs: float) -> None:
        super().__init__()
        self.ssm = OpticalDepthConditionedSS2D(channels, state_dim, conditioned, **kwargs)
        self.ffn = ConvFFN(channels)

    def forward(self, x: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.ssm(x, depth))

class LocalBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, channels * 2, 1)
        self.dw = nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2)
        self.pw2 = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pw2(F.gelu(self.dw(self.pw1(self.norm(x)))))

# Generated: 2026-10-01 09:00 +08:00
