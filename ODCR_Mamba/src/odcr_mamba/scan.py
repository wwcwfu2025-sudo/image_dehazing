from __future__ import annotations

from typing import Tuple

import torch


def diagonal_selective_scan_reference(
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    skip: torch.Tensor,
) -> torch.Tensor:
    """Loop reference for h_t=exp(delta_t A)h_(t-1)+delta_t B_t u_t.

    Shapes: u/delta [B,C,L], a [C,N], b/c [B,N,L], skip [C].
    """
    batch, channels, length = u.shape
    state_dim = a.shape[1]
    state = u.new_zeros(batch, channels, state_dim)
    outputs = []
    for index in range(length):
        dt = delta[:, :, index].unsqueeze(-1)
        transition = torch.exp(dt * a.unsqueeze(0))
        drive = dt * b[:, :, index].unsqueeze(1) * u[:, :, index].unsqueeze(-1)
        state = transition * state + drive
        readout = (state * c[:, :, index].unsqueeze(1)).sum(dim=-1)
        outputs.append(readout + skip.view(1, -1) * u[:, :, index])
    return torch.stack(outputs, dim=-1)


def diagonal_selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    skip: torch.Tensor,
) -> torch.Tensor:
    """Vectorized real diagonal selective scan for short 2-D row/column routes.

    The prefix-product form exactly solves the same recurrence as the reference.
    ODCR only scans rows/columns at reduced resolution, keeping sequence lengths
    short enough to avoid prefix-product underflow in float32.
    """
    # Chunking keeps each prefix product well inside float32 range while looping
    # over only L/32 chunks instead of all L states.
    state = u.new_zeros(u.shape[0], u.shape[1], a.shape[1])
    outputs = []
    chunk_size = 32
    for start in range(0, u.shape[-1], chunk_size):
        end = min(start + chunk_size, u.shape[-1])
        chunk_u = u[:, :, start:end]
        chunk_delta = delta[:, :, start:end]
        chunk_b = b[:, :, start:end]
        chunk_c = c[:, :, start:end]
        transition = torch.exp(
            chunk_delta.unsqueeze(-1) * a.view(1, a.shape[0], 1, a.shape[1])
        )
        drive = (
            chunk_delta.unsqueeze(-1)
            * chunk_b.permute(0, 2, 1).unsqueeze(1)
            * chunk_u.unsqueeze(-1)
        )
        prefix = torch.cumprod(transition, dim=2)
        states = prefix * (
            state.unsqueeze(2) + torch.cumsum(drive / prefix.clamp_min(1.0e-20), dim=2)
        )
        readout = (states * chunk_c.permute(0, 2, 1).unsqueeze(1)).sum(dim=-1)
        outputs.append(readout + skip.view(1, -1, 1) * chunk_u)
        state = states[:, :, -1, :]
    return torch.cat(outputs, dim=-1)


def pack_horizontal(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    batch, channels, height, width = x.shape
    return x.permute(0, 2, 1, 3).reshape(batch * height, channels, width), x.shape


def unpack_horizontal(x: torch.Tensor, shape: Tuple[int, int, int, int]) -> torch.Tensor:
    batch, channels, height, width = shape
    return x.reshape(batch, height, channels, width).permute(0, 2, 1, 3).contiguous()


def pack_vertical(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    batch, channels, height, width = x.shape
    return x.permute(0, 3, 1, 2).reshape(batch * width, channels, height), x.shape


def unpack_vertical(x: torch.Tensor, shape: Tuple[int, int, int, int]) -> torch.Tensor:
    batch, channels, height, width = shape
    return x.reshape(batch, width, channels, height).permute(0, 2, 3, 1).contiguous()

# Generated: 2026-10-01 09:00 +08:00
