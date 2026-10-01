from __future__ import annotations

from typing import Dict
import math

import torch

from dataclasses import dataclass


@dataclass(frozen=True)
class LossConfig:
    """Loss coefficients used by the fixed standalone training configuration."""
    lambda_direct: float = 0.20
    lambda_physical: float = 0.05
    lambda_redegrade: float = 0.20
    lambda_depth_smooth: float = 0.01
    lambda_depth_supervision: float = 0.10
    lambda_airlight_supervision: float = 0.02
    lambda_cvar: float = 0.30
    cvar_fraction: float = 0.20



def charbonnier(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-3) -> torch.Tensor:
    return torch.sqrt((prediction - target).pow(2) + eps * eps).mean()


def edge_aware_depth_smoothness(depth: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    dx_depth = (depth[:, :, :, 1:] - depth[:, :, :, :-1]).abs()
    dy_depth = (depth[:, :, 1:, :] - depth[:, :, :-1, :]).abs()
    dx_image = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean(1, keepdim=True)
    dy_image = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs().mean(1, keepdim=True)
    return (dx_depth * torch.exp(-5.0 * dx_image)).mean() + (dy_depth * torch.exp(-5.0 * dy_image)).mean()


@torch.no_grad()
def paired_atmospheric_targets(hazy: torch.Tensor, clear: torch.Tensor):
    """Estimate global A and optical depth from I=A+t(J-A) on paired training data."""
    batch, _, height, width = hazy.shape
    dark = hazy.min(dim=1)[0].flatten(1)
    count = max(1, height * width // 100)
    indices = dark.topk(count, dim=1).indices
    flat = hazy.flatten(2)
    gather_index = indices.unsqueeze(1).expand(batch, 3, count)
    airlight = torch.gather(flat, 2, gather_index).mean(dim=2, keepdim=True).unsqueeze(-1)
    clear_offset = clear - airlight
    hazy_offset = hazy - airlight
    transmission = (hazy_offset * clear_offset).sum(1, keepdim=True) / clear_offset.pow(2).sum(1, keepdim=True).clamp_min(1.0e-4)
    transmission = transmission.clamp(0.08, 1.0)
    depth = -torch.log(transmission)
    depth = torch.nn.functional.avg_pool2d(depth, kernel_size=7, stride=1, padding=3)
    return depth, airlight


def reconstruction_loss(
    result: Dict[str, torch.Tensor],
    hazy: torch.Tensor,
    target: torch.Tensor,
    config: LossConfig,
) -> Dict[str, torch.Tensor]:
    main = charbonnier(result["output"], target)
    direct = charbonnier(result["direct"], target)
    physical = charbonnier(result["physical"], target)
    rehazed = result["output"] * result["transmission"] + result["airlight"] * (1.0 - result["transmission"])
    redegrade = charbonnier(rehazed, hazy)
    smooth = edge_aware_depth_smoothness(result["depth"], hazy)
    target_depth, target_airlight = paired_atmospheric_targets(hazy, target)
    depth_supervision = charbonnier(result["depth"], target_depth)
    airlight_supervision = charbonnier(result["airlight"], target_airlight)
    per_sample = torch.sqrt((result["output"] - target).pow(2) + 1.0e-6).flatten(1).mean(1)
    tail_count = max(1, int(math.ceil(per_sample.numel() * config.cvar_fraction)))
    cvar = per_sample.topk(tail_count).values.mean()
    total = (
        main
        + config.lambda_direct * direct
        + config.lambda_physical * physical
        + config.lambda_redegrade * redegrade
        + config.lambda_depth_smooth * smooth
        + config.lambda_depth_supervision * depth_supervision
        + config.lambda_airlight_supervision * airlight_supervision
        + config.lambda_cvar * cvar
    )
    return {
        "total": total,
        "main": main,
        "direct": direct,
        "physical": physical,
        "redegrade": redegrade,
        "depth_smooth": smooth,
        "depth_supervision": depth_supervision,
        "airlight_supervision": airlight_supervision,
        "cvar": cvar,
    }

# Generated: 2026-10-01 09:00 +08:00
