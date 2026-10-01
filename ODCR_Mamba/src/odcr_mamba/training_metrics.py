# Training-validation metrics only; not a universal evaluator for the released paper tables.
from __future__ import annotations

import torch
import torch.nn.functional as F


def batch_psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (prediction - target).pow(2).flatten(1).mean(1).clamp_min(1.0e-12)
    return -10.0 * torch.log10(mse)


def batch_ssim(prediction: torch.Tensor, target: torch.Tensor, window: int = 11) -> torch.Tensor:
    padding = window // 2
    mu_x = F.avg_pool2d(prediction, window, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, window, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(prediction * prediction, window, 1, padding) - mu_x.pow(2)
    sigma_y = F.avg_pool2d(target * target, window, 1, padding) - mu_y.pow(2)
    sigma_xy = F.avg_pool2d(prediction * target, window, 1, padding) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
    )
    return score.flatten(1).mean(1)


# Generated: 2026-10-01 09:00 +08:00
