from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .blocks import LocalBlock, ODCBlock


class ODStage(nn.Module):
    """A stack of optical-depth-conditioned state-space blocks."""

    def __init__(
        self,
        channels: int,
        blocks: int,
        state_dim: int,
        conditioned: bool,
        alpha_init: float,
        delta_min: float,
        delta_max: float,
    ) -> None:
        super().__init__()
        if blocks < 1:
            raise ValueError("blocks must be positive")
        arguments = dict(
            state_dim=state_dim,
            conditioned=conditioned,
            alpha_init=alpha_init,
            delta_min=delta_min,
            delta_max=delta_max,
            depth_modulation=True,
            adaptive_gate=True,
        )
        self.blocks = nn.ModuleList([ODCBlock(channels, **arguments) for _ in range(blocks)])

    def forward(self, x: torch.Tensor, optical_depth: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, optical_depth)
        return x

class StandaloneODCRMamba(nn.Module):
    """Single-input ODCR-Mamba network for end-to-end image dehazing.

    The network predicts optical depth and atmospheric light from the hazy
    image, injects the estimated depth into four-route state-space blocks,
    and fuses direct and atmospheric-scattering reconstruction branches.
    It does not require an externally restored image at training or inference.
    """

    def __init__(
        self,
        channels: Sequence[int] = (24, 48, 96, 192),
        encoder_blocks: Sequence[int] = (1, 2, 4, 2),
        decoder_blocks: Sequence[int] = (1, 2, 1),
        state_dim: int = 8,
        conditioned: bool = True,
        alpha_init: float = 0.5,
        delta_min: float = 1.0e-4,
        delta_max: float = 8.0e-2,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if len(channels) != 4 or len(encoder_blocks) != 4 or len(decoder_blocks) != 3:
            raise ValueError("Expected four encoder scales and three decoder scales")
        if any(channels[index + 1] != 2 * channels[index] for index in range(3)):
            raise ValueError("Each scale must double the channel count")

        self.channels = tuple(int(value) for value in channels)
        self.conditioned = conditioned
        self.residual_scale = float(residual_scale)
        stage_args = dict(
            state_dim=state_dim,
            conditioned=conditioned,
            alpha_init=alpha_init,
            delta_min=delta_min,
            delta_max=delta_max,
        )

        c0, c1, c2, c3 = self.channels
        self.stem = nn.Conv2d(3, c0, 3, padding=1)
        self.local_stem = nn.Sequential(LocalBlock(c0), LocalBlock(c0))

        self.learned_depth_head = nn.Sequential(
            LocalBlock(c0), nn.Conv2d(c0, c0 // 2, 3, padding=1), nn.SiLU(), nn.Conv2d(c0 // 2, 1, 3, padding=1)
        )
        self.airlight_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(c0, c0, 1), nn.SiLU(), nn.Conv2d(c0, 3, 1)
        )
        self.depth_mix_logit = nn.Parameter(torch.tensor(0.0))

        self.encoder = nn.ModuleList(
            [
                ODStage(c0, encoder_blocks[0], **stage_args),
                ODStage(c1, encoder_blocks[1], **stage_args),
                ODStage(c2, encoder_blocks[2], **stage_args),
                ODStage(c3, encoder_blocks[3], **stage_args),
            ]
        )
        self.down = nn.ModuleList(
            [
                nn.Conv2d(c0, c1, 4, stride=2, padding=1),
                nn.Conv2d(c1, c2, 4, stride=2, padding=1),
                nn.Conv2d(c2, c3, 4, stride=2, padding=1),
            ]
        )

        self.up_projection = nn.ModuleList(
            [nn.Conv2d(c3, c2, 3, padding=1), nn.Conv2d(c2, c1, 3, padding=1), nn.Conv2d(c1, c0, 3, padding=1)]
        )
        self.skip_fusion = nn.ModuleList(
            [nn.Conv2d(c2 * 2, c2, 1), nn.Conv2d(c1 * 2, c1, 1), nn.Conv2d(c0 * 2, c0, 1)]
        )
        self.decoder = nn.ModuleList(
            [
                ODStage(c2, decoder_blocks[0], **stage_args),
                ODStage(c1, decoder_blocks[1], **stage_args),
                ODStage(c0, decoder_blocks[2], **stage_args),
            ]
        )

        self.direct_head = nn.Sequential(LocalBlock(c0), nn.Conv2d(c0, 3, 3, padding=1))
        self.quality_head = nn.Sequential(LocalBlock(c0), nn.Conv2d(c0, 1, 3, padding=1))

    @staticmethod
    def prior_optical_depth(hazy: torch.Tensor, airlight: torch.Tensor) -> torch.Tensor:
        normalized = hazy / airlight.clamp_min(0.3)
        dark_channel = normalized.min(dim=1, keepdim=True).values
        dark_channel = -F.max_pool2d(-dark_channel, kernel_size=15, stride=1, padding=7)
        transmission = (1.0 - 0.90 * dark_channel).clamp(0.05, 1.0)
        return -torch.log(transmission)

    def estimate_atmosphere(self, hazy: torch.Tensor, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        airlight = torch.sigmoid(self.airlight_head(features))
        learned_depth = 4.0 * torch.sigmoid(self.learned_depth_head(features))
        prior_depth = self.prior_optical_depth(hazy, airlight)
        mixing_weight = torch.sigmoid(self.depth_mix_logit)
        optical_depth = (mixing_weight * learned_depth + (1.0 - mixing_weight) * prior_depth).clamp(0.0, 4.0)
        return {
            "airlight": airlight,
            "learned_depth": learned_depth,
            "prior_depth": prior_depth,
            "depth_mix": mixing_weight,
            "depth": optical_depth,
        }

    def forward(self, hazy: torch.Tensor) -> Dict[str, torch.Tensor]:
        shallow = self.local_stem(self.stem(hazy))
        atmosphere = self.estimate_atmosphere(hazy, shallow)
        depth = atmosphere["depth"]

        skips = []
        x = shallow
        for index, stage in enumerate(self.encoder):
            x = stage(x, depth)
            skips.append(x)
            if index < len(self.down):
                x = self.down[index](x)

        for index, stage in enumerate(self.decoder):
            skip = skips[-2 - index]
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = self.up_projection[index](x)
            x = self.skip_fusion[index](torch.cat([x, skip], dim=1))
            x = stage(x, depth)

        direct = (hazy + self.residual_scale * torch.tanh(self.direct_head(x))).clamp(0.0, 1.0)
        transmission = torch.exp(-depth).clamp(0.05, 1.0)
        physical = ((hazy - atmosphere["airlight"]) / transmission + atmosphere["airlight"]).clamp(0.0, 1.0)
        quality = torch.sigmoid(self.quality_head(x))
        output = (quality * direct + (1.0 - quality) * physical).clamp(0.0, 1.0)
        return {
            "output": output,
            "direct": direct,
            "physical": physical,
            "quality": quality,
            "transmission": transmission,
            **atmosphere,
        }

# Generated: 2026-10-01 09:00 +08:00
