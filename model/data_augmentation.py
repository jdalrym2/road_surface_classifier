#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import kornia

from color_jitter_nohuesat import ColorJitterNoHueSat


class DataAugmentation(nn.Module):

    def __init__(self):
        super().__init__()

        # Random positional transformations
        self.transform_pos = nn.Sequential(
            kornia.augmentation.RandomHorizontalFlip(p=0.5),
            kornia.augmentation.RandomVerticalFlip(p=0.5))

        # Transform RGB
        self.transform_color = nn.Sequential(
            kornia.augmentation.RandomPlasmaBrightness(roughness=(0.1, 0.5),
                                                       intensity=(0.1, 0.3)),
            kornia.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1))

        # Transform NIR
        self.transform_nir = nn.Sequential(ColorJitterNoHueSat(0.1, 0.1))

    @torch.no_grad()
    def forward(
            self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Apply position transform to image + masks
        x_aug = self.transform_pos(x)

        # RGB + NIR color transformations
        im_rgb_aug = self.transform_color(x_aug[:, 0:3, :, :])
        im_nir_aug = self.transform_nir(x_aug[:, 3:4, :, :])

        # Combine into 4-channel RGB+NIR image
        im_aug = torch.concat((im_rgb_aug, im_nir_aug), dim=1)

        # Extract mask and probmask
        m_aug = x_aug[:, 4:5, :, :]
        pm_aug = x_aug[:, 5:6, :, :]

        return im_aug, m_aug, pm_aug


class PreProcess(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, x) -> torch.Tensor:

        # NOTE: applies to full 6-channel input (image + mask + probmask)

        # Convert to tensor
        x = kornia.utils.image_to_tensor(x, keepdim=True).float()

        # Normalize between 0 and 1
        x = torch.divide(x, 255.)

        return x
