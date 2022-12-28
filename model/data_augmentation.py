#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import kornia


class DataAugmentation(nn.Module):

    def __init__(self):
        super().__init__()

        self.transform_pos = nn.Sequential(
            kornia.augmentation.RandomHorizontalFlip(p=0.5),
            kornia.augmentation.RandomVerticalFlip(p=0.5))
        self.transform_color = nn.Sequential(
            kornia.augmentation.RandomPlasmaBrightness(roughness=(0.1, 0.5),
                                                       intensity=(0.1, 0.3)),
            kornia.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1))

    @torch.no_grad()
    def forward(
            self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_aug = self.transform_pos(x)
        im_aug = self.transform_color(x_aug[:, 0:3, :, :])
        m_aug = x_aug[:, 3:4, :, :]
        pm_aug = x_aug[:, 4:5, :, :]
        return im_aug, m_aug, pm_aug


class PreProcess(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, x) -> torch.Tensor:
        x = kornia.utils.image_to_tensor(x, keepdim=True).float()
        x = torch.divide(x, 255.)
        return x
