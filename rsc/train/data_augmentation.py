#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple
import torch
import torch.nn as nn
import kornia

from .color_jitter_nohuesat import ColorJitterNoHueSat


class DataAugmentation(nn.Module):

    def __init__(self, has_nir: bool=True):
        super().__init__()

        # Are we augmenting NIR as well?
        self.has_nir = has_nir

        # Random positional transformations
        self.transform_flip = nn.Sequential(
            kornia.augmentation.RandomHorizontalFlip(p=0.5),
            kornia.augmentation.RandomVerticalFlip(p=0.5))

        # Random offset of mask to re-inforce proper segmentation
        # when labels may be inaccurate
        self.transform_offset = nn.Sequential(
            kornia.augmentation.RandomAffine(degrees=(-15, 15),
                                             translate=(0.0625, 0.0625),
                                             p=0.5))

        # Transform RGB
        self.transform_color = nn.Sequential(
            kornia.augmentation.RandomPlasmaBrightness(roughness=(0.1, 0.5),
                                                       intensity=(0.1, 0.3)),
            kornia.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1))

        # Transform NIR
        self.transform_nir = nn.Sequential(ColorJitterNoHueSat(0.1, 0.1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # Apply position transform to image + masks
        x_aug = self.transform_flip(x)

        # Break down what all of the channels are
        mask_c = slice(4, 5) if self.has_nir else slice(3, 4)
        probmask_c = slice(5, 6) if self.has_nir else slice(4, 5)

        # RGB color transformation
        im_aug = self.transform_color(x_aug[:, 0:3, ...])

        if self.has_nir:
            # NIR color transformation
            im_nir_aug = self.transform_nir(x_aug[:, 3:4, ...])

            # Combine NIR with RGB image
            im_aug = torch.concat((im_aug, im_nir_aug), dim=1)

        # Apply offset transform to mask
        mask_aug = self.transform_offset(x_aug[:, mask_c, ...])

        # Combine into color image + location mask
        im_aug = torch.concat((im_aug, mask_aug), dim=1)

        # Extract probmask for training
        pm_aug = x_aug[:, probmask_c, :, :]

        return im_aug, pm_aug
