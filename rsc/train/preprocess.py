#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import kornia


class PreProcess(nn.Module):

    def __init__(self):
        super().__init__()
        self.resize = kornia.augmentation.Resize(224, keepdim=True)

    @torch.no_grad()
    def forward(self, x) -> torch.Tensor:

        # NOTE: applies to full 6-channel input (image + mask + probmask)

        # Convert to tensor
        x = kornia.utils.image_to_tensor(x, keepdim=True).float()
        x = self.resize(x)

        # Normalize between 0 and 1
        x = torch.divide(x, 255.)

        return x