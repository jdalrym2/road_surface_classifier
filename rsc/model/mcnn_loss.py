#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .dice_bce_loss import DiceBCELoss


def custom_sigmoid(d: torch.Tensor,
                   x_range: tuple[float, float] = (0, 1),
                   y_range: tuple[float, float] = (0, 1),
                   k: float = 0.1) -> torch.Tensor:
    """
    Create a custom sigmoid (logit: `1 / (1 + exp(x))`), in
    which the transition occurs approximately within `x_range`, between
    the values in `y_range`

    Args:
        d (torch.Tensor): Input data
        x_range (tuple[float, float], optional): X- domain. Defaults to (0, 1).
        y_range (tuple[float, float], optional): Y- domain. Defaults to (0, 1).
        k (float, optional): Slope. Smaller `k` -> bigger slope. Defaults to 0.1.

    Returns:
        torch.Tensor: Sigmoid output
    """
    # Parse input range
    x_min, x_max = x_range
    y_min, y_max = y_range

    # Normalize d to [0, 1]
    d = (d - x_min) / (x_max - x_min)

    # Now normalize to [-1, 1]
    d = 2 * (d - 0.5)

    # Compute sigmoid
    s = 1 / (1 + torch.exp(-d / k))

    # Normalize to y_range
    s = s * (y_max - y_min) + y_min

    return s


class MCNNLoss(nn.Module):

    def __init__(self, class_weights, loss_lambda):
        super().__init__()
        self.class_weights = torch.Tensor(class_weights).float().cuda()
        self.loss_lambda = loss_lambda
        self.dice_loss = DiceBCELoss()

        self.loss1 = 0.
        self.loss2 = 0.
        self.stage = 0

    def forward(self, y_hat, y, z_hat, z):

        if self.stage in (0, 1):
            self.loss1 = self.dice_loss(y_hat, y)
        else:
            self.loss1 = 0.

        if self.stage in (0, 2):
            # Compute weight based on "true" obscuaration
            # Fit a sigmoid from 1->0.1 between 0.9->1 in range
            # This will punish the model less for obscuration values
            # >= 0.9
            #obsc_w = custom_sigmoid(z[..., 2], (0.4, 0.9), (1, 5), 0.15)

            self.loss2 = functional.cross_entropy(
                z_hat, z, weight=self.class_weights,
                reduction='mean')     # reduction='none'
            #self.loss2 = torch.sum(obsc_w * self.loss2) / torch.sum(obsc_w)
        else:
            self.loss2 = 0.

        loss = self.loss_lambda * self.loss1 + self.loss2
        return loss