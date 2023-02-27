#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as functional


class MCNNLoss(nn.Module):

    def __init__(self, class_weights, loss_lambda):
        super().__init__()
        self.class_weights = torch.Tensor(class_weights).float().cuda()
        self.loss_lambda = loss_lambda
        self.smooth = 1     # Dice-BCE loss smoothing parameter, will just hardcode to 1

        self.loss1 = 0.
        self.loss2 = 0.
        self.stage = 0

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor,
                z_hat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss given the model's segmentation and classification results.

        Segmentation loss is DICE + BCE loss, inspired from:
        https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook#BCE-Dice-Loss

        Args:
            y_hat (torch.Tensor): Segmentation model result
            y (torch.Tensor): Segmentation truth
            z_hat (torch.Tensor): Classification model result
            z (torch.Tensor): Classification truth

        Returns:
            torch.Tensor: Computed loss for the model
        """

        # Loss 1: Dice BCE loss for segmentation
        if self.stage in (0, 1):
            intersection = (y_hat * y).sum()
            dice_loss = 1 - (2. * intersection + self.smooth) / (
                y_hat.sum() + y.sum() + self.smooth)
            binary_ce = functional.binary_cross_entropy(y_hat,
                                                        y,
                                                        reduction='mean')
            self.loss1 = binary_ce + dice_loss
        else:
            self.loss1 = 0.

        # Loss 2: Cross entropy for classification result
        if self.stage in (0, 2):
            self.loss2 = functional.cross_entropy(
                z_hat,
                z,
                weight=self.class_weights,
            )

        else:
            self.loss2 = 0.

        # Combine and return the combined loss
        loss = self.loss_lambda * self.loss1 + self.loss2
        return loss