#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as functional

from dice_bce_loss import DiceBCELoss


class MCNNLoss(nn.Module):

    __slots__ = [
        'loss_lambda', 'class_weights', 'dice_loss', 'loss1', 'loss2', 'stage'
    ]

    def __init__(self, loss_lambda, class_weights):
        super().__init__()
        self.loss_lambda = loss_lambda
        self.class_weights = class_weights
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
            self.loss2 = functional.cross_entropy(z_hat,
                                                  z,
                                                  weight=self.class_weights)
        else:
            self.loss2 = 0.

        loss = self.loss_lambda * self.loss1 + self.loss2
        return loss