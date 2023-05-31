#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .rsc_hxe_loss import RSCHXELoss


class MCNNLoss(nn.Module):
    """ Combined MaskCNN loss function """

    def __init__(self, top_lv_map, class_weights, seg_k, ob_k):
        super().__init__()

        # Inputs
        self.top_lv_map = torch.IntTensor(top_lv_map).cuda()
        self.class_weights = torch.Tensor(class_weights).float().cuda()
        self.seg_k = seg_k
        self.ob_k = ob_k

        # Loss functions
        self.hxe_loss = RSCHXELoss(self.top_lv_map, self.class_weights)
        self.o_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

        self.seg_loss = 0.
        self.cl_loss = 0.
        self.ob_loss = 0.
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

        # Loss 1: Dice BCE loss for segmentation (seg_loss)
        if self.stage in (0, 1):
            intersection = (y_hat * y).sum()
            smooth = 1  # Dice BCE smoothing parameter, hardcoded to 1 does just fine
            dice_loss = 1 - (2. * intersection + 1) / (
                y_hat.sum() + y.sum() + smooth)
            binary_ce = functional.binary_cross_entropy(y_hat,
                                                        y,
                                                        reduction='mean')
            self.seg_loss = binary_ce + dice_loss
        else:
            self.seg_loss = 0.

        # Loss 2: BCE Loss for estimating road obscuration (ob_loss)
        # This operates on the last logit produced by the model
        if self.stage in (0, 2):
            self.ob_loss = self.o_loss(z_hat[:, -1], z[:, -1])
        else:
            self.ob_loss = 0.

        # Loss 3: Cross entropy for classification result (cl_loss)
        # This operates on all but the last model logit
        if self.stage in (0, 2):
            self.cl_loss = self.hxe_loss(z_hat[:, :-1], z[:, :-1])
        else:
            self.cl_loss = 0.

        # Combine and return the combined loss
        loss = self.seg_k * self.seg_loss + self.ob_k * self.ob_loss + self.cl_loss
        return loss