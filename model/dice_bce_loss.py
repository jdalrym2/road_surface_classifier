#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as functional


class DiceBCELoss(nn.Module):
    """ https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook#BCE-Dice-Loss """

    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.flatten()
        targets = targets.flatten()

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() +
                                                        targets.sum() + smooth)
        BCE = functional.binary_cross_entropy(inputs,
                                              targets,
                                              reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE