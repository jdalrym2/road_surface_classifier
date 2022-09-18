#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

import torch
import torch.nn.functional as functional
import torchvision
import pytorch_lightning as pl

from data_augmentation import DataAugmentation

from mcnn import MaskCNN
from dice_bce_loss import DiceBCELoss


class PLMaskCNN(pl.LightningModule):

    def __init__(self):
        super().__init__()

        # Get Resnet50 w/ default weights
        self.model = MaskCNN()

        weights_df = pd.read_csv(
            '/data/road_surface_classifier/dataset_simple/class_weights.csv')

        self.weights = torch.tensor(weights_df['weight']).float().cuda()
        self.labels = list(weights_df['class_name'])
        self.transform = DataAugmentation()

        self.loss = DiceBCELoss()

    def forward(self, x, xm):
        return self.model(torch.concat((x, xm), dim=1))

    def training_step(self, batch, batch_idx):
        x, z = batch
        x, xm = self.transform(x)
        y_hat, z_hat = self.forward(x, xm)
        loss1 = self.loss(y_hat, xm)
        loss2 = functional.cross_entropy(z_hat, z, weight=self.weights)
        loss = 1e-4 * loss1 + loss2
        self.log('train_loss1', loss1, on_step=False, on_epoch=True)
        self.log('train_loss2', loss2, on_step=False, on_epoch=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, z = batch
        xm = x[:, 3:, :, :]
        x = x[:, :3, :, :]
        y_hat, z_hat = self.forward(x, xm)
        loss1 = self.loss(y_hat, xm)
        loss2 = functional.cross_entropy(z_hat, z, weight=self.weights)
        loss = 1e-4 * loss1 + loss2
        self.log('val_loss1', loss1)
        self.log('val_loss2', loss2)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-6)
