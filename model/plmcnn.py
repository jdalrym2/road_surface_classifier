#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

import torch
import torch.nn.functional as functional
import pytorch_lightning as pl

from data_augmentation import DataAugmentation

from mcnn import MaskCNN
from dice_bce_loss import DiceBCELoss


class PLMaskCNN(pl.LightningModule):

    def __init__(
        self,
        weights_path='/data/road_surface_classifier/dataset/class_weights.csv'
    ):
        super().__init__()

        weights_df = pd.read_csv(weights_path)

        self.weights = torch.tensor(weights_df['weight']).float().cuda()
        self.labels = list(weights_df['class_name'])
        self.transform = DataAugmentation()

        self.loss = DiceBCELoss()

        self.model = MaskCNN(num_classes=len(self.labels))

    def forward(self, x, xm):
        return self.model(torch.concat((x, xm), dim=1))

    def training_step(self, batch, batch_idx):
        x, z = batch
        x, xm, xpm = self.transform(x)
        y_hat, z_hat = self.forward(x, xm)
        loss1 = self.loss(y_hat, xpm)
        loss2 = functional.cross_entropy(z_hat, z, weight=self.weights)
        loss = 1e-1 * loss1 + loss2
        self.log('train_loss1', loss1, on_step=False, on_epoch=True)
        self.log('train_loss2', loss2, on_step=False, on_epoch=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, z = batch

        # Create image, mask, probmask (be careful, order matters!)
        xm = x[:, 3:4, :, :]
        xpm = x[:, 4:5, :, :]
        x = x[:, 0:3, :, :]

        y_hat, z_hat = self.forward(x, xm)
        loss1 = self.loss(y_hat, xpm)
        loss2 = functional.cross_entropy(z_hat, z, weight=self.weights)
        loss = 1e-1 * loss1 + loss2
        self.log('val_loss1', loss1)
        self.log('val_loss2', loss2)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        lr = 1e-4
        return torch.optim.Adam(self.parameters(), lr=lr)
