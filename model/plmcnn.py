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
        weights_path='/data/road_surface_classifier/rsc_naip_L17c/dataset_simple/class_weights.csv',
        highways_path='/data/road_surface_classifier/rsc_naip_L17c/dataset_simple/highway_weights.csv'
    ):
        super().__init__()

        weights_df = pd.read_csv(weights_path)
        highways_df = pd.read_csv(highways_path)

        self.weights = torch.tensor(weights_df['weight']).float().cuda()
        self.labels = list(weights_df['class_name'])
        self.labels = list(weights_df['class_name'])
        self.highways = list(highways_df['highway_name'])
        self.transform = DataAugmentation()

        self.loss = DiceBCELoss()

        self.model = MaskCNN(num_classes=len(self.labels),
                             num_highways=len(self.highways))

    def forward(self, x, xm, hwy):
        return self.model(torch.concat((x, xm), dim=1), hwy)

    def training_step(self, batch, batch_idx):
        x, z, h = batch
        x, xm = self.transform(x)
        y_hat, z_hat = self.forward(x, xm, h)
        loss1 = self.loss(y_hat, xm)
        loss2 = functional.cross_entropy(z_hat, z, weight=self.weights)
        loss = 1e-1 * loss1 + loss2
        self.log('train_loss1', loss1, on_step=False, on_epoch=True)
        self.log('train_loss2', loss2, on_step=False, on_epoch=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, z, h = batch
        xm = x[:, 3:, :, :]
        x = x[:, :3, :, :]
        y_hat, z_hat = self.forward(x, xm, h)
        loss1 = self.loss(y_hat, xm)
        loss2 = functional.cross_entropy(z_hat, z, weight=self.weights)
        loss = 1e-1 * loss1 + loss2
        self.log('val_loss1', loss1)
        self.log('val_loss2', loss2)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        lr = 5e-6     # Orig: 1e-4
        return torch.optim.Adam(self.parameters(), lr=lr)
