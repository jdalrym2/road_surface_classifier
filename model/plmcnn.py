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
        learning_rate=1e-4,
        loss_lambda=1e-1,
    ):
        super().__init__()

        # Hyperparameters
        self.learning_rate = learning_rate
        self.loss_lambda = loss_lambda

        self.save_hyperparameters()

        weights_df = pd.read_csv(
            '/data/road_surface_classifier/dataset/class_weights.csv')

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
        loss = self.loss_lambda * loss1 + loss2
        self.log_dict(
            {
                'train_loss_im': loss1,
                'train_loss_cl': loss2,
                'train_loss': loss,
            },
            on_step=True,
            on_epoch=True)
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
        loss = self.loss_lambda * loss1 + loss2

        self.log_dict(
            {
                'val_loss_im': loss1,
                'val_loss_cl': loss2,
                'val_loss': loss,
            },
            on_step=True,
            on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
