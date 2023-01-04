#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import pytorch_lightning as pl

from data_augmentation import DataAugmentation

from mcnn import MaskCNN
from mcnn_loss import MCNNLoss


class PLMaskCNN(pl.LightningModule):

    def __init__(self,
                 labels,
                 weights,
                 learning_rate=1e-4,
                 loss_lambda=0.1,
                 frozen=None):
        super().__init__()

        # Hyperparameters
        self.learning_rate = learning_rate
        self.loss_lambda = loss_lambda
        self.frozen = frozen
        self.save_hyperparameters()

        # Set labels and weights for training
        self.labels = labels
        self.weights = torch.tensor(weights).float().cuda()

        self.transform = DataAugmentation()
        self.loss = MCNNLoss(self.loss_lambda, self.weights)
        self.model = MaskCNN(num_classes=len(self.labels))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, z = batch
        x, xpm = self.transform(x)
        y_hat, z_hat = self.forward(x)
        loss = self.loss(y_hat, xpm, z_hat, z)
        self.log_dict(
            {
                'train_loss_im': self.loss.loss1,
                'train_loss_cl': self.loss.loss2,
                'train_loss': loss,
            },
            on_step=True,
            on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, z = batch

        # Create image + mask, probmask (be careful, order matters!)
        y = x[:, 5:6, :, :]
        x = x[:, 0:5, :, :]

        y_hat, z_hat = self.forward(x)
        loss = self.loss(y_hat, y, z_hat, z)
        self.log_dict(
            {
                'val_loss_im': self.loss.loss1,
                'val_loss_cl': self.loss.loss2,
                'val_loss': loss,
            },
            on_step=True,
            on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
