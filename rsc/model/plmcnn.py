#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import pytorch_lightning as pl

from .data_augmentation import DataAugmentation

from .mcnn import MaskCNN
from .mcnn_loss import MCNNLoss


class PLMaskCNN(pl.LightningModule):

    def __init__(self,
                 labels,
                 weights,
                 learning_rate: tuple | float = 1e-4,
                 loss_lambda=0.1,
                 staging_order=(0, )):
        super().__init__()

        # Hyperparameters
        if isinstance(learning_rate, float):
            learning_rate = tuple([learning_rate] * len(staging_order))
        else:
            assert len(learning_rate) == len(staging_order)
        self.learning_rate = tuple(learning_rate)
        self.loss_lambda = loss_lambda
        self.staging_order = staging_order
        self.labels = labels
        self.weights = weights
        self.save_hyperparameters()

        # Stateful learning rate
        self._lr = learning_rate[0]

        self.transform = DataAugmentation()
        self.loss = MCNNLoss(self.weights, self.loss_lambda)
        self.model = MaskCNN(num_classes=len(self.labels))

    def set_stage(self, v, lr):
        first_stage = (self.model.encoder, self.model.decoder)
        second_stage = (self.model.encoder2, self.model.avgpool, self.model.fc)

        # Freeze / unfreeze components based on stage
        if v == 0:
            [e.unfreeze() for e in first_stage]
            [e.unfreeze() for e in second_stage]
        elif v == 1:
            [e.unfreeze() for e in first_stage]
            [e.freeze() for e in second_stage]
        elif v == 2:
            [e.freeze() for e in first_stage]
            [e.unfreeze() for e in second_stage]
        else:
            raise ValueError(f'Unknown v: {repr(v):s}')

        # Loss function requires stage
        self.loss.stage = v

        # Learning rate depends on stage
        self._lr = lr

        # Set stage
        self.stage = v

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
        return torch.optim.Adam(self.parameters(), lr=self._lr)
