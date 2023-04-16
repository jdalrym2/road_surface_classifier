#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import optuna
import pytorch_lightning as pl

from .data_augmentation import DataAugmentation

from .mcnn import MaskCNN
from .mcnn_loss import MCNNLoss


class PLMaskCNN(pl.LightningModule):

    def __init__(self,
                 trial: optuna.trial.Trial | None,
                 labels,
                 top_level_map,
                 weights,
                 learning_rate: float = 1e-4,
                 loss_lambda: float = 0.1):
        super().__init__()

        # Hyperparameters
        self.learning_rate = learning_rate
        self.loss_lambda = loss_lambda
        self.labels = labels
        self.top_level_map = top_level_map
        self.weights = weights
        self.save_hyperparameters()

        # Optuna trial
        self.trial = trial

        # Stateful min val_loss_cl
        self.min_val_loss = float('inf')
        self.min_val_loss_im = float('inf')
        self.min_val_loss_cl = float('inf')

        # Stateful learning rate
        self._lr = learning_rate

        self.transform = DataAugmentation()
        self.loss = MCNNLoss(self.top_level_map, self.weights, self.loss_lambda)
        self.model = MaskCNN(num_classes=len(self.labels) + 2)  # for obscuration

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
            on_step=False,
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
            on_step=False,
            on_epoch=True)
        return loss

    def on_validation_epoch_end(self):

        metrics = self.trainer.logged_metrics
        this_val_loss = float(metrics['val_loss'])
        this_val_loss_im = float(metrics['val_loss_im'])
        this_val_loss_cl = float(metrics['val_loss_cl'])

        self.min_val_loss = min(self.min_val_loss, this_val_loss)
        self.min_val_loss_im = min(self.min_val_loss_im, this_val_loss_im)
        self.min_val_loss_cl = min(self.min_val_loss_cl, this_val_loss_cl)

        self.log_dict({
            'min_val_loss_im': self.min_val_loss_im,
            'min_val_loss_cl': self.min_val_loss_cl,
            'min_val_loss': self.min_val_loss,
        })

        if self.trial is not None:
            self.trial.report(this_val_loss_cl, self.current_epoch)
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._lr)
