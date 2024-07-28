#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import optuna
import pytorch_lightning as pl

from .data_augmentation import DataAugmentation

from .mcnn import MaskCNN
from .mcnn_loss import MCNNLoss


class PLMaskCNN(pl.LightningModule):
    """ PyTorch Lightning wrapper for training the 
        RSC MaskCNN """

    def __init__(self,
                 labels,
                 top_level_map,
                 weights,
                 learning_rate: float = 1e-4,
                 seg_k: float = 1.0,
                 ob_k: float = 1.0,
                 nc: int = 4):
        super().__init__()

        # Hyperparameters
        self.learning_rate = learning_rate
        self.seg_k = seg_k
        self.ob_k = ob_k
        self.labels = labels
        self.top_level_map = top_level_map
        self.weights = weights
        self.save_hyperparameters()

        # Number of channels
        self.nc = nc

        # Optuna trial (use set_optuna_trial)
        self.trial: optuna.trial.Trial | None = None

        # Stateful min val_loss_cl
        self.min_val_loss = float('inf')
        self.min_val_loss_im = float('inf')
        self.min_val_loss_cl = float('inf')
        self.min_val_loss_ob = float('inf')

        # Stateful learning rate
        self._lr = learning_rate

        self.transform = DataAugmentation(has_nir=(nc == 4))
        self.loss = MCNNLoss(self.top_level_map, self.weights,
                             self.seg_k, self.ob_k)
        
        # Labels: add 1 for "obscuartion"
        # Channels: add 1 for "mask" (e.g. RGB + mask, RGB + NIR + mask)
        self.model = MaskCNN(num_classes=len(self.labels) + 1,
                            num_channels=nc + 1)

    def set_optuna_trial(self, trial: optuna.trial.Trial | None):
        self.trial = trial

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
                'train_loss_im': self.loss.seg_loss,
                'train_loss_cl': self.loss.cl_loss,
                'train_loss_ob': self.loss.ob_loss,
                'train_loss': loss,
            },
            on_step=False,
            on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, z = batch

        img_mask_c = slice(0, self.nc + 1)
        probmask_c = slice(self.nc + 1, self.nc + 2)

        # Create probmask, image + mask (be careful, order matters!)
        y = x[:, probmask_c, :, :]
        x = x[:, img_mask_c, :, :]

        y_hat, z_hat = self.forward(x)
        loss = self.loss(y_hat, y, z_hat, z)
        self.log_dict(
            {
                'val_loss_im': self.loss.seg_loss,
                'val_loss_cl': self.loss.cl_loss,
                'val_loss_ob': self.loss.ob_loss,
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
        this_val_loss_ob = float(metrics['val_loss_ob'])

        if this_val_loss < self.min_val_loss:
            self.min_val_loss = this_val_loss
            self.min_val_loss_im = this_val_loss_im
            self.min_val_loss_cl = this_val_loss_cl
            self.min_val_loss_ob = this_val_loss_ob

        self.log_dict({
            'min_val_loss_im': self.min_val_loss_im,
            'min_val_loss_cl': self.min_val_loss_cl,
            'min_val_loss_ob': self.min_val_loss_ob,
            'min_val_loss': self.min_val_loss,
        })

        if self.trial is not None:
            self.trial.report(this_val_loss_cl, self.current_epoch)
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._lr)
