#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy

import torch

from .plmcnn import PLMaskCNN

# Get PyTorch device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PLMaskCNNCoTeach(PLMaskCNN):
    """ PyTorch Lightning MaskCNN implementation w/ coteaching"""

    def __init__(self, *args, rt=0.8, **kwargs):
        super().__init__(*args, **kwargs)

        # Enable manual optimization
        self.automatic_optimization = False

        # Define co-teaching model as copy
        self.model_ct = deepcopy(self.model)

        # Hyperparameters
        self.rt = rt
        self.save_hyperparameters()

    def set_stage(self, v, lr):
        self.set_stage_for_model(self.model, v)
        self.set_stage_for_model(self.model_ct, v)

        # Loss function requires stage
        self.loss.stage = v

        # Learning rate depends on stage
        self._lr = lr

        # Set stage
        self.stage = v

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, z = self.training_prep_batch(batch)

        # Inference each model
        y_hat_1, z_hat_1 = self.model.forward(x)
        y_hat_2, z_hat_2 = self.model_ct.forward(x)

        # Compute loss for each model
        loss_1 = self.loss(y_hat_1, y, z_hat_1, z, reduce=False)
        loss_2 = self.loss(y_hat_2, y, z_hat_2, z, reduce=False)

        # Co-teach (note this also does reduction)
        loss_1, loss_2 = self.coteach_loss(loss_1, loss_2, self.rt)

        # Optimize
        opt_1, opt_2 = self.optimizers()
        opt_1.zero_grad()
        self.manual_backward(loss_1)
        opt_1.step()
        opt_2.zero_grad()
        self.manual_backward(loss_2)
        opt_2.step()

    @staticmethod
    def coteach_loss(loss_1, loss_2, rt):
        # https://github.com/yeachan-kr/pytorch-coteaching/blob/master/runs/train_coteaching.py

        # Get samples of small loss
        _, loss_1_sm_idx = torch.topk(loss_1,
                                      k=int(int(loss_1.size(0)) * rt),
                                      largest=False)
        _, loss_2_sm_idx = torch.topk(loss_2,
                                      k=int(int(loss_2.size(0)) * rt),
                                      largest=False)

        # Co-teaching loss adjustment routine
        loss_1_filter = torch.zeros((loss_1.size(0))).to(device)
        loss_1_filter[loss_2_sm_idx] = 1.0
        loss_1 = (loss_1_filter * loss_1).sum()

        loss_2_filter = torch.zeros((loss_2.size(0))).to(device)
        loss_2_filter[loss_1_sm_idx] = 1.0
        loss_2 = (loss_2_filter * loss_2).sum()

        return loss_1, loss_2

    def configure_optimizers(self):
        opt_1 = torch.optim.Adam(self.model.parameters(), lr=self._lr)
        opt_2 = torch.optim.Adam(self.model_ct.parameters(), lr=self._lr)
        return opt_1, opt_2
