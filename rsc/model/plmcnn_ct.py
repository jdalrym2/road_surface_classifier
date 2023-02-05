#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy

import torch

from .plmcnn import PLMaskCNN
from .mcnn import MaskCNN

# Get PyTorch device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PLMaskCNNCoTeach(PLMaskCNN):
    """ PyTorch Lightning MaskCNN implementation w/ coteaching"""

    def __init__(self, *args, rt=0.2, ne=15, **kwargs):
        super().__init__(*args, **kwargs)

        # Enable manual optimization
        self.automatic_optimization = False

        # Define co-teaching model as copy
        self.model_ct = MaskCNN(num_classes=len(self.labels))
        for m in (self.model, self.model_ct):
            m.encoder2.reset_parameters()

        # Hyperparameters
        self.rt = rt
        self.ne = ne
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

        # For computing accuracy
        z_true = torch.argmax(z[:, (0, 1)], 1)

        # Inference each model
        y_hat_1, z_hat_1 = self.model.forward(x)
        y_hat_2, z_hat_2 = self.model_ct.forward(x)

        # Compute loss for each model
        loss_1 = self.loss(y_hat_1, y, z_hat_1, z, reduce=False)
        loss_2 = self.loss(y_hat_2, y, z_hat_2, z, reduce=False)

        acc_1 = (z_true == torch.argmax(z_hat_1[:,
                                                (0, 1)], 1)).sum() / z.shape[0]
        acc_2 = (z_true == torch.argmax(z_hat_2[:,
                                                (0, 1)], 1)).sum() / z.shape[0]

        self.log_dict(
            {
                'loss_1': loss_1.mean(),
                'loss_2': loss_2.mean(),
                'acc_1': acc_1,
                'acc_2': acc_2
            },
            on_step=True,
            on_epoch=True)
        rt = 1 - self.rt * min(1, self.current_epoch / self.ne)

        # Co-teach (note this also does reduction)
        loss_1, loss_2 = self.coteach_loss(loss_1, loss_2, rt)

        self.log_dict(
            {
                'loss_1_af': loss_1,
                'loss_2_at': loss_2,
                'loss_diff': abs(loss_1 - loss_2),
            },
            on_step=True,
            on_epoch=True)

        self.log_dict({'rt': rt}, on_epoch=True)

        # Optimize
        opt_1, opt_2 = self.optimizers()     # type: ignore
        opt_1.zero_grad()     # type: ignore
        self.manual_backward(loss_1)
        opt_1.step()
        opt_2.zero_grad()     # type: ignore
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
        loss_1 = (loss_1_filter * loss_1).sum() / loss_1_filter.sum()

        loss_2_filter = torch.zeros((loss_2.size(0))).to(device)
        loss_2_filter[loss_1_sm_idx] = 1.0
        loss_2 = (loss_2_filter * loss_2).sum() / loss_2_filter.sum()

        return loss_1, loss_2

    def configure_optimizers(self):
        opt_1 = torch.optim.Adam(self.model.parameters(), lr=self._lr)
        opt_2 = torch.optim.Adam(self.model_ct.parameters(), lr=self._lr)
        return opt_1, opt_2
