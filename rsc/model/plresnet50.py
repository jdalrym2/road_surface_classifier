#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

import torch
import torch.nn.functional as functional
import torchvision
import pytorch_lightning as pl

from data_augmentation import DataAugmentation


class PLResnet50(pl.LightningModule):

    def __init__(self):
        super().__init__()

        # Get Resnet50 w/ default weights
        self.rnet = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.DEFAULT)
        del self.rnet.fc

        # Freeze all layers
        # for param in rnet.parameters():
        #     param.requires_grad = False

        self.mask_layer = torch.nn.Sequential(
            torch.nn.Conv2d(1,
                            64,
                            kernel_size=(7, 7),
                            stride=(2, 2),
                            padding=(3, 3),
                            bias=False), torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64,
                            512,
                            kernel_size=(1, 1),
                            stride=(1, 1),
                            bias=False), torch.nn.BatchNorm2d(512),
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)))

        # Switch output layer to 2 classes
        self.avg_pool_1 = torch.nn.AvgPool1d(4, 4)
        self.linear_2 = torch.nn.Linear(in_features=1024,
                                        out_features=2,
                                        bias=True)

        weights_df = pd.read_csv(
            '/data/road_surface_classifier/dataset_simple/class_weights.csv')

        self.weights = torch.tensor(weights_df['weight']).float().cuda()
        self.labels = list(weights_df['class_name'])
        self.transform = DataAugmentation()

    def forward(self, x, xm):
        x = self.rnet.conv1(x)
        x = self.rnet.bn1(x)
        x = self.rnet.relu(x)
        x = self.rnet.maxpool(x)
        x = self.rnet.layer1(x)
        x = self.rnet.layer2(x)
        x = self.rnet.layer3(x)
        x = self.rnet.layer4(x)
        x = self.rnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.avg_pool_1(x)
        xm = self.mask_layer(xm)
        xm = torch.flatten(xm, 1)
        x = torch.concat((x, xm), axis=-1)     # type: ignore
        return self.linear_2(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, xm = self.transform(x)
        y_hat = self.forward(x, xm)
        loss = functional.cross_entropy(y_hat, y, weight=self.weights)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        xm = x[:, 3:, :, :]
        x = x[:, :3, :, :]
        y_hat = self.forward(x, xm)
        loss = functional.cross_entropy(y_hat, y, weight=self.weights)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0005)