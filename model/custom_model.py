#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%

import torch
import torchvision

# Get Resnet50 w/ default weights
rnet = torchvision.models.resnet50(
    weights=torchvision.models.ResNet50_Weights.DEFAULT)
rnet.fc = torch.nn.Identity()

position_layer = torch.nn.Sequential(
    torch.nn.Conv2d(1,
                    64,
                    kernel_size=(7, 7),
                    stride=(2, 2),
                    padding=(3, 3),
                    bias=False), torch.nn.BatchNorm2d(64),
    torch.nn.Conv2d(64, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
    torch.nn.BatchNorm2d(512))

# Switch output layer to 2 classes
linear_1 = torch.nn.Linear(in_features=2048, out_features=512, bias=True)
linear_2 = torch.nn.Linear(in_features=512, out_features=2, bias=True)

# %%
