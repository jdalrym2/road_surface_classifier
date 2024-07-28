#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torchvision

# NOTE: Patch for MaxUnpool2d needs to be applied
# when exporting to ONNX
from torch.nn import MaxUnpool2d
#from patch import MaxUnpool2d


class Freezable:

    def parameters(self):
        # This is overridden
        return []

    def freeze(self):
        self._set_freeze(True)

    def unfreeze(self):
        self._set_freeze(False)

    def _set_freeze(self, v):
        req_grad = not bool(v)
        for param in self.parameters():
            param.requires_grad = req_grad


class FreezableModule(nn.Module, Freezable):
    pass


class FreezableModuleList(nn.ModuleList, Freezable):
    pass


class FreezableAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, Freezable):
    pass


class FreezableLinear(nn.Linear, Freezable):
    pass


class DecoderBlock(FreezableModule):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size)

    def forward(self, x, encoder_feats=None):

        x = self.upconv(x)

        enc_ftrs = self.crop(encoder_feats, x)
        x = torch.concat((x, enc_ftrs), dim=1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, h, w = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([h, w])(enc_ftrs)
        return enc_ftrs


class Resnet18Encoder(FreezableModule):

    def __init__(self, in_channels=3):
        super().__init__()

        # Get Resnet18 w/ default weights
        self.rnet = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT)

        if in_channels != 3:
            self.rnet.conv1 = nn.Conv2d(in_channels,
                                        64,
                                        kernel_size=(7, 7),
                                        stride=(2, 2),
                                        padding=(3, 3),
                                        bias=False)

        self.rnet.maxpool.return_indices = True

        # Delete final avg pool / linear layer
        del self.rnet.avgpool
        del self.rnet.fc

        # Features for cross-connections to decoder
        self.feats = [
            torch.Tensor([]),
            torch.Tensor([]),
            torch.Tensor([]),
            torch.Tensor([])
        ]
        self.maxpool_idxs = None

    def forward(self, x):
        # Building up to stuff
        x = self.rnet.conv1(x)
        x = self.rnet.bn1(x)
        x = self.rnet.relu(x)
        x, self.maxpool_idxs = self.rnet.maxpool(x)

        # Now the main layers
        self.feats[0] = x
        x = self.rnet.layer1(x)

        self.feats[1] = x
        x = self.rnet.layer2(x)

        self.feats[2] = x
        x = self.rnet.layer3(x)

        self.feats[3] = x
        x = self.rnet.layer4(x)

        return x


class Resnet18Decoder(FreezableModuleList):

    def __init__(self):
        super().__init__()

        self.layer_1 = DecoderBlock(512, 256)
        self.layer_2 = DecoderBlock(256, 128)
        self.layer_3 = DecoderBlock(128, 64)
        self.layer_4 = DecoderBlock(64, 64, kernel_size=1)

        self.unpool = MaxUnpool2d(kernel_size=3, stride=2, padding=1)
        self.final_1 = nn.Conv2d(64, 2, 3)
        self.relu = nn.ReLU(inplace=True)
        self.final_2 = nn.Conv2d(2, 2, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, encoder_feats, maxpool_idxs):

        x = self.layer_1(x, encoder_feats[3])
        x = self.layer_2(x, encoder_feats[2])
        x = self.layer_3(x, encoder_feats[1])
        x = self.layer_4(x, encoder_feats[0])
        x = self.unpool(x, maxpool_idxs, output_size=(112, 112))

        x = self.final_1(x)
        x = self.relu(x)
        x = self.final_2(x)
        x = functional.interpolate(x, (224, 224))
        x = self.softmax(x)

        return x


class MaskCNN(nn.Module):

    def __init__(self, num_classes=2, num_channels=5):
        super().__init__()

        # Segmentation stage
        self.encoder = Resnet18Encoder(in_channels=num_channels)
        self.decoder = Resnet18Decoder()

        # Classification stage
        self.encoder2 = Resnet18Encoder(in_channels=num_channels)
        self.avgpool = FreezableAdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = FreezableLinear(512, num_classes, bias=True)

    def forward(self, x):

        # Image -> Features
        y = self.encoder(x)

        # Features -> Mask
        y = self.decoder(y, self.encoder.feats, self.encoder.maxpool_idxs)
        y = y[:, 0:1, ...]

        # Adjust image from mask: we only fetch the former
        # NOTE: - RGB: image is (0, 1, 2). Input mask is (3,),
        #       - RGB + NIR image is (0, 1, 2, 3). Input mask is (4,),
        #       - In both cases this is the image is :-1, mask is -1
        # NOTE: the paper recommends multiplication, but in this case
        # concat-ing the segmentation mask seems to produce better results
        # x = torch.multiply(x[:, :-1, ...], y)
        x = torch.concat((x[:, :-1, ...], y), dim=1)

        # Updated Image -> Features
        x = self.encoder2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        z = self.fc(x)

        return y, z