from turtle import forward, pd
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torchvision


class DecoderBlock(nn.Module):

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


class Resnet18Encoder(nn.Module):

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


class Resnet50Decoder(nn.ModuleList):

    def __init__(self):
        super().__init__()

        self.layer_1 = DecoderBlock(2048, 1024)
        self.layer_2 = DecoderBlock(1024, 512)
        self.layer_3 = DecoderBlock(512, 256)
        self.layer_4 = DecoderBlock(256, 64, kernel_size=5)

        self.unpool = nn.MaxUnpool2d(kernel_size=3, stride=2, padding=1)
        self.final_1 = nn.Conv2d(64, 1, 3)
        self.relu = nn.ReLU(inplace=True)
        self.final_2 = nn.Conv2d(1, 1, 3)

    def forward(self, x, encoder_feats, maxpool_idxs):

        x = self.layer_1(x, encoder_feats[3])
        x = self.layer_2(x, encoder_feats[2])
        x = self.layer_3(x, encoder_feats[1])
        x = self.layer_4(x, encoder_feats[0])
        x = self.unpool(x, maxpool_idxs, output_size=(128, 128))

        x = self.final_1(x)
        x = self.relu(x)
        x = self.final_2(x)
        x = functional.interpolate(x, (256, 256))

        return x


class Resnet18Decoder(nn.ModuleList):

    def __init__(self):
        super().__init__()

        self.layer_1 = DecoderBlock(512, 256)
        self.layer_2 = DecoderBlock(256, 128)
        self.layer_3 = DecoderBlock(128, 64)
        self.layer_4 = DecoderBlock(64, 64, kernel_size=5)

        self.unpool = nn.MaxUnpool2d(kernel_size=3, stride=2, padding=1)
        self.final_1 = nn.Conv2d(64, 2, 3)
        self.relu = nn.ReLU(inplace=True)
        self.final_2 = nn.Conv2d(2, 2, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, encoder_feats, maxpool_idxs):

        x = self.layer_1(x, encoder_feats[3])
        x = self.layer_2(x, encoder_feats[2])
        x = self.layer_3(x, encoder_feats[1])
        x = self.layer_4(x, encoder_feats[0])
        x = self.unpool(x, maxpool_idxs, output_size=(128, 128))

        x = self.final_1(x)
        x = self.relu(x)
        x = self.final_2(x)
        x = functional.interpolate(x, (256, 256))
        x = self.softmax(x)

        return x


class MaskCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Resnet18Encoder(in_channels=4)
        self.encoder2 = Resnet18Encoder(in_channels=3)
        self.decoder = Resnet18Decoder()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, 2, bias=True)     # Resnet50: 2048

    def forward(self, x):

        # Image -> Features
        y = self.encoder(x)

        # Features -> Mask
        y = self.decoder(y, self.encoder.feats, self.encoder.maxpool_idxs)
        y = y[:, 0:1, ...]

        # Adjust image from mask
        x = torch.multiply(x[:, :3, ...], y)

        # Updated Image -> Features
        x = self.encoder2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        z = self.fc(x)

        return y, z


if __name__ == '__main__':

    model = MaskCNN()

    inp = torch.rand((1, 4, 256, 256))

    x, y = model(inp)
    print(y)

    inp = inp[0, :3, :, :]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2)
    x = x.detach()
    ax[0].imshow(inp.moveaxis(0, -1) / inp.max())
    ax[1].imshow(x[0, 0, ...] / x.max())
    plt.show()