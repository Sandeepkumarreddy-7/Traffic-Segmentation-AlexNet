import torch
import torch.nn as nn


class Decoder(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.up1 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=2
        )

        self.relu1 = nn.ReLU(inplace=True)
        self.up2 = nn.ConvTranspose2d(
        in_channels=128,
        out_channels=64,
        kernel_size=2,
        stride=2
        )
        self.relu2 = nn.ReLU(inplace=True)

        self.up3 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=2,
            stride=2
        )
        self.relu3 = nn.ReLU(inplace=True)

        self.up4 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=16,
            kernel_size=2,
            stride=2
        )
        self.relu4 = nn.ReLU(inplace=True)

        self.final = nn.Conv2d(
            in_channels=16,
            out_channels=num_classes,   # BDD100K classes
            kernel_size=1
        )

    def forward(self, x):

        x = self.up1(x)
        x = self.relu1(x)

        x = self.up2(x)
        x = self.relu2(x)

        x = self.up3(x)
        x = self.relu3(x)

        x = self.up4(x)
        x = self.relu4(x)

        x = self.final(x)

        return x
