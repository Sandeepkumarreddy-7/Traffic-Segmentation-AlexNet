import torch
import torch.nn as nn


class AlexNetEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=11,
            stride=4,
            padding=2
        )

        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(
            kernel_size=3,
            stride=2
        )
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=192,
            kernel_size=5,
            stride=1,
            padding=2
        )

        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(
        kernel_size=3,
        stride=2
        )
        self.conv3 = nn.Conv2d(
            in_channels=192,
            out_channels=384,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(
            in_channels=384,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.relu5 = nn.ReLU(inplace=True)

        self.pool5 = nn.MaxPool2d(
            kernel_size=3,
            stride=2
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)


        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)

 
        return x