import torch
import torch.nn as nn
import torch.nn.functional as F

from models.alexnet_encoder import AlexNetEncoder
from models.decoder import Decoder


class SegmentationModel(nn.Module):

    def __init__(self, num_classes=19):
        super().__init__()

        self.encoder = AlexNetEncoder()
        self.decoder = Decoder()

    def forward(self, x):

        # Save original image size
        original_size = x.shape[2:]

        # Encoder
        x = self.encoder(x)

        # Decoder
        x = self.decoder(x)

        # Resize to original image size
        x = F.interpolate(
            x,
            size=original_size,
            mode="bilinear",
            align_corners=False
        )

        return x