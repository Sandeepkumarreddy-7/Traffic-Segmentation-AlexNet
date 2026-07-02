import torch

from models.segmentation_model import SegmentationModel

model = SegmentationModel()

x = torch.randn(1, 3, 720, 1280)

output = model(x)

print("Output Shape:", output.shape)