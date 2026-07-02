import torch
from models.alexnet_encoder import AlexNetEncoder

model = AlexNetEncoder()

x = torch.randn(1, 3, 720, 1280)

output = model(x)

print(output.shape)