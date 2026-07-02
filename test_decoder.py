import torch
from models.decoder import Decoder

model = Decoder()

x = torch.randn(1, 256, 21, 39)

output = model(x)

print(output.shape)