import torch
import torchvision

print("=" * 50)
print("PyTorch Version :", torch.__version__)
print("TorchVision Version :", torchvision.__version__)
print("CUDA Available :", torch.cuda.is_available())
print("Current Device :", "GPU" if torch.cuda.is_available() else "CPU")
print("=" * 50)

x = torch.rand(3, 3)
print("\nRandom Tensor:\n")
print(x)