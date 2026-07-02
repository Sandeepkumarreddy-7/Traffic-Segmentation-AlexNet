import torch
from PIL import Image
import torchvision.transforms as transforms

from config import *
from models.segmentation_model import SegmentationModel
from utils import load_model


def test():

    device = DEVICE

    model = SegmentationModel(
        num_classes=NUM_CLASSES
    ).to(device)

    model = load_model(
        model,
        MODEL_PATH,
        device
    )

    image = Image.open(
        "dataset/10k/val/0000f77c-6257be58.jpg"
    ).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor()
    ])

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        output = model(image)

        prediction = torch.argmax(
            output,
            dim=1
        )

    print("Prediction Shape :", prediction.shape)


if __name__ == "__main__":
    test()