import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from config import *
from models.segmentation_model import SegmentationModel
from utils import load_model
from visualize import colorize_mask


def test():

    device = DEVICE

    # ============================
    # Load Model
    # ============================
    model = SegmentationModel(
        num_classes=NUM_CLASSES
    ).to(device)

    model = load_model(
        model,
        MODEL_PATH,
        device
    )

    # ============================
    # Image
    # ============================
    image_dir = VAL_IMAGE_DIR

    image_name = sorted(os.listdir(image_dir))[0]

    image_path = os.path.join(image_dir, image_name)

    print("Testing:", image_name)

    image = Image.open(image_path).convert("RGB")

    original_image = image.copy()

    # ============================
    # Ground Truth Mask
    # ============================
    mask_name = image_name.replace(".jpg", "_train_id.png")

    mask_path = os.path.join(
        VAL_MASK_DIR,
        mask_name
    )

    mask = Image.open(mask_path)

    mask = TF.resize(
        mask,
        (256, 512),
        interpolation=InterpolationMode.NEAREST
    )

    mask = np.array(mask)

    ground_truth = colorize_mask(mask)

    # ============================
    # Transform Image
    # ============================
    transform = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor()
    ])

    image = transform(image)

    image = image.unsqueeze(0).to(device)

    # ============================
    # Prediction
    # ============================
    model.eval()

    with torch.no_grad():

        output = model(image)

        prediction = torch.argmax(
            output,
            dim=1
        )

    prediction = prediction.squeeze(0).cpu().numpy()

    prediction = colorize_mask(prediction)

    # ============================
    # Display
    # ============================
    plt.figure(figsize=(18,6))

    plt.subplot(1,3,1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(ground_truth)
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(prediction)
    plt.title("Prediction")
    plt.axis("off")

    plt.tight_layout()
    
    os.makedirs("outputs/predictions", exist_ok=True)

    plt.savefig(
        "outputs/predictions/prediction_result.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()


if __name__ == "__main__":
    test()