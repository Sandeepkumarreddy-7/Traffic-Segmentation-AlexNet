import torch
import torch.nn as nn

from config import *
from models.segmentation_model import SegmentationModel
from preprocessing.dataloader import get_dataloader
from utils import load_model
from metrics import calculate_miou


def evaluate():

    # ==========================================
    # Device
    # ==========================================
    device = DEVICE

    # ==========================================
    # Validation DataLoader
    # ==========================================
    val_loader = get_dataloader(
        image_dir=VAL_IMAGE_DIR,
        mask_dir=VAL_MASK_DIR,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # ==========================================
    # Load Model
    # ==========================================
    model = SegmentationModel(
        num_classes=NUM_CLASSES
    ).to(device)

    model = load_model(
        model,
        MODEL_PATH,
        device
    )

    # ==========================================
    # Loss Function
    # ==========================================
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # ==========================================
    # Evaluation Mode
    # ==========================================
    model.eval()

    total_loss = 0.0
    total_miou = 0.0

    correct_pixels = 0
    total_pixels = 0

    # ==========================================
    # No Gradient Calculation
    # ==========================================
    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.long().to(device)

            # Forward Pass
            outputs = model(images)

            # Loss
            loss = criterion(outputs, masks)

            total_loss += loss.item()

            # Prediction
            predictions = torch.argmax(outputs, dim=1)

            # Mean IoU
            miou = calculate_miou(
                predictions,
                masks,
                NUM_CLASSES
            )

            total_miou += miou

            # Ignore pixels with label 255
            valid_mask = masks != 255

            correct_pixels += (
                (predictions == masks) & valid_mask
            ).sum().item()

            total_pixels += valid_mask.sum().item()

    # ==========================================
    # Final Metrics
    # ==========================================
    avg_loss = total_loss / len(val_loader)

    pixel_accuracy = (
        correct_pixels / total_pixels
    ) * 100

    mean_iou = total_miou / len(val_loader)

    # ==========================================
    # Print Results
    # ==========================================
    print("\n========== Evaluation ==========")
    print(f"Validation Loss : {avg_loss:.4f}")
    print(f"Pixel Accuracy  : {pixel_accuracy:.2f}%")
    print(f"Mean IoU        : {mean_iou:.4f}")
    print("================================")


if __name__ == "__main__":
    evaluate()