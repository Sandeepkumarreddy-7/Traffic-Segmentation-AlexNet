import torch
import torch.nn as nn
import torch.optim as optim

from config import *

from models.segmentation_model import SegmentationModel
from preprocessing.dataloader import get_dataloader
from utils import save_model


def train():

    # ==========================================
    # Device
    # ==========================================
    device = DEVICE
    print(f"Using Device: {device}")

    # ==========================================
    # DataLoader
    # ==========================================
    train_loader = get_dataloader(
        image_dir=TRAIN_IMAGE_DIR,
        mask_dir=TRAIN_MASK_DIR,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    print(f"Total Training Batches: {len(train_loader)}")

    # ==========================================
    # Model
    # ==========================================
    model = SegmentationModel(
        num_classes=NUM_CLASSES
    ).to(device)

    # ==========================================
    # Loss Function
    # ==========================================
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # ==========================================
    # Optimizer
    # ==========================================
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # ==========================================
    # Training
    # ==========================================
    print("\nTraining Started...\n")

    for epoch in range(NUM_EPOCHS):

        model.train()

        running_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):

            # Move data to device
            images = images.to(device)
            masks = masks.long().to(device)

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward Pass
            outputs = model(images)

            # Calculate Loss
            loss = criterion(outputs, masks)

            # Backpropagation
            loss.backward()

            # Update Weights
            optimizer.step()

            # Accumulate Loss
            running_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                    print(
                        f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
                        f"Batch [{batch_idx+1}/{len(train_loader)}] "
                        f"Loss: {loss.item():.4f}"
                    )

        epoch_loss = running_loss / len(train_loader)

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"Loss: {epoch_loss:.4f}"
        )

    # ==========================================
    # Save Model
    # ==========================================
    save_model(model, MODEL_PATH)

    print("\n===================================")
    print(" Training Completed Successfully ")
    print("===================================")


if __name__ == "__main__":
    train()