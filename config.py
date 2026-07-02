import torch

# ===============================
# Dataset Paths
# ===============================

TRAIN_IMAGE_DIR = "dataset/10k/train"
TRAIN_MASK_DIR = "dataset/labels/train"

VAL_IMAGE_DIR = "dataset/10k/val"
VAL_MASK_DIR = "dataset/labels/val"

# ===============================
# Training Parameters
# ===============================

BATCH_SIZE = 4

NUM_EPOCHS = 1

LEARNING_RATE = 0.001

NUM_CLASSES = 19

# ===============================
# Image Size
# ===============================

IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1280

# ===============================
# Device
# ===============================

DEVICE = torch.device("cpu")

# ===============================
# Model Save Path
# ===============================

MODEL_PATH = "alexnet_segmentation.pth"