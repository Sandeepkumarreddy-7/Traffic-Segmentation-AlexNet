import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import gradio as gr

from config import *
from models.segmentation_model import SegmentationModel
from visualize import colorize_mask
from utils import load_model


# ==========================================
# Device
# ==========================================
device = DEVICE

# ==========================================
# Load Model
# ==========================================
model = SegmentationModel(num_classes=NUM_CLASSES).to(device)

model = load_model(
    model,
    MODEL_PATH,
    device
)

model.eval()

# ==========================================
# Image Transform
# ==========================================
transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor()
])

# ==========================================
# Prediction Function
# ==========================================
def predict(image):

    original = image.copy()

    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(image)

        prediction = torch.argmax(output, dim=1)

    prediction = prediction.squeeze(0).cpu().numpy()

    color_prediction = colorize_mask(prediction)

    color_prediction = Image.fromarray(color_prediction)

    return original, color_prediction


# ==========================================
# Gradio Interface
# ==========================================
demo = gr.Interface(
    fn=predict,

    inputs=gr.Image(type="pil", label="Upload Traffic Image"),

    outputs=[
        gr.Image(label="Original Image"),
        gr.Image(label="Predicted Segmentation")
    ],

    title="🚗 Traffic Scene Semantic Segmentation",

    description="""
Upload a traffic scene image.

The model predicts semantic segmentation using a
Custom AlexNet Encoder-Decoder Architecture trained on BDD100K.
"""
)

demo.launch()