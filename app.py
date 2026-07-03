import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from config import *
from models.segmentation_model import SegmentationModel
from utils import load_model
from visualize import colorize_mask


# =====================================================
# Device
# =====================================================

device = DEVICE


# =====================================================
# Load Trained Model
# =====================================================

model = SegmentationModel(num_classes=NUM_CLASSES)
model = model.to(device)

model = load_model(
    model=model,
    path=MODEL_PATH,
    device=device
)

model.eval()


# =====================================================
# Image Transform
# =====================================================

transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor()
])
# =====================================================
# Prediction Function
# =====================================================

def predict(image):

    if image is None:
        return None, None, "Please upload an image."

    original = image.copy()

    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1)

    prediction = prediction.squeeze(0).cpu().numpy()

    color_mask = colorize_mask(prediction)
    color_mask = Image.fromarray(color_mask)

    status = "✅ Segmentation Completed Successfully"

    return original, color_mask, status
# =====================================================
# Professional Gradio UI
# =====================================================

with gr.Blocks(
    title="Traffic Scene Semantic Segmentation",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        """
# 🚗 Traffic Scene Semantic Segmentation

### Custom AlexNet-based Encoder–Decoder Architecture

Semantic segmentation of traffic scenes using **PyTorch** and the **BDD100K** dataset.

Upload a traffic scene image and click **Predict Segmentation**.
"""
    )

    with gr.Row():

        input_image = gr.Image(
            type="pil",
            label="📤 Upload Traffic Image"
        )

        output_image = gr.Image(
            label="🎯 Predicted Segmentation"
        )

    status = gr.Textbox(
        label="Status",
        interactive=False
    )

    predict_button = gr.Button(
        "🚀 Predict Segmentation",
        variant="primary",
        size="lg"
    )

    predict_button.click(
        fn=predict,
        inputs=input_image,
        outputs=[
            input_image,
            output_image,
            status
        ]
    )
    gr.Markdown("---")

    with gr.Row():

        with gr.Column():

            gr.Markdown("## 📊 Model Information")

            gr.Markdown("""
**Model:** Custom AlexNet Encoder–Decoder

**Dataset:** BDD100K

**Framework:** PyTorch

**Semantic Classes:** 19

**Device:** CPU
""")

        with gr.Column():

            gr.Markdown("## 📈 Performance")

            gr.Markdown("""
✅ Validation Loss : **0.7282**

✅ Pixel Accuracy : **77.90%**

✅ Mean IoU : **0.2564**
""")

    gr.Markdown("---")

    gr.Markdown("""
## 📖 About

This application performs **pixel-wise semantic segmentation** of traffic scene images.

Each pixel is assigned to one of **19 semantic classes**, allowing the model to identify roads, vehicles, buildings, vegetation, sky, pedestrians, and other important traffic scene objects.

The model is built using a **Custom AlexNet-based Encoder–Decoder Architecture** and trained on the **BDD100K** dataset.
""")
    gr.Markdown("---")

    gr.Markdown(
        """
### 👨‍💻 Developed by

**Dammuru Sandeep Kumar Reddy**

🔗 GitHub Repository:

https://github.com/Sandeepkumarreddy-7/Traffic-Segmentation-AlexNet

⭐ If you like this project, consider giving it a star on GitHub!
"""
    )

# =====================================================
# Launch Application
# =====================================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )