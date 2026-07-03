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

    overlay = create_overlay(original, color_mask)

    status = "✅ Segmentation Completed Successfully"

    return original, overlay, color_mask, status
def create_overlay(original, color_mask, alpha=0.50):
    """
    Blend original image with segmentation mask.
    """

    # Resize segmentation mask to original image size
    color_mask = color_mask.resize(original.size)

    original = np.array(original).astype(np.float32)
    color_mask = np.array(color_mask).astype(np.float32)

    overlay = (1 - alpha) * original + alpha * color_mask
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)
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

        overlay_image = gr.Image(
            label="🌈 Overlay Result"
        )

        output_image = gr.Image(
            label="🎯 Segmentation Mask"
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
            overlay_image,
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
    gr.Markdown("## 🎨 Semantic Class Color Legend")

    legend_html = """
    <table style="width:100%; border-collapse: collapse;">
        <tr><td><div style="width:20px;height:20px;background:rgb(128,64,128);display:inline-block;border:1px solid black;"></div></td><td>Road</td></tr>
        <tr><td><div style="width:20px;height:20px;background:rgb(244,35,232);display:inline-block;border:1px solid black;"></div></td><td>Sidewalk</td></tr>
        <tr><td><div style="width:20px;height:20px;background:rgb(70,70,70);display:inline-block;border:1px solid black;"></div></td><td>Building</td></tr>
        <tr><td><div style="width:20px;height:20px;background:rgb(102,102,156);display:inline-block;border:1px solid black;"></div></td><td>Wall</td></tr>
        <tr><td><div style="width:20px;height:20px;background:rgb(190,153,153);display:inline-block;border:1px solid black;"></div></td><td>Fence</td></tr>
        <tr><td><div style="width:20px;height:20px;background:rgb(153,153,153);display:inline-block;border:1px solid black;"></div></td><td>Pole</td></tr>
        <tr><td><div style="width:20px;height:20px;background:rgb(250,170,30);display:inline-block;border:1px solid black;"></div></td><td>Traffic Light</td></tr>
        <tr><td><div style="width:20px;height:20px;background:rgb(220,220,0);display:inline-block;border:1px solid black;"></div></td><td>Traffic Sign</td></tr>
        <tr><td><div style="width:20px;height:20px;background:rgb(107,142,35);display:inline-block;border:1px solid black;"></div></td><td>Vegetation</td></tr>
        <tr><td><div style="width:20px;height:20px;background:rgb(152,251,152);display:inline-block;border:1px solid black;"></div></td><td>Terrain</td></tr>
        <tr><td><div style="width:20px;height:20px;background:rgb(70,130,180);display:inline-block;border:1px solid black;"></div></td><td>Sky</td></tr>
        <tr><td><div style="width:20px;height:20px;background:rgb(220,20,60);display:inline-block;border:1px solid black;"></div></td><td>Person</td></tr>
        <tr><td><div style="width:20px;height:20px;background:rgb(255,0,0);display:inline-block;border:1px solid black;"></div></td><td>Rider</td></tr>
        <tr><td><div style="width:20px;height:20px;background:rgb(0,0,142);display:inline-block;border:1px solid black;"></div></td><td>Car</td></tr>
        <tr><td><div style="width:20px;height:20px;background:rgb(0,0,70);display:inline-block;border:1px solid black;"></div></td><td>Truck</td></tr>
        <tr><td><div style="width:20px;height:20px;background:rgb(0,60,100);display:inline-block;border:1px solid black;"></div></td><td>Bus</td></tr>
        <tr><td><div style="width:20px;height:20px;background:rgb(0,80,100);display:inline-block;border:1px solid black;"></div></td><td>Train</td></tr>
        <tr><td><div style="width:20px;height:20px;background:rgb(0,0,230);display:inline-block;border:1px solid black;"></div></td><td>Motorcycle</td></tr>
        <tr><td><div style="width:20px;height:20px;background:rgb(119,11,32);display:inline-block;border:1px solid black;"></div></td><td>Bicycle</td></tr>
    </table>
    """

    gr.HTML(legend_html)

    gr.Markdown(
        """
### 👨‍💻 Developed by

**Sandeep**

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