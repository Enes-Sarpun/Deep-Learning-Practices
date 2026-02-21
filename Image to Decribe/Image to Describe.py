"""
Problem Description: 
    Image to Describe is a Python script.
    BLIP Model Technology is used to generate descriptions for images. Automatically processes images with Hugging Face.
    - Multi Modal Model
    - Captioning
    - Input: Image, Output: Description
    - Builds;
        - Image Encoder : Image Transformer.
        - Text Decoder : Transformer based language model.
        - Cross Attention Mechanism to connect image and text features.
"""

# Load necessary libraries;
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import torch

# Download the Image;
url = 'https://st2.depositphotos.com/1000877/6437/i/600/depositphotos_64377443-stock-photo-mother-and-child-playing-with.jpg'
try:
    resp = requests.get(url, stream=True, timeout=10)
    resp.raise_for_status()
    image = Image.open(resp.raw).convert('RGB')
except Exception as e:
    raise SystemExit(f"Failed to download or open image from {url}: {e}")

# Image to Tensor;
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")  # Converts the image into input tensors suitable for the model.
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")  # Loads the BLIP model that can generate text from visual input.

# Caption Generation Function;
def generate_caption(image, max_length: int = 50, num_beams: int = 5):
    """Generate a caption for a PIL image.

    Returns the decoded caption string. Handles device placement.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Convert Image to Tensor;
    inputs = processor(image, return_tensors="pt")  # Normalization for Image and conversion to tensor format.
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Caption Text Generation;
    with torch.no_grad():  # It disables gradient calculations only for inference (output generation).
        out = model.generate(**inputs, max_length=max_length, num_beams=num_beams)

    # Tokens for Humans;
    caption = processor.decode(out[0], skip_special_tokens=True)  # It converts the generated tokens into human-readable text.
    return caption


if __name__ == "__main__":
    caption = generate_caption(image)
    print("Generated Caption:", caption)

# Finished.