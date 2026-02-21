"""
Vision Transformer (ViT
"""
# import libraries;
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer
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

# Processor and Model Loading;
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-base")

# Image Caption Generation Function;
pixel_values = processor(images = image, return_tensors="pt").pixel_values 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
pixel_values = pixel_values.to(device)

output_ids = model.generate(pixel_values, max_length = 32)

# Results;
caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Generated Caption:", caption)





# Finished.