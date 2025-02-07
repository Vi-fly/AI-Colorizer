import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import os
from .gan_model import UNetGenerator

# Initialize model loader
def load_gan_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    model_path = os.path.join(model_dir, "colorization_gan_80.pth")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"GAN model 80 not found at: {model_path}")
    
    generator = UNetGenerator().to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    generator.load_state_dict(state_dict)
    generator.eval()
    return generator

# Global model cache
GAN_MODEL = None

def enhance_contrast(image, clip_limit=3.0):
    """Adjustable contrast enhancement using CLAHE"""
    img = np.array(image)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    limg = cv2.merge([clahe.apply(l), a, b])
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced)

def adjust_saturation(image, saturation_factor=1.0):
    """Adjustable color saturation"""
    converter = ImageEnhance.Color(image)
    return converter.enhance(saturation_factor)

def colorize_image(image, contrast_factor=1.0, saturation_factor=1.2):
    """Colorize with GAN v70 and adjustable parameters"""
    global GAN_MODEL
    if not GAN_MODEL:
        GAN_MODEL = load_gan_model()
    
    # Convert to tensor and pad
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    
    # Calculate padding
    w, h = image.size
    pad_w = (256 - w % 256) % 256
    pad_h = (256 - h % 256) % 256
    padding = (pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2)
    
    # Apply padding
    input_tensor = transform(image).unsqueeze(0).to(GAN_MODEL.device)
    padded_tensor = torch.nn.functional.pad(input_tensor, padding, mode='reflect')
    
    # Process through model
    with torch.no_grad():
        output_tensor = GAN_MODEL(padded_tensor)
    
    # Remove padding
    output_tensor = output_tensor[:, :, pad_h//2 : pad_h//2 + h, pad_w//2 : pad_w//2 + w]
    
    # Convert to image
    output_image = transforms.ToPILImage()(output_tensor.squeeze(0).cpu() * 0.5 + 0.5)
    
    # Apply post-processing adjustments
    output_image = enhance_contrast(output_image, clip_limit=contrast_factor*3.0)
    output_image = adjust_saturation(output_image, saturation_factor)
    
    return output_image