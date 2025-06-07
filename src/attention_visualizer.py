from diffusers import StableDiffusionPipeline
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def overlay_attention_map(image: Image.Image, attention_map: np.ndarray) -> Image.Image:
    attention_map = attention_map - np.min(attention_map)
    attention_map = attention_map / np.max(attention_map)
    attention_map = np.uint8(attention_map * 255)

    attention_map = Image.fromarray(attention_map).convert("L")
    attention_map = attention_map.resize(image.size, Image.BILINEAR)

    heatmap = Image.new("RGBA", image.size)
    heatmap.paste(attention_map, (0, 0), attention_map)

    combined = Image.alpha_composite(image.convert("RGBA"), heatmap)
    return combined

def save_attention_visualization(image: Image.Image, attention_map: np.ndarray, output_path: str):
    combined_image = overlay_attention_map(image, attention_map)
    combined_image.save(output_path)

def display_attention_visualization(image: Image.Image, attention_map: np.ndarray):
    combined_image = overlay_attention_map(image, attention_map)
    plt.imshow(combined_image)
    plt.axis('off')
    plt.show()