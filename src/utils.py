import cv2
import numpy as np

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def save_image(image, save_path):
    # Convert RGB to BGR for OpenCV saving
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, image_bgr)

def preprocess_image(image, target_size=(512, 512)):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)

def overlay_attention_map(image, attention_map, alpha=0.5, colormap=cv2.COLORMAP_JET):
    attention_map = attention_map - np.min(attention_map)
    if np.max(attention_map) != 0:
        attention_map = attention_map / np.max(attention_map)
    attention_map = (attention_map * 255).astype(np.uint8)
    attention_map = cv2.resize(attention_map, (image.shape[1], image.shape[0]))
    colored_map = cv2.applyColorMap(attention_map, colormap)
    colored_map = cv2.cvtColor(colored_map, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image, 1 - alpha, colored_map, alpha, 0)
    return overlay

def display_image(image):
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.axis('off')
    plt.show()