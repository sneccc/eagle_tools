import cv2
import numpy as np
import torch
from PIL import Image

def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to BGR."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def cv2_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert a BGR image to a normalized RGB tensor."""
    image_rgb = bgr_to_rgb(image)
    image_float = image_rgb.astype(np.float32) / 255.0
    return torch.from_numpy(image_float).permute(2, 0, 1).unsqueeze(0).cuda()

def tensor_to_cv2(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized RGB tensor to a BGR image."""
    image_np = tensor.mul(255).byte().cpu().numpy()
    image_np = image_np.transpose((1, 2, 0))  # Convert to [H, W, C]
    return rgb_to_bgr(image_np)

def convert_color(img, to='RGB'):
    """
    Convert image color space between BGR and RGB.

    Parameters:
    - img: Image as a NumPy array (OpenCV) or PIL Image.
    - to: Target color space ('RGB' or 'BGR').

    Returns:
    - Converted image in the desired color space.
    """
    if isinstance(img, np.ndarray):
        if to == 'RGB':
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif to == 'BGR':
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(f"Unsupported target color space: {to}")
    elif isinstance(img, Image.Image):
        if to == 'RGB' and img.mode == 'BGR':
            return img.convert('RGB')
        elif to == 'BGR' and img.mode == 'RGB':
            return img.convert('BGR')  # Note: PIL doesn't support BGR directly
        else:
            return img
    else:
        raise TypeError("Unsupported image type. Must be a NumPy array or PIL Image.")