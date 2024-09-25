import os
import random
import math
import io
from PIL import Image
import cv2
import numpy as np
from typing import Tuple


def resize_and_crop_to_fit_cv2(img, target_resolutions):
    original_h, original_w = img.shape[:2]
    if original_h <= 10 or original_w <= 10:
        raise ValueError(f"BEFORE: Image too small: {original_w}x{original_h}")
    
    original_aspect = original_w / original_h if original_h != 0 else 1

    # Find the closest target resolution
    target_w, target_h = min(target_resolutions, 
                             key=lambda res: abs((res[0]/res[1]) - original_aspect))
    target_aspect = target_w / target_h

    if original_aspect > target_aspect:
        # Image is wider than target
        scale = target_h / original_h
        new_w, new_h = int(original_w * scale), target_h
    else:
        # Image is taller than target
        scale = target_w / original_w
        new_w, new_h = target_w, int(original_h * scale)

    # Resize the image
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Ensure correct channel order (BGR to RGB)
    if len(resized.shape) == 3 and resized.shape[2] == 3:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    elif len(resized.shape) == 3 and resized.shape[2] == 4:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGBA)

    # Crop to final size
    start_x = max((new_w - target_w) // 2, 0)
    start_y = max((new_h - target_h) // 2, 0)
    cropped = resized[start_y:start_y+target_h, start_x:start_x+target_w]

    if cropped.shape[0] <= 10 or cropped.shape[1] <= 10:
        raise ValueError(f"AFTER: Image too small: {cropped.shape[1]}x{cropped.shape[0]}")
    
    return cropped

def random_light_color():
    lower = 250
    return (random.randint(lower, 255), random.randint(lower, 255), random.randint(lower, 255))

def fill_transparent_with_color_cv2(img, padding=0):
    """
    Fill transparent areas of an image with a random light color using OpenCV and add padding.
    
    :param img: OpenCV image (numpy array) with alpha channel
    :param padding: Amount of padding to add around the image (default: 0)
    :return: Image with transparent areas filled and padding added
    """
    if img.shape[2] == 4:
        # Check if alpha channel is valid
        if np.all(img[:, :, 3] == 0):
            # If alpha is all zero, treat as RGB
            img = img[:, :, :3]
        else:
            # Fill transparent with random light color
            alpha = img[:, :, 3]
            rgb = img[:, :, :3]
            background_color = random_light_color()
            background = np.full_like(rgb, background_color)
            mask = alpha[:, :, np.newaxis] / 255.0
            img = (rgb * mask + background * (1 - mask)).astype(np.uint8)
    
    if padding > 0:
        # Add padding with random light color
        h, w = img.shape[:2]
        padded_img = np.full((h + 2 * padding, w + 2 * padding, 3), random_light_color(), dtype=np.uint8)
        padded_img[padding:padding + h, padding:padding + w] = img
        return padded_img
    else:
        return img
    
def fill_transparent_with_color(img,padding):
    if img.mode == 'P':
        img = img.convert("RGBA") 
    if img.mode == 'RGBA':
        new_size = (img.width + 2 * padding, img.height + 2 * padding)
        bg_color = random_light_color()
        background = Image.new("RGBA", new_size, bg_color)
        position = (padding, padding)

        # Ensure the image and background are in RGBA mode
        img = img.convert("RGBA")
        background = background.convert("RGBA")

        # Create an empty image with the same size as the background
        temp_img = Image.new("RGBA", background.size)

        # Position the foreground on the temp image
        temp_img.paste(img, position, img)

        # Blend the images using alpha compositing
        blended = Image.alpha_composite(background, temp_img)

        return blended
    elif img.mode == 'RGB':
        return img
    else:
        print("Unknown mode: ", img.mode)
    return img

def center_crop_square(image):
    width, height = image.size
    min_side = min(width, height)
    left = (width - min_side) / 2
    top = (height - min_side) / 2
    right = (width + min_side) / 2
    bottom = (height + min_side) / 2
    return image.crop((left, top, right, bottom))

def svg_scaling(image_path,max_side_length,output_path,do_center_square_crop,flip_probability):
    import cairosvg
    png_data = cairosvg.svg2png(url=image_path, output_width=2 * max_side_length, output_height=2 * max_side_length)
    img = Image.open(io.BytesIO(png_data))

    if do_center_square_crop:
        img = center_crop_square(img)

    if random.random() < flip_probability:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    width, height = img.size
    scale_factor = max_side_length / max(width, height)
    new_width = math.floor(width * scale_factor)
    new_height = math.floor(height * scale_factor)

    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    #fill transparent with color
    resized_img = fill_transparent_with_color(resized_img, 0)
    resized_img.save(os.path.splitext(output_path)[0] + ".png", format='PNG', quality=98)

    os.remove(image_path)
    return resized_img



def upscale_to_1024(img: np.ndarray) -> np.ndarray:
    assert img.ndim == 4, "Expected a batch of images"
    batch_size, height, width, channels = img.shape
    target_sizes = [
        (
            math.floor(width * 1024 / min(width, height)),
            math.floor(height * 1024 / min(width, height))
        )
        for _ in range(batch_size)
    ]
    upscaled_imgs = np.array([
        cv2.resize(img[i], target_sizes[i], interpolation=cv2.INTER_AREA)
        for i in range(batch_size)
    ])
    return upscaled_imgs