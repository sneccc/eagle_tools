import os
import random
import math
import io
from PIL import Image
import cv2
import numpy as np
from realesrgan_ncnn_py import Realesrgan

def resize_and_crop_to_fit_cv2(img, target_resolutions):
    """
    Resize and crop the image to fit the closest target resolution using OpenCV.
    
    :param img: OpenCV image (numpy array)
    :param target_resolutions: List of tuples (width, height)
    :return: Resized and cropped image
    """
    original_h, original_w = img.shape[:2]
    original_aspect = original_w / original_h

    # Find the closest target resolution
    target_w, target_h = min(target_resolutions, 
                             key=lambda res: abs(res[0]/res[1] - original_aspect))
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
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR )

    # Crop to final size
    start_x = (new_w - target_w) // 2
    start_y = (new_h - target_h) // 2
    cropped = resized[start_y:start_y+target_h, start_x:start_x+target_w]

    return cropped


def resize_and_crop_to_fit(image,target_resolutions):
    original_width, original_height = image.size
    original_aspect_ratio = original_width / original_height

    # Find the closest target resolution based on aspect ratio
    closest_diff = float('inf')
    for target_width, target_height in target_resolutions:
        target_aspect_ratio = target_width / target_height
        aspect_ratio_diff = abs(target_aspect_ratio - original_aspect_ratio)
        
        if aspect_ratio_diff < closest_diff:
            closest_diff = aspect_ratio_diff
            closest_resolution = (target_width, target_height)
    
    # Resize image to closest resolution maintaining aspect ratio
    image_resized = image.resize(closest_resolution, Image.LANCZOS)
    
    # Crop to the exact target resolution if necessary
    resized_width, resized_height = image_resized.size
    if resized_width != closest_resolution[0] or resized_height != closest_resolution[1]:
        # Calculate crop area to keep the crop centered
        left = (resized_width - closest_resolution[0]) / 2
        top = (resized_height - closest_resolution[1]) / 2
        right = (resized_width + closest_resolution[0]) / 2
        bottom = (resized_height + closest_resolution[1]) / 2

        image_cropped = image_resized.crop((left, top, right, bottom))
    else:
        image_cropped = image_resized

    return image_cropped

def random_light_color():
    lower = 250
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def fill_transparent_with_color_cv2(img):
    """
    Fill transparent areas of an image with a random light color using OpenCV.
    
    :param img: OpenCV image (numpy array) with alpha channel
    :return: Image with transparent areas filled
    """
    if img.shape[2] == 4:
        # Check if alpha channel is valid
        if np.all(img[:,:,3] == 0):
            # If alpha is all zero, treat as RGB
            return img[:,:,:3]
        else:
            # Fill transparent with random light color
            alpha = img[:,:,3]
            rgb = img[:,:,:3]
            background_color = random_light_color()
            background = np.full_like(rgb, background_color)
            mask = alpha[:,:,np.newaxis] / 255.0
            return (rgb * mask + background * (1 - mask)).astype(np.uint8)
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
    resized_img.save(os.path.splitext(output_path)[0] + ".png", format='PNG', quality=98)

    os.remove(image_path)
    return resized_img

def upscale_image_cv2(img, output_path, pixelart=False, isEsganUpscale=False, gpuid=0,model_id=4):
    """
    Upscale an image based on given parameters and save it.
    
    :param img: OpenCV image (numpy array)
    :param output_path: Path to save the upscaled image
    :param pixelart: Boolean, if True, use nearest neighbor upscaling
    :param isEsganUpscale: Boolean, if True, use Realesrgan upscaling
    :param gpuid: Integer, GPU ID to use for Realesrgan
    :return: Upscaled OpenCV image (numpy array)
    """
    if pixelart:
        upscaled_img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
    elif isEsganUpscale:
        realesrgan = Realesrgan(model=model_id,gpuid=gpuid)
        upscaled_img = realesrgan.process_cv2(img)
    else:
        upscaled_img = img
    return upscaled_img


def upscale_to_1024(img):
    target_size = (
        math.floor(img.shape[1] * 1024 / min(img.shape[1], img.shape[0])),
        math.floor(img.shape[0] * 1024 / min(img.shape[1], img.shape[0]))
    )
    upscaled_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return upscaled_img