import sys
import os
import torch
import cv2
import numpy as np

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spandrel import ModelLoader, ImageModelDescriptor
from utils.config import spandrel_model_path

# Load the model
model = ModelLoader().load_from_file(spandrel_model_path)
assert isinstance(model, ImageModelDescriptor)
model.cuda().eval()

def process(images: torch.Tensor) -> torch.Tensor:
    """
    Processes a batch of images through the model.
    
    Parameters:
    - images: A batch tensor of shape [Batch, Channels, Height, Width]
    
    Returns:
    - A batch tensor of processed images
    """
    with torch.no_grad():
        return model(images)

def open_image(image_path: str) -> torch.Tensor:
    """
    Loads an image from the given path and converts it to a tensor.
    
    Parameters:
    - image_path: Path to the image file.
    
    Returns:
    - A tensor of shape [1, Channels, Height, Width]
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.join(script_dir, image_path)
    print(f"Attempting to load image from: {abs_path}")

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"The image file does not exist: {abs_path}")

    cv2_image = cv2.imread(abs_path)

    if cv2_image is None:
        raise ValueError(f"Failed to load image. The file may be corrupted or in an unsupported format: {abs_path}")

    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

    # Convert to float and normalize to [0, 1]
    rgb_image = rgb_image.astype(np.float32) / 255.0

    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).unsqueeze(0).cuda()
    
    return image_tensor

def save_image(tensor: torch.Tensor, output_path: str):
    """
    Saves a tensor as an image to the specified path.
    
    Parameters:
    - tensor: A tensor of shape [Channels, Height, Width]
    - output_path: Path where the image will be saved.
    """
    # Convert tensor to CPU, remove batch dimension, permute to HWC, and convert to NumPy
    image_np = (tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    
    # Convert RGB back to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Save the image
    cv2.imwrite(output_path, image_bgr)
    print(f"Saved upscaled image to: {output_path}")

if __name__ == "__main__":
    # List of image filenames to process
    image_filenames = ["vector_white.png", "vector_black.png"]
    
    # Load all images and store their tensors in a list
    image_tensors = []
    for filename in image_filenames:
        tensor = open_image(filename)
        image_tensors.append(tensor)
    
    # Concatenate all tensors to form a batch
    batch_tensor = torch.cat(image_tensors, dim=0)  # Shape: [Batch, Channels, Height, Width]
    print(f"Batch tensor shape: {batch_tensor.shape}")
    
    # Process the batch
    output_batch = process(batch_tensor)
    print(f"Output batch tensor shape: {output_batch.shape}")
    
    # Save each processed image individually
    for i, output_tensor in enumerate(output_batch):
        # Define the output filename
        input_filename = image_filenames[i]
        base_name, ext = os.path.splitext(input_filename)
        output_filename = f"{base_name}_upscaled.png"
        
        # Save the image
        save_image(output_tensor, output_filename)