import sys
import os
import torch
import cv2
import numpy as np

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spandrel import ModelLoader, ImageModelDescriptor
from utils.config import spandrel_model_path

model = ModelLoader().load_from_file(spandrel_model_path)

assert isinstance(model, ImageModelDescriptor)
model.cuda().eval()

if __name__ == "__main__":
    def process(image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model(image)
    
def open_image(image_path: str) -> torch.Tensor:
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

    image = torch.from_numpy(rgb_image).permute(2, 0, 1).unsqueeze(0).cuda()

    return image

output = process(open_image("vector_white.png"))

#save image
cv2.imwrite(
    "results.png",
    (output.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
)

print(output.shape)


