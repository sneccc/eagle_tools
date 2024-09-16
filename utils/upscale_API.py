import threading
import queue
import os
import logging
import cv2
import torch
import numpy as np
from spandrel import ImageModelDescriptor, ModelLoader
import utils.config as config
MODEL_PATH = config.spandrel_model_path
logger = logging.getLogger(__name__)

class UpscaleAPI:
    def __init__(self, batch_size, use_esrgan, gpu_id, model_id):
        self.batch_size = batch_size
        self.use_esrgan = use_esrgan
        self.gpu_id = gpu_id
        self.model_id = model_id
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._process_queue, name="UpscaleAPIThread")
        self.thread.start()
        logger.info(f"UpscaleAPI initialized with batch_size={batch_size}, use_esrgan={use_esrgan}, gpu_id={gpu_id}, model_id={model_id}")
        self.model = self.load_model()
        
    def queue_image(self, img, output_path):
        self.queue.put((img, output_path))
        logger.debug(f"Image queued for upscaling: {output_path}")
    
    def load_model(self):
        model = ModelLoader().load_from_file(MODEL_PATH)
        assert isinstance(model, ImageModelDescriptor)
        model.cuda().eval()
        
        return model
    def _process_queue(self):
        batch = []
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                # Try to get images from the queue until the batch is full
                while len(batch) < self.batch_size:
                    img, output_path = self.queue.get(timeout=1)
                    batch.append((img, output_path))
                    logger.debug(f"Retrieved image from queue for batching: {output_path}")
            except queue.Empty:
                # If the queue is empty, process the current batch if it has images
                if batch:
                    self._process_batch(batch)
                    batch = []
                continue

            # Process the batch when the batch size is reached
            if batch:
                self._process_batch(batch)
                batch = []

        # Process any remaining images after stop is requested
        if batch:
            self._process_batch(batch)

        logger.info("UpscaleAPI processing thread has stopped.")

    def _process_batch(self, batch):
        imgs = [item[0] for item in batch]
        output_paths = [item[1] for item in batch]
        logger.info(f"Processing batch of {len(batch)} images.")

        try:
            # Perform upscaling on the batch of images
            upscaled_images = self._upscale_images(imgs)

            for img, output_path in zip(upscaled_images, output_paths):
                # Save the upscaled image
                cv2.imwrite(f"{os.path.splitext(output_path)[0]}.png", img)
                logger.debug(f"Upscaled image saved: {output_path}")
        except Exception as e:
            logger.error(f"Error during batch upscaling: {e}")

    def _upscale_images(self, imgs):
        logger.debug(f"Upscaling {len(imgs)} images.")
        tensors = [self.cv2_to_tensor(img) for img in imgs]
        batch_tensor = torch.cat(tensors, dim=0)  
        with torch.no_grad():
            output_batch = self.model(batch_tensor)
            upscaled_images = [self.tensor_to_cv2(tensor.squeeze(0)) for tensor in output_batch]
        return upscaled_images

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        logger.info("UpscaleAPI processing thread has been stopped.")
        
    @staticmethod
    def cv2_to_tensor(img: np.ndarray) -> torch.Tensor:
        assert img.ndim == 3, f"Expected img to have 3 dimensions, but got {img.ndim} in cv2_to_tensor"        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        img_float = img_rgb.astype(np.float32) / 255.0
        # Convert to tensor and add batch dimension, final shape is [1, 3, H, W]
        return torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).cuda()

    @staticmethod
    def tensor_to_cv2(tensor: torch.Tensor) -> np.ndarray:
        #input shape is [C, H, W]
        assert tensor.dim() == 3, f"Expected tensor to have 3 dimensions, but got {tensor.dim()} in tensor_to_cv2"
        assert tensor.shape[0] == 3, f"Expected tensor to have 3 channels, but got {tensor.shape[0]} in tensor_to_cv2"

        img = tensor.mul(255).byte().cpu().numpy()  # Convert to [0, 255] and NumPy
        img = img.transpose((1, 2, 0))  # Convert to [H, W, C]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV
        #final shape is [H, W, C]
        return img