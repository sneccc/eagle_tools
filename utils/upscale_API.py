import threading
import queue
import os
import logging
import cv2
import torch
import numpy as np
from spandrel import ImageModelDescriptor, ModelLoader
import utils.color_utils as color_utils
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
            # Group images by their dimensions
            size_to_images = {}
            for img, output_path in zip(imgs, output_paths):
                size = img.shape[:2]
                if size not in size_to_images:
                    size_to_images[size] = []
                size_to_images[size].append((img, output_path))

            # Process each group of images with the same dimensions
            for size, batch in size_to_images.items():
                batch_imgs = [item[0] for item in batch]
                batch_output_paths = [item[1] for item in batch]

                # Perform upscaling on the batch of images
                upscaled_images = self._upscale_images(batch_imgs)

                for img, output_path in zip(upscaled_images, batch_output_paths):
                    # Save to RGB
                    img = color_utils.convert_color(img, to='RGB')
                    cv2.imwrite(f"{os.path.splitext(output_path)[0]}.png", img)
                    logger.debug(f"Upscaled image saved: {output_path}")
        except Exception as e:
            logger.error(f"Error during batch upscaling: {e}")

    def _upscale_images(self, imgs):
        logger.debug(f"Upscaling {len(imgs)} images.")
        processed_imgs = []
        for img in imgs:
            h, w = img.shape[:2]
            if h > 1024 or w > 1024:
                scale = 1024 / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            processed_imgs.append(img)
        
        tensors = [color_utils.cv2_to_tensor(img) for img in processed_imgs]
        batch_tensor = torch.cat(tensors, dim=0)  
        with torch.no_grad():
            output_batch = self.model(batch_tensor)
            upscaled_images = [color_utils.tensor_to_cv2(tensor.squeeze(0)) for tensor in output_batch]
        return upscaled_images

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        logger.info("UpscaleAPI processing thread has been stopped.")

    def cv2_to_tensor(self, img: np.ndarray) -> torch.Tensor:
        return color_utils.cv2_to_tensor(img)

    def tensor_to_cv2(self, tensor: torch.Tensor) -> np.ndarray:
        return color_utils.tensor_to_cv2(tensor)