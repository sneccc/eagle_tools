import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import torch
import asyncio
from asyncio import Queue, QueueFull  # Import QueueFull
import utils.image_utils as image_utils
import utils.config as config
import os
import queue
from tqdm import tqdm
import logging
import threading
from spandrel import ImageModelDescriptor, ModelLoader

MODEL_PATH = config.spandrel_model_path

logger = logging.getLogger(__name__)

class UpscaleAPI:
    def __init__(self, batch_size=4, use_esrgan=True, gpu_id=0, model_id=3, max_queue_size=1000):
        self.batch_size = batch_size
        self.use_esrgan = use_esrgan
        self.gpu_id = gpu_id
        self.model_id = model_id
        self.max_queue_size = max_queue_size
        self.stop_signal = False
        self.queue = asyncio.Queue(maxsize=self.max_queue_size)
        self.executor = ThreadPoolExecutor()
        
        # Start the event loop in a separate thread
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self.loop.run_forever)
        self.loop_thread.start()
        
        # Schedule the processing task on the loop
        asyncio.run_coroutine_threadsafe(self.process_queue(), self.loop)

        self.upscale_progress = tqdm(total=0, desc="Upscaling Images", position=1, dynamic_ncols=True)
        self.gpu_queue_pbar = tqdm(total=self.max_queue_size, desc="GPU Queue", position=2, leave=False, dynamic_ncols=True)
        self.progress_lock = threading.Lock()

        #spandrel
        self.model = self.load_model()
        
    def load_model(self):
        model = ModelLoader().load_from_file(MODEL_PATH)
        assert isinstance(model, ImageModelDescriptor)
        model.cuda().eval()
        
        return model

    def queue_image(self, img, output_path):
        try:
            future = asyncio.run_coroutine_threadsafe(self.queue.put((img, output_path)), self.loop)
            future.result()  # Wait until the item is enqueued
            with self.progress_lock:
                self.gpu_queue_pbar.update(1)
        except Exception as e:
            logger.error(f"Error enqueuing image {output_path}: {e}")

    async def process_queue(self):
        while not self.stop_signal or not self.queue.empty():
            batch = []
            try:
                for _ in range(self.batch_size):
                    item = await self.queue.get()
                    if item is None:
                        break
                    batch.append(item)
                    with self.progress_lock:
                        self.gpu_queue_pbar.update(-1)
                if len(batch) > 0:
                    futures = [self.executor.submit(self.process_batch_images, [img], [output_path]) for img, output_path in batch]
                    for future in futures:
                        try:
                            await asyncio.wrap_future(future)
                            with self.progress_lock:
                                self.upscale_progress.update(1)
                        except Exception as e:
                            logger.error(f"Error in future processing: {e}")
                
            except Exception as e:
                logger.error(f"Error in process_queue: {e}")
                continue
        with self.progress_lock:
            self.upscale_progress.close()
            self.gpu_queue_pbar.close()

    def stop(self):
        self.stop_signal = True
        # Put a sentinel value to unblock any waiting coroutines
        asyncio.run_coroutine_threadsafe(self.queue.put(None), self.loop).result()
        # Stop the event loop
        self.loop.call_soon_threadsafe(self.loop.stop)
        # Wait for the loop thread to finish
        self.loop_thread.join()
        # Shutdown the executor
        self.executor.shutdown()

    def _process_single_image(self, img, output_path):
        try:
            upscaled_img = self.upscale_image_cv2(
                img, self.model, pixelart=False,
                useEsrganModel=self.use_esrgan,
            )
            
            # Resize and crop to fit (if applicable)
            if config.doBucketing:
                target_resolutions = config.target_resolutions
                if target_resolutions:
                    upscaled_img = image_utils.resize_and_crop_to_fit_cv2(upscaled_img, target_resolutions)
            else:
                upscaled_img = image_utils.upscale_to_1024(upscaled_img)


            # Save the image
            cv2.imwrite(f"{os.path.splitext(output_path)[0]}.png", upscaled_img)
            return upscaled_img

        except Exception as e:
            logger.error(f"Error upscaling image {output_path}: {e}")
            logger.info("Falling back to non-upscaled processing.")
            return None
            
    @staticmethod
    def upscale_image_cv2(img: np.ndarray, _model: torch.nn.Module, pixelart: bool = False, useEsrganModel: bool = False) -> np.ndarray:
        """
        Upscales an image using either OpenCV or an ESRGAN model.

        Parameters:
        - img: Input image as a NumPy array.
        - _model: Pre-trained ESRGAN model.
        - pixelart: Boolean indicating if the image is pixel art.
        - useEsrganModel: Boolean indicating if ESRGAN should be used.

        Returns:
        - Upscaled image as a NumPy array.
        """
        if img.ndim != 3:
            raise ValueError(f"Expected image with 3 dimensions, but got {img.ndim}")
        
        if useEsrganModel:
            img_tensor = UpscaleAPI.cv2_to_tensor(img)  # Shape: [1, 3, H, W]
            with torch.no_grad():
                upscaled_tensor = _model(img_tensor)  # Model expects [Batch, C, H, W]
            upscaled_img = UpscaleAPI.tensor_to_cv2(upscaled_tensor.squeeze(0))  # Shape: [H_new, W_new, C]
        else:
            # Example: Simple 2x upscale using OpenCV
            upscaled_img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
        
        logger.debug(f"Upscaled image shape: {upscaled_img.shape}.")
        return upscaled_img

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

    def process_batch_images(self, images: list, output_paths: list):
        """
        Processes a batch of images: upscales, resizes/crops, and saves them.
        
        Parameters:
        - images: List of input images as NumPy arrays.
        - output_paths: List of paths where the upscaled images will be saved.
        
        Returns:
        - List of upscaled images as NumPy arrays.
        """
        if len(images) != len(output_paths):
            raise ValueError("The number of images must match the number of output paths.")
        
        try:
            # Convert images to tensors and create a batch
            tensors = [self.cv2_to_tensor(img) for img in images]
            batch_tensor = torch.cat(tensors, dim=0)  # Shape: [Batch, Channels, Height, Width]
            logger.debug(f"Batch tensor shape: {batch_tensor.shape}")
            
            # Process the batch
            with torch.no_grad():
                output_batch = self.model(batch_tensor)  # Shape: [Batch, Channels, Height, Width]
            
            # Convert tensors back to images
            upscaled_images = [self.tensor_to_cv2(tensor.squeeze(0)) for tensor in output_batch]
            logger.debug(f"Output batch tensor shape: {output_batch.shape}")
            
            # Resize and crop each image
            if config.doBucketing:
                target_resolutions = config.target_resolutions
                if target_resolutions:
                    upscaled_images = [image_utils.resize_and_crop_to_fit_cv2(img, target_resolutions) for img in upscaled_images]
            else:
                upscaled_images = [image_utils.upscale_to_1024(img) for img in upscaled_images]
            
            # Save all images
            for img, output_path in zip(upscaled_images, output_paths):
                cv2.imwrite(f"{os.path.splitext(output_path)[0]}.png", img)
                logger.info(f"Saved upscaled image to: {output_path}.png")
            
            return upscaled_images

        except Exception as e:
            logger.error(f"Error processing batch images: {e}")
            return [None] * len(images)
