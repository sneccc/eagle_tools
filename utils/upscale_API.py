import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import asyncio
from asyncio import Queue, QueueFull  # Import QueueFull
import utils.image_utils as image_utils
import utils.config as config
import os
from realesrgan_ncnn_py import Realesrgan
import queue
from tqdm import tqdm
import logging
import threading

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
                if batch:
                    futures = [self.executor.submit(self._process_single_image, img, output_path) for img, output_path in batch]
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
                img, output_path, pixelart=False,
                useEsrganModel=self.use_esrgan,
                gpuid=self.gpu_id, model_id=self.model_id
            )
            
            # Resize and crop to fit (if applicable)
            if config.doBucketing:
                target_resolutions = config.target_resolutions
                if target_resolutions:
                    upscaled_img = image_utils.resize_and_crop_to_fit_cv2(upscaled_img, target_resolutions)
            else:
                upscaled_img = image_utils.upscale_to_1024(upscaled_img)
            
            # Save the image
            cv2.imwrite(f"{os.path.splitext(output_path)[0]}.png", cv2.cvtColor(upscaled_img, cv2.COLOR_RGB2BGR))
            return upscaled_img

        except Exception as e:
            logger.error(f"Error upscaling image {output_path}: {e}")
            logger.info("Falling back to non-upscaled processing.")
            return None
            
    @staticmethod
    def upscale_image_cv2(img, output_path, pixelart=False, useEsrganModel=False, gpuid=0, model_id=4):
        """
        Upscale an image based on given parameters and save it.
        
        :param img: OpenCV image (numpy array)
        :param output_path: Path to save the upscaled image
        :param pixelart: Boolean, if True, use nearest neighbor upscaling
        :param useEsrganModel: Boolean, if True, use Realesrgan upscaling
        :param gpuid: Integer, GPU ID to use for Realesrgan
        :return: Upscaled OpenCV image (numpy array)
        """
        min_dimension = min(img.shape[0], img.shape[1])
        
        if min_dimension > 1024:
            return image_utils.upscale_to_1024(img)
        
        if pixelart:
            upscaled_img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
        elif useEsrganModel:
            realesrgan = Realesrgan(model=model_id, gpuid=gpuid)
            upscaled_img = realesrgan.process_cv2(img)
        else:
            upscaled_img = img
        return upscaled_img