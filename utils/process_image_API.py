import asyncio
import os
from PIL import Image, ImageFile
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
import utils.config as config
import utils.image_utils as image_utils
import utils.upscale_API as upscale_API
from tqdm import tqdm
from threading import Lock
import logging
import threading
import re
import hashlib

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ProcessImageAPI:
    def __init__(self):
        self.llm_processor = None
        self.rename_output_file = config.rename_output_file
        if config.use_LLM:
            from .LLM_API import AsyncLLMProcessor
            self.llm_processor = AsyncLLMProcessor(
                add_tags=config.add_tags,
                shuffle_content=config.shuffle_content
            )
        self.upscale_api = upscale_API.UpscaleAPI(
            batch_size=config.gpu_batch_size,
            use_esrgan=config.useEsrganModel,
            gpu_id=config.gpuid,
            model_id=3
        )
        self.progress_lock = threading.Lock()
        
    def normalize_and_rename_paths(self, image_paths_batch, folder_path):
        sanitized_paths = []
        for image_path in image_paths_batch:
            # Extract the filename and extension
            base_name, ext = os.path.splitext(os.path.basename(image_path))
            # Normalize the filename by removing weird characters
            sanitized_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', base_name)
            
            # Generate a unique hash for the filename
            hash_object = hashlib.sha256(sanitized_name.encode())
            unique_hash = hash_object.hexdigest()
            new_path = os.path.join(folder_path, unique_hash + ext)
            
            # Rename the file if the new path does not exist
            if not os.path.exists(new_path):
                os.rename(image_path, new_path)
            sanitized_paths.append(new_path)
        return sanitized_paths
    
    def process_image_batch(self, images_batch, folder_path, basefolder, image_paths_batch, counter, lock, augment=None, queue_pbar=None):
        # Normalize and rename paths
        #normalized_image_paths_batch = self.normalize_and_rename_paths(image_paths_batch, folder_path)
        normalized_image_paths_batch = image_paths_batch
        self.process_images(images_batch, folder_path, basefolder, normalized_image_paths_batch, counter, lock, rename_output_file=self.rename_output_file, augment=augment, queue_pbar=queue_pbar)
    
    def create_tags_file(self, annotation, tags, folder_path, output_path, augment=None):
        import utils.caption_utils as caption_utils
        import utils.folder_utils as folder_utils

        tags_file_path = os.path.join(folder_path, f"{output_path}.txt")
        folder_utils.ensure_directory_exists(os.path.dirname(tags_file_path))

        if os.path.exists(tags_file_path):
            return

        if self.llm_processor:
            self.llm_processor.add_to_queue(folder_path, output_path, annotation, tags, augment)
        else:
            content = caption_utils.prepare_content(annotation, tags, config.add_tags,
                                   config.shuffle_content, augment=augment)
            content = caption_utils.remove_duplicate_phrases(content)
            caption_utils.write_tags_file(output_path=tags_file_path, content_list=[content])
    
    def process_image(self, image, folder_path, basefolder, image_path, count, lock, augment=None, global_pbar=None):
        try:
            #transform image_path to absolute path
            image_path = os.path.abspath(image_path)
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                return
            
            # Lock the output path determination and existence check
            with lock:
                output_path = os.path.join(basefolder, f"img_{count:03d}" if self.rename_output_file else image['name'])
                #transform output_path to absolute path
                output_path = os.path.abspath(output_path)
                if os.path.exists(output_path):
                    logger.warning(f"Output path already exists: {output_path}")
                    return
            
            # Create tags file
            self.create_tags_file(image.get("annotation"), image.get("tags"), folder_path, output_path, augment)

            # Process image
            if image['ext'].lower() == "svg":
                logger.info(f"Processing SVG {image_path}")
                self._process_svg(image_path, output_path)
            else:
                self._process_raster(image_path, output_path)
        
            if global_pbar:
                with self.progress_lock:
                    global_pbar.update(1)
                    
        except Exception as e:
            logger.error(f"Error in processing image {image_path}: {e}")
    
    def _process_svg(self, image_path, output_path):
        try:
            img = image_utils.svg_scaling(image_path, 1024, output_path, config.do_center_square_crop, 0.0)
            
            img.save(f"{os.path.splitext(output_path)[0]}.png", format='PNG', quality=98)
            return img
        except Exception as e:
            logger.error(f"Error in processing SVG {image_path}: {e}")
            return None
    
    def _process_raster(self, image_path, output_path):
        try:
            # Convert to absolute path
            image_path = os.path.abspath(image_path)
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            h, w = img.shape[:2]
            min_dimension = 10
            if h < min_dimension or w < min_dimension:
                raise ValueError(f"Image dimensions too small: {w}x{h}")

            # Detect transparency and correct channel organization
            if len(img.shape) == 2:  # Grayscale image
                has_transparency = False
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif len(img.shape) == 3:
                if img.shape[2] == 3:  # RGB image
                    has_transparency = False
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif img.shape[2] == 4:  # RGBA image
                    has_transparency = self._detect_transparency(img)
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
                else:
                    raise ValueError(f"Unexpected number of channels: {img.shape[2]}")
            else:
                raise ValueError(f"Unexpected image shape: {img.shape}")
            
            # Handle transparent images
            if has_transparency:
                logger.info(f"Filling transparent image {image_path}")
                img = image_utils.fill_transparent_with_color_cv2(img, config.padding)
            else:
                logger.info(f"No transparency detected in image {image_path}")
            
            # Center square crop (if applicable)
            if config.do_center_square_crop:
                size = min(h, w)
                start_y = (h - size) // 2
                start_x = (w - size) // 2
                img = img[start_y:start_y+size, start_x:start_x+size]

            if config.doUpscale:
                try:
                    self.upscale_api.queue_image(img=img, output_path=output_path)
                except Exception as upscale_error:
                    logger.error(f"Error in upscaling image {image_path}: {upscale_error}")
                    logger.error("Falling back to non-upscaled processing.")
                    
                    img=self._process_without_upscale(img, output_path)
                    path = f"{os.path.splitext(output_path)[0]}.png"
                    cv2.imwrite(path, img)
            else:
                img=self._process_without_upscale(img, output_path)
                path = f"{os.path.splitext(output_path)[0]}.png"
                cv2.imwrite(path, img)
        except Exception as e:
            logger.error(f"Error processing raster image {image_path}: {e}")
            if 'img' in locals():
                logger.error(f"Image shape: {img.shape}")
            return None
    
    def _process_without_upscale(self, img, output_path):
        try:

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if config.doBucketing:
                target_resolutions = config.target_resolutions
                if target_resolutions:
                    img = image_utils.resize_and_crop_to_fit_cv2(img, target_resolutions,config.pixelart)
            else:
                img = np.expand_dims(img, axis=0)
                img = image_utils.upscale_to_1024(img,config.pixelart_target,config.pixelart)
                img = img.squeeze(axis=0)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
           

            
            return img
        except Exception as e:
            logger.error(f"Error in non-upscaled processing for image {output_path}: {e}, with shape: {img.shape}")
            return None
        


    def process_images(self, images, folder_path, basefolder, image_paths, counter, lock, rename_output_file=True, augment=None, queue_pbar=None):
        with ThreadPoolExecutor(max_workers=config.number_of_jobs) as executor:
            futures = []
            for image, image_path in zip(images, image_paths):
                with lock:
                    count = counter.value
                    counter.value += 1
                futures.append(
                    executor.submit(
                        self.process_image,
                        image,
                        folder_path,
                        basefolder,
                        image_path,
                        count,
                        lock,
                        augment,
                        queue_pbar
                    )
                )
        
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing image: {e}")
    
    def close(self):
        if self.llm_processor:
            self.llm_processor.stop()
        self.upscale_api.stop()

    def _detect_transparency(self, img: np.ndarray) -> bool:
        # Check if the image has an alpha channel
        if img.shape[2] == 4:
            # Extract the alpha channel
            alpha_channel = img[:, :, 3]
            
            # Check for unique values in the alpha channel
            unique_values = np.unique(alpha_channel)
            
            # Detect transparency efficiently
            has_transparency = len(unique_values) > 1 and np.min(unique_values) < 255
            return has_transparency
        return False

