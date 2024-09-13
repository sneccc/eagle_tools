import os
from PIL import Image, ImageFile
from concurrent.futures import ThreadPoolExecutor
from .caption_utils import write_tags_file, prepare_content, remove_duplicate_phrases
from .image_utils import resize_and_crop_to_fit, fill_transparent_with_color, center_crop_square, svg_scaling
from .io_utils import ensure_directory_exists
import cv2
import numpy as np
from .image_utils import resize_and_crop_to_fit_cv2

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ProcessImageAPI:
    def __init__(self, config):
        self.config = config
        self.llm_processor = None
        if config.get('use_LLM', False):
            from .LLM_API import AsyncLLMProcessor
            self.llm_processor = AsyncLLMProcessor(
                add_tags=config.get('add_tags', False),
                shuffle_content=config.get('shuffle_content', False)
            )

    def create_tags_file(self, annotation, tags, folder_path, output_path, augment=None):
        tags_file_path = os.path.join(folder_path, f"{output_path}.txt")
        ensure_directory_exists(os.path.dirname(tags_file_path))

        if os.path.exists(tags_file_path):
            return

        if self.llm_processor:
            self.llm_processor.add_to_queue(folder_path, output_path, annotation, tags, augment)
        else:
            content = prepare_content(annotation, tags, self.config.get('add_tags', False),
                                      self.config.get('shuffle_content', False), augment=augment)
            content = remove_duplicate_phrases(content)
            write_tags_file(output_path=tags_file_path, content_list=[content])

    def process_image(self, image, folder_path, basefolder, image_path, count, rename=True, augment=None):
        try:
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                return

            output_path = os.path.join(basefolder, f"img_{count:03d}" if rename else image['name'])
            self.create_tags_file(image.get("annotation"), image.get("tags"), folder_path, output_path, augment)

            if image['ext'] == "svg":
                self._process_svg(image_path, output_path)
            else:
                self._process_raster(image_path, output_path)

        except Exception as e:
            print(f"Error in processing image {image_path}: {e}")

    def _process_svg(self, image_path, output_path):
        try:
            img = svg_scaling(image_path, 1024, output_path, self.config.get('do_center_square_crop', False), 0.0)
            img.save(f"{os.path.splitext(output_path)[0]}.png", format='PNG', quality=98)
        except Exception as e:
            print(f"Error resizing SVG image: {e}")

    def _process_raster(self, image_path, output_path):
        try:
            # Read image with OpenCV
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")

            # Check image dimensions
            h, w = img.shape[:2]
            min_dimension = 10  # Set a minimum acceptable dimension
            if h < min_dimension or w < min_dimension:
                raise ValueError(f"Image dimensions too small: {w}x{h}")

            # Check and correct channel organization
            if len(img.shape) == 2:  # Grayscale image
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 3:  # Color image without alpha channel
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif img.shape[2] == 4:  # Image with alpha channel
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            else:
                raise ValueError(f"Unexpected number of channels: {img.shape[2]}")

            # Handle RGBA images
            if img.shape[2] == 4:
                # Check if alpha channel is valid
                if np.all(img[:,:,3] == 0):
                    # If alpha is all zero, treat as RGB
                    img = img[:,:,:3]
                else:
                    # Fill transparent with color
                    alpha = img[:,:,3]
                    rgb = img[:,:,:3]
                    background = np.full_like(rgb, self.config.get('padding', 90))
                    mask = alpha[:,:,np.newaxis] / 255.0
                    img = (rgb * mask + background * (1 - mask)).astype(np.uint8)

            # Center square crop (if applicable)
            if self.config.get('do_center_square_crop', False):
                size = min(h, w)
                start_y = (h - size) // 2
                start_x = (w - size) // 2
                img = img[start_y:start_y+size, start_x:start_x+size]

            # Resize and crop to fit (if applicable)
            if self.config.get('doBucketing', True):
                target_resolutions = self.config.get('target_resolutions', [])
                if target_resolutions:
                    img = resize_and_crop_to_fit_cv2(img, target_resolutions)

            # Final check on image dimensions
            h, w = img.shape[:2]
            if h < min_dimension or w < min_dimension:
                raise ValueError(f"Resulting image dimensions too small: {w}x{h}")

            # Save the image
            cv2.imwrite(f"{os.path.splitext(output_path)[0]}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            # Optionally, you could log this error or handle it in some other way

    def process_images(self, images, folder_path, basefolder, image_paths, counter, lock, rename=True, augment=None):
        with ThreadPoolExecutor() as executor:
            futures = []
            for image, image_path in zip(images, image_paths):
                with lock:
                    count = counter.value
                    counter.value += 1
                futures.append(executor.submit(self.process_image, image, folder_path, basefolder, image_path, count, rename, augment))
            
            for future in futures:
                future.result()  # This will raise any exceptions that occurred during processing

    def close(self):
        if self.llm_processor:
            self.llm_processor.stop()
