import os
from PIL import Image, ImageFile
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import utils.config as config
import utils.image_utils as image_utils


Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ProcessImageAPI:
    def __init__(self):
        self.llm_processor = None
        self.rename = config.rename
        if config.use_LLM:
            from .LLM_API import AsyncLLMProcessor
            self.llm_processor = AsyncLLMProcessor(
                add_tags=config.add_tags,
                shuffle_content=config.shuffle_content
            )
    
    def process_image_batch(self, images_batch, folder_path, basefolder, image_paths_batch, counter, lock, augment=None):
        self.process_images(images_batch, folder_path, basefolder, image_paths_batch, counter, lock, rename=self.rename, augment=augment)
        
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
            import utils.image_utils as image_utils
            img = image_utils.svg_scaling(image_path, 1024, output_path, config.do_center_square_crop, 0.0)
            img.save(f"{os.path.splitext(output_path)[0]}.png", format='PNG', quality=98)
        except Exception as e:
            print(f"Error resizing SVG image: {e}")

    def _process_raster(self, image_path, output_path):
        try:
            import utils.image_utils as image_utils
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
                img = image_utils.fill_transparent_with_color_cv2(img,config.padding)

            # Center square crop (if applicable)
            if config.do_center_square_crop:
                size = min(h, w)
                start_y = (h - size) // 2
                start_x = (w - size) // 2
                img = img[start_y:start_y+size, start_x:start_x+size]
                
                
            if config.doUpscale:
                import utils.image_utils as image_utils
                img = image_utils.upscale_image_cv2(img, output_path, config.pixelart, config.useEsrganModel, config.gpuid,model_id=3)
            
            # Resize and crop to fit (if applicable)
            if config.doBucketing:
                target_resolutions = config.target_resolutions
                if target_resolutions:
                    import utils.image_utils as image_utils
                    img = image_utils.resize_and_crop_to_fit_cv2(img, target_resolutions)
            else:
                img = image_utils.upscale_to_1024(img)
            
            # Final check on image dimensions
            h, w = img.shape[:2]
            if h < min_dimension or w < min_dimension:
                raise ValueError(f"Resulting image dimensions too small: {w}x{h}")

            # Save the image
            cv2.imwrite(f"{os.path.splitext(output_path)[0]}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error processing image {image_path}:")
            print(f"Exception: {e}")
            print("Traceback:")
            print(error_trace)
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

