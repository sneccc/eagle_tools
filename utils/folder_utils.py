import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # Use the CLI version of tqdm
import utils.config as config
import utils.process_image_API as process_image_API
import logging
import asyncio
import utils

logger = logging.getLogger(__name__)

def ensure_directory_exists(directory):
    os.makedirs(directory, exist_ok=True)

async def make_folders_recursively(root, folder, images, basefolder, processed_dir, counter, lock, augment=None):
    safe_folder_name = folder["name"].replace("/", "_")
    current_path = os.path.join(root, safe_folder_name)
    ensure_directory_exists(current_path)

    images_in_folder = [image for image in images if folder["id"] in image["folders"]]

    cpu_batch_size = config.cpu_batch_size
    api_instance = process_image_API.ProcessImageAPI()

    with tqdm(total=len(images_in_folder), desc="Total Images Processed", position=0, dynamic_ncols=True) as global_pbar:
        tasks = []
        for i in range(0, len(images_in_folder), cpu_batch_size):
            images_batch = images_in_folder[i:i + cpu_batch_size]
            image_paths_batch = [
                os.path.join(processed_dir, f"{image['id']}.info", f"{image['name']}.{image['ext']}") for image in images_batch
            ]
            tasks.append(api_instance.process_image_batch(
                images_batch,
                current_path,
                basefolder,
                image_paths_batch,
                counter,
                lock,
                augment=augment,
                global_pbar=global_pbar
            ))
        
        # Await all tasks concurrently
        await asyncio.gather(*tasks)

    for child in folder.get("children", []):
        await make_folders_recursively(current_path, child, images, basefolder, processed_dir, counter, lock, augment=augment)

    api_instance.close()