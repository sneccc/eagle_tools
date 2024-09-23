import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # Use the CLI version of tqdm
import utils.config as config
import utils.process_image_API as process_image_API
import utils.io_utils as io_utils  # Import the normalize_and_rename_files function
import logging

logger = logging.getLogger(__name__)

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def make_folders_recursively(root, folder, images, basefolder, processed_dir, counter, lock, augment=None):
    safe_folder_name = folder["name"].replace("/", "_")
    current_path = os.path.join(root, safe_folder_name)
    ensure_directory_exists(current_path)

    images_in_folder = [image for image in images if folder["id"] in image["folders"]]

    cpu_batch_size = config.cpu_batch_size
    api_instance = process_image_API.ProcessImageAPI()


    # Create a global progress bar for the queue
    with tqdm(total=len(images_in_folder), desc="Images Queued", position=0, dynamic_ncols=True) as queue_pbar:
        with ThreadPoolExecutor(max_workers=config.number_of_jobs) as executor:
            futures = []
            for i in range(0, len(images_in_folder), cpu_batch_size):
                images_batch = images_in_folder[i:i + cpu_batch_size]
                image_paths_batch = [
                    os.path.join(processed_dir, f"{image['id']}.info", f"{image['name']}.{image['ext']}") for image in images_batch
                ]
                futures.append(
                    executor.submit(
                        api_instance.process_image_batch,
                        images_batch,
                        current_path,
                        basefolder,
                        image_paths_batch,
                        counter,
                        lock,
                        augment=augment,
                        queue_pbar=queue_pbar
                    )
                )

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing image batch: {e}")

    # Process child folders recursively
    for child in folder.get("children", []):
        make_folders_recursively(current_path, child, images, basefolder, processed_dir, counter, lock, augment=augment)

    # Close the API instance to ensure all images are processed
    api_instance.close()