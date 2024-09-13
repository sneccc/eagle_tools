import os
from joblib import Parallel, delayed
from tqdm import tqdm
import utils.config as config
import utils.process_image_API as process_image_API

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
def make_folders_recursively(root, folder, images, basefolder, processed_dir, counter, lock, augment=None):
    safe_folder_name = folder["name"].replace("/", "_")
    current_path = os.path.join(root, safe_folder_name)
    ensure_directory_exists(current_path)

    images_in_folder = [image for image in images if folder["id"] in image["folders"]]

    batch_size = 10  # Adjust batch size as needed
    image_processing_tasks = []
    api_instance = process_image_API.ProcessImageAPI({
        'pixelart': config.pixelart,
        'append_filename_to_captions': config.append_filename_to_captions,
        'doBucketing': config.doBucketing,
        'isEsganUpscale': config.isEsganUpscale,
        'usePilSave': config.usePilSave,
        'do_center_square_crop': config.do_center_square_crop,
        'padding': config.padding,
        'add_tags': config.add_tags,
        'shuffle_content': config.shuffle_content,
        'use_LLM': config.use_LLM,
        'target_resolutions': config.target_resolutions,
        'rename': config.rename
    })

    for i in range(0, len(images_in_folder), batch_size):
        images_batch = images_in_folder[i:i + batch_size]
        image_paths_batch = [
            os.path.join(processed_dir, f"{image['id']}.info", f"{image['name']}.{image['ext']}") for image in images_batch
        ]

        # Add the task to process the batch
        image_processing_tasks.append(
            delayed(api_instance.process_image_batch)(images_batch, current_path, basefolder, image_paths_batch, counter, lock, augment)
        )

    if image_processing_tasks:
        Parallel(n_jobs=config.number_of_jobs)(
            tqdm(image_processing_tasks, desc="Processing image batches")
        )

    for child in folder.get("children", []):
        make_folders_recursively(current_path, child, images, basefolder, processed_dir, counter, lock, augment=augment)

    api_instance.close()  # Ensure any resources are properly released