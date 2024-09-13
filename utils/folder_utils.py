import os
from io_utils import ensure_directory_exists
from joblib import Parallel, delayed
from tqdm import tqdm
from process_image_API import ProcessImageAPI
from config import pixelart,append_filename_to_captions,doBucketing,isEsganUpscale,usePilSave,number_of_jobs,do_center_square_crop,padding,add_tags,shuffle_content,use_LLM,target_resolutions,augment_list
from image_utils import process_image_batch

def make_folders_recursively(root, folder, images, basefolder, processed_dir, counter, lock, augment=None):
    safe_folder_name = folder["name"].replace("/", "_")
    current_path = os.path.join(root, safe_folder_name)
    ensure_directory_exists(current_path)

    images_in_folder = [image for image in images if folder["id"] in image["folders"]]

    batch_size = 10  # Adjust batch size as needed
    image_processing_tasks = []
    api_instance = ProcessImageAPI({
        'pixelart': pixelart,
        'append_filename_to_captions': append_filename_to_captions,
        'doBucketing': doBucketing,
        'isEsganUpscale': isEsganUpscale,
        'usePilSave': usePilSave,
        'do_center_square_crop': do_center_square_crop,
        'padding': padding,
        'add_tags': add_tags,
        'shuffle_content': shuffle_content,
        'use_LLM': use_LLM,
        'target_resolutions': target_resolutions
    })

    for i in range(0, len(images_in_folder), batch_size):
        images_batch = images_in_folder[i:i + batch_size]
        image_paths_batch = [
            os.path.join(processed_dir, f"{image['id']}.info", f"{image['name']}.{image['ext']}") for image in images_batch
        ]

        # Add the task to process the batch
        image_processing_tasks.append(
            delayed(process_image_batch)(api_instance, images_batch, current_path, basefolder, image_paths_batch, counter, lock, augment)
        )

    if image_processing_tasks:
        Parallel(n_jobs=number_of_jobs)(
            tqdm(image_processing_tasks, desc="Processing image batches")
        )

    for child in folder.get("children", []):
        make_folders_recursively(current_path, child, images, basefolder, processed_dir, counter, lock, augment=augment)

    api_instance.close()  # Ensure any resources are properly released