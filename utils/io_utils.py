import utils.config as config
import json
import os
import zipfile
from multiprocessing import Manager
import time
import asyncio
import logging
import utils.folder_utils as folder_utils

logger = logging.getLogger(__name__)

async def extract_EaglePack_and_process(eaglepacks_path):
    processed_dir = os.path.join(eaglepacks_path, "processed")
    basefolder = os.path.join(eaglepacks_path, "base")
    
    # Unzip all the eaglepacks
    folder_utils.ensure_directory_exists(processed_dir)
    for filename in os.listdir(eaglepacks_path):
        if filename.endswith(".eaglepack") and zipfile.is_zipfile(os.path.join(eaglepacks_path, filename)):
            with zipfile.ZipFile(os.path.join(eaglepacks_path, filename), 'r') as zipObj:
                zipObj.extractall(processed_dir)
    
    # Json to folders
    json_file = os.path.join(processed_dir, "pack.json")
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        images = data["images"]
        folders = data["folder"]
        
        folder_utils.ensure_directory_exists(basefolder)
        with Manager() as manager:
            counter = manager.Value('i', 0)  # Shared counter for parallel processes
            lock = manager.Lock()  # Lock to prevent race conditions

            for augment in config.augment_list:
                start_time = time.time()  # Start timer
                if augment is None or (isinstance(augment, dict) and augment.get('type') == "skip_augmentation"):
                    # No augmentation
                    await folder_utils.make_folders_recursively(
                        processed_dir, folders, images, basefolder, processed_dir, counter, lock, augment=augment
                    )
                else:
                    aug_type = augment['type'] if isinstance(augment, dict) else augment[0]
                    aug_value = augment['value'] if isinstance(augment, dict) else augment[1]
                    aug_probability = augment['probability'] if isinstance(augment, dict) else augment[2]
                    # With augmentation
                    await folder_utils.make_folders_recursively(
                        processed_dir, folders, images, basefolder, processed_dir, counter, lock,
                        augment=(aug_type, aug_value, aug_probability)
                    )           
                end_time = time.time()  # End timer
                logger.info(f"Time taken for augmentation {augment}: {end_time - start_time:.2f} seconds")

    # No need to manually initialize UpscaleAPI here as it's handled within ProcessImageAPI
    # The ProcessImageAPI instance manages its own UpscaleAPI
