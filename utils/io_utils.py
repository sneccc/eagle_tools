import os,time,json,zipfile
from multiprocessing import Manager
from folder_utils import make_folders_recursively

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_EaglePack_and_process(eaglepacks_path,augment_list):
    processed_dir = os.path.join(eaglepacks_path, "processed")
    basefolder = os.path.join(eaglepacks_path, "base")
    
    #Unzip all the eaglepacks
    ensure_directory_exists(processed_dir)
    for filename in os.listdir(eaglepacks_path):
        if filename.endswith(".eaglepack") and zipfile.is_zipfile(os.path.join(eaglepacks_path, filename)):
            with zipfile.ZipFile(os.path.join(eaglepacks_path, filename), 'r') as zipObj:
                zipObj.extractall(processed_dir)

    #Json to folders
    json_file = os.path.join(processed_dir, "pack.json")
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        images = data["images"]
        folders = data["folder"]
        
        ensure_directory_exists(basefolder)
        with Manager() as manager:
            counter = manager.Value('i', 0)  # Shared counter for parallel processes
            lock = manager.Lock()  # Lock to prevent race conditions

            for augment in augment_list:
                start_time = time.time()  # Start timer
                if augment is None:
                    # No augmentation
                    make_folders_recursively(processed_dir, folders, images, basefolder, processed_dir, counter, lock)
                else:
                    aug_type, aug_value, aug_percentage = augment
                    # With augmentation
                    make_folders_recursively(
                        processed_dir, folders, images, basefolder, processed_dir, counter, lock,
                        augment=(aug_type, aug_value, aug_percentage)
                    )           
                end_time = time.time()  # End timer
                print(f"Time taken for augmentation {augment}: {end_time - start_time:.2f} seconds")
