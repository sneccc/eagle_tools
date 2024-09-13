import random
import re
import os

def prepare_content(annotation, tags, add_tags=True, shuffle_content=False, augment=None):
    content_list = []
    if annotation:
        content_list.extend([tag.strip() for tag in annotation.split(',')])
    if add_tags and tags:
        content_list.extend(tags)
    
    content_list = apply_augmentation(content_list, augment)
    
    if shuffle_content:
        random.shuffle(content_list)
    
    return ", ".join(content_list)

def apply_augmentation(content_list, augment):
    if not augment:
        return content_list

    aug_name, aug_value, aug_percentage = augment
    aug_percentage = float(aug_percentage)  # Ensure aug_percentage is a float
    if random.random() >= aug_percentage:
        return content_list

    if aug_name == "prepend":
        content_list.insert(0, aug_value)
    elif aug_name == "append":
        content_list.append(aug_value)
    elif aug_name == "tokenOnly":
        content_list = [aug_value]
    elif "random_dropout" in aug_name:
        match = re.search(r'random_dropout_keep_(\d+)', aug_name)
        if match:
            keep_n = int(match.group(1))
            content_list = content_list[:keep_n]

    return content_list

def remove_duplicate_phrases(text):

    phrases = [phrase.strip() for phrase in text.split(',')]
    seen_phrases = set()
    unique_phrases = []
    for phrase in phrases:
        normalized_phrase = phrase.lower().strip()
        if normalized_phrase not in seen_phrases and normalized_phrase:
            seen_phrases.add(normalized_phrase)
            unique_phrases.append(phrase)
    return ', '.join(unique_phrases)

def write_tags_file(output_path, content_list):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    content = remove_duplicate_phrases(", ".join(content_list))
    output_path_txt = f"{output_path}.txt" if not output_path.endswith('.txt') else output_path
    with open(output_path_txt, "w", encoding="utf-8") as f:
        f.write(content)

def create_tags_file(annotation, tags, folder_path, output_path, augment=None):
    import config,folder_utils

    output_extension = '.txt'
    tags_file_name = f"{output_path}{output_extension}"
    tags_file_path = os.path.join(folder_path, tags_file_name)
    folder_utils.ensure_directory_exists(os.path.dirname(tags_file_path))

    if os.path.exists(tags_file_path):
        return
    
    if config.use_LLM:
        config.llm_processor.add_to_queue(folder_path, output_path, annotation, tags, augment)
    else:
        content = prepare_content(annotation, tags, config.add_tags, config.shuffle_content, augment=augment)
        content = remove_duplicate_phrases(content)
        write_tags_file(output_path=tags_file_path, content_list=[content])