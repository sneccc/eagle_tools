import toml
import random
from LLM_API import AsyncLLMProcessor

# Load configuration
with open('config.toml', 'r') as f:
    config = toml.load(f)

# General settings
globals().update(config['general'])

# Input path
input_path = config['paths']['input_path']

# Handle special cases
if config['general']['padding'] == "random(80, 100)":
    padding = random.randint(80, 100)

# LLM settings
if config['general']['use_LLM']:
    llm_processor = AsyncLLMProcessor(**config['llm'])
    number_of_jobs = 1

# Augmentation settings
augment_list = config['augment']['list']

# Resolutions
target_resolutions = config['resolutions']['target_resolutions']



# augment_list = [    
# #   None,
# #    ('random_dropout_keep_4', None, 0.7),
# #    ('prepend', 'ohwx artstyle, ', 1),
#     ('prepend', 'ohwx artstyle, ', 1),
# #    ('append', ' ,regular_icon', 0.8)
#     # None,  # First pass: No augmentation
#     # ('tokenOnly', '@Bold_icon', 1),
#     # ('tokenOnly', '@bold_icon', 0.2),
#     # ('tokenOnly', 'Simple Icon', 0.2),
#     # ('tokenOnly', 'regular icon', 0.2),
#     # ('random_dropout_keep_1', None, 0.7),
#     # ('random_dropout_keep_2', None, 0.7),
#     # ('random_dropout_keep_3', None, 0.7),
#     # ('random_dropout_keep_4', None, 0.7),
#     # ('prepend', 'start_', 1),  # Prepend augmentation
#     # ('append', '_end', 1)      # Append augmentation
# ]