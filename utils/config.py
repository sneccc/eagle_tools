import toml
import random
from LLM_API import AsyncLLMProcessor

# Load configuration
with open('params.toml', 'r') as f:
    config = toml.load(f)

# Parallel processing settings
number_of_jobs = config['parallel_processing']['number_of_jobs']

# Image processing settings
globals().update(config['image_processing'])

# Handle padding
padding = config['image_processing']['padding']
if isinstance(padding, str) and padding.startswith("random"):
    min_val, max_val = map(int, padding.strip("random()").split(","))
    padding = random.randint(min_val, max_val)
elif isinstance(padding, int):
    padding = padding
else:
    raise ValueError("Invalid padding value in params.toml")

# Caption processing settings
globals().update(config['caption_processing'])

# Input path
input_path = config['paths']['input_path']

# LLM settings
if config['llm']['use_LLM']:
    llm_processor = AsyncLLMProcessor(**config['llm'])
    number_of_jobs = 1

# Augmentation settings
augment_list = config['augment']['list']

# Resolutions
target_resolutions = config['resolutions']['target_resolutions']