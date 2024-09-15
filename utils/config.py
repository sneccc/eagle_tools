import random
import toml
import os
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = toml.load(f)
    
    # Convert augment probabilities to float
    if 'augment' in config and 'list' in config['augment']:
        for i, aug in enumerate(config['augment']['list']):
            if isinstance(aug, dict) and 'probability' in aug:
                aug['probability'] = float(aug['probability'])
                config['augment']['list'][i] = (aug['type'], aug['value'], aug['probability'])
    
    return config

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), '..', 'params.toml')
config = load_config(config_path)

# Parallel processing settings
number_of_jobs = config['parallel_processing']['number_of_jobs']
cpu_batch_size = config['parallel_processing']['cpu_batch_size']
gpu_batch_size = config['parallel_processing']['gpu_batch_size']
max_queue_size = config['parallel_processing']['max_queue_size']

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

# Spandrel settings
full_spandrel_model_path = config['paths']['spandrel_model_path']
spandrel_model_path = os.path.join(os.path.dirname(__file__), '..', full_spandrel_model_path)

# LLM settings
if config['llm']['use_LLM']:
    import LLM_API
    print("Using LLM")
    use_LLM = True
    llm_processor = LLM_API.AsyncLLMProcessor(**config['llm'])
    number_of_jobs = 1
else:
    print("Not using LLM")
    use_LLM = False
    llm_processor = None

# Augmentation settings
augment_list = config['augment']['list']

# Resolutions
target_resolutions = config['resolutions']['target_resolutions']


gpuid = config['image_processing']['gpuid']