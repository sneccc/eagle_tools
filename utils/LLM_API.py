import os
import asyncio
from asyncio import Queue
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm.asyncio import tqdm
import logging



# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.random.manual_seed(0)

class AsyncLLMProcessor:
    def __init__(self, cpu_batch_size=4, add_tags=True, shuffle_content=False, max_queue_size=1000):
        self.queue = Queue(maxsize=max_queue_size)
        self.cpu_batch_size = cpu_batch_size
        self.add_tags = add_tags
        self.shuffle_content = shuffle_content
        self.pipe = self.load_llm_model()
        self.stop_signal = False
        self.progress_bar = None
        self.loop = asyncio.get_event_loop()
        self.processing_task = self.loop.create_task(self.process_queue())
    
    @staticmethod
    def load_llm_model():
        logger.info("Loading LLM model...")
        model = AutoModelForCausalLM.from_pretrained( 
            "microsoft/Phi-3-mini-4k-instruct",  
            device_map="auto",  # Let transformers decide the best device
            torch_dtype="auto",  
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        logger.info("LLM model loaded successfully.")
        return pipe

    def process_queue(self):
        while not self.stop_signal or not self.queue.empty():
            batch = []
            try:
                for _ in range(self.batch_size):
                    item = self.queue.get_nowait()
                    batch.append(item)
                    if self.queue.empty():
                        break
            except queue.Empty:
                if not batch:
                    continue

            futures = []
            for img, output_path in batch:
                future = self.executor.submit(self._process_single_image, img, output_path)
                futures.append(future)
            
            for future in futures:
                future.result()

        self.processing_task = None

    async def process_batch(self, batch):
        import caption_utils
        # Prepare captions
        captions = [
            caption_utils.prepare_content(
                item['annotation'], 
                item['tags'], 
                self.add_tags, 
                self.shuffle_content, 
                item.get('augment')
            ) 
            for item in batch
        ]

        # Generate captions using LLM
        results = await self.use_llm_batch(captions)

        # Write results to files asynchronously
        tasks = []
        for item, result in zip(batch, results):
            tags_file_path = os.path.join(item['folder_path'], f"{item['output_path']}.txt")
            os.makedirs(os.path.dirname(tags_file_path), exist_ok=True)
            tasks.append(self.loop.run_in_executor(None, caption_utils.write_tags_file, tags_file_path, [result]))
        
        await asyncio.gather(*tasks)

    async def use_llm_batch(self, captions):
        # Prepare prompts for each caption
        prompts = [
            "You are a helpful AI assistant. Your goal is to provide good captions based on the text given, if possible very detailed with respect to the tags.\n"
            f"Text to be captioned: {caption}"
            for caption in captions
        ]

        generation_args = {
            "max_new_tokens": 150,  # Adjust as needed per caption
            "return_full_text": False,
            "temperature": 0.4,
            "do_sample": False,
            "num_return_sequences": 1,
        }

        # Run the pipeline in an executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(None, lambda: self.pipe(prompts, **generation_args))

        # Extract generated texts
        generated_texts = [output['generated_text'].strip().strip('"').strip("'") for output in outputs]
        return generated_texts

    async def add_to_queue(self, folder_path, output_path, annotation, tags, augment=None):
        await self.queue.put({
            'folder_path': folder_path, 
            'output_path': output_path, 
            'annotation': annotation,
            'tags': tags,
            'augment': augment
        })

    async def stop(self):
        self.stop_signal = True
        await self.queue.put(None)  # Ensure the queue can terminate
        await self.processing_task
