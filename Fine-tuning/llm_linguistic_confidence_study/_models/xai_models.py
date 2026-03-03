from openai import OpenAI
from omegaconf import DictConfig
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from datetime import datetime

OPENAI_SYSTEM_PROMPT = "You are a helpful assistant."

class Grok():
    def __init__(self, model_cfg: DictConfig):
        self.model_cfg = model_cfg

    def __call__(self, prompts: list[str], task_name: str, retrieval_id: str = None) -> list[str]:
        if retrieval_id is None:
            retrieval_id = f"grok_{task_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}"
            task_file_path = f'llm_linguistic_confidence_study/batch_tasks/{retrieval_id}.npy'
            print("Grok Retrieval ID:", retrieval_id)
            responses = self.real_time_inference(prompts, retrieval_id)
            responses = np.array(responses, dtype=object)  # convert to numpy array
            np.save(task_file_path, responses)  # save to disk
            responses = responses.tolist()  # convert back to list for return
        else:
            task_file_path = f'llm_linguistic_confidence_study/batch_tasks/{retrieval_id}.npy'
            responses = np.load(task_file_path, allow_pickle=True)
            responses = list(responses)
        return responses

    
    def real_time_inference(self, prompts: list[str], id) -> list[str]:
        client = OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")

        def fetch_response(prompt: str) -> str:
            while True:
                try:
                    response = client.chat.completions.create(
                        model=self.model_cfg.name,
                        messages=[
                            {"role": "system", "content": OPENAI_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                    )
                    print(id, response.choices[0].message.content.strip())
                    return response.choices[0].message.content.strip()
                except Exception as e:
                    print("Error:", e, "Retrying in 10 seconds...")
                    time.sleep(10)

        responses = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=15) as executor:
            future_to_idx = {
                executor.submit(fetch_response, prompt): idx
                for idx, prompt in enumerate(prompts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    responses[idx] = future.result()
                except Exception as e:
                    print(f"Prompt at index {idx} failed with error: {e}")
                    responses[idx] = None 

        return responses
        

        