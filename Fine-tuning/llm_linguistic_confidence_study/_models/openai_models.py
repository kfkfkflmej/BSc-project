from openai import OpenAI
from omegaconf import DictConfig, ListConfig
import json
import os
import time
from datetime import datetime
import logging

OPENAI_SYSTEM_PROMPT = "You are a helpful assistant."

class GPT():
    def __init__(self, model_cfg: DictConfig):
        self.model_cfg = model_cfg

    def __call__(self, prompts: list[str], task_name: str, batch_job_id: ListConfig | str = None) -> list[str]:
        if isinstance(batch_job_id, ListConfig):
            responses = []
            for batch_job_id in batch_job_id:
                responses.extend(self.retrieve_batch_job_output(batch_job_id))
        elif isinstance(batch_job_id, str):
            while True:
                batch_job = self.check_batch_job(batch_job_id)
                if batch_job.status != "completed" and batch_job.status != "expired":
                    logging.info(f"Batch job {batch_job.id} for {task_name} is {batch_job.status}, waiting for 60 seconds...")
                    time.sleep(60)
                else:
                    logging.info(f"Batch job {batch_job.id} for {task_name} is {batch_job.status}")
                    break
            responses = self.retrieve_batch_job_output(batch_job_id)
        elif batch_job_id is None:
            batch_job_id, task_file_path = self.prepare_batch_task_and_submit(prompts, task_name)
            # check batch job status
            while True:
                batch_job = self.check_batch_job(batch_job_id)
                if batch_job.status != "completed" and batch_job.status != "expired":
                    logging.info(f"Batch job {batch_job.id} for {task_name} is {batch_job.status}, waiting for 60 seconds...")
                    time.sleep(60)
                else:
                    logging.info(f"Batch job {batch_job.id} for {task_name} is {batch_job.status}")
                    break
            responses = self.retrieve_batch_job_output(batch_job_id)
        else:
            responses = self.retrieve_batch_job_output(batch_job_id)
        return responses
    
    def prepare_batch_task_and_submit(self, prompts: list[str], task_name: str) -> list[str]:
        #########################################################
        # prepare request for batch grading
        #########################################################
        tasks = []
        for idx, prompt in enumerate(prompts):
            task = {
                "custom_id": task_name + "_" + str(idx),             # Custom ID must be a string
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model_cfg.name,         # Use gpt-5-mini
                    "reasoning_effort": "minimal",   # minimal reasoning
                    "messages": [
                        {
                            "role": "system",
                            "content": OPENAI_SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }
            }
            tasks.append(task)

        # Write the tasks to a JSONL file
        task_file_path = f'llm_linguistic_confidence_study/batch_tasks/{task_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl'
        os.makedirs(os.path.dirname(task_file_path), exist_ok=True)
        with open(task_file_path, 'w', encoding='utf-8') as f:
            for t in tasks:
                f.write(json.dumps(t, ensure_ascii=False) + '\n')

        logging.info(f"Batch task file saved to {task_file_path}")

        #########################################################
        # send request to batch grading
        #########################################################
        # Initialize OpenAI client with API key from environment variable
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Upload the batch task file; purpose must be "batch"
        batch_file = client.files.create(
            file=open(task_file_path, 'rb'),
            purpose='batch'
        )

        # Create the batch job; completion window can be 24h, 48h, etc.
        batch_job = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        logging.info(f"Batch job created: {batch_job.id}")
        # return batch job id and task file path
        return batch_job.id, task_file_path
    
    '''
    This function checks a batch job.
    batch_job_id: the id of the batch job
    return: the batch job object
    '''
    def check_batch_job(self, batch_job_id: str) -> str:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        batch_job = client.batches.retrieve(batch_job_id)
        return batch_job


    '''
    This function retrieves the output file of a batch job.
    batch_job_id: the id of the batch job
    return: a list of strings, each string is a response of a simple-qa question, in the same order as the simple-qa questions
    '''

    def retrieve_batch_job_output(self, batch_job_id: str) -> list[str | None]:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        batch_job = client.batches.retrieve(batch_job_id)

        # --- Parse results file ---
        result_bytes = client.files.content(batch_job.output_file_id).content
        result_text = result_bytes.decode("utf-8")

        merged = {}
        for line in result_text.strip().split("\n"):
            entry = json.loads(line)
            custom_id = entry["custom_id"]
            answer = entry["response"]["body"]["choices"][0]["message"]["content"].strip()
            merged[custom_id] = answer

        # --- Parse errors file ---
        if batch_job.error_file_id:
            error_bytes = client.files.content(batch_job.error_file_id).content
            error_text = error_bytes.decode("utf-8")

            for line in error_text.strip().split("\n"):
                entry = json.loads(line)
                custom_id = entry["custom_id"]
                if custom_id not in merged:  # don’t overwrite a valid answer
                    merged[custom_id] = ""

        # --- Build sorted list of responses by trailing number ---
        responses = [
            merged[k]
            for k in sorted(
                merged.keys(),
                key=lambda x: int(x.rsplit("_", 1)[-1])  # grab the number after the last "_"
            )
        ]

        return responses
