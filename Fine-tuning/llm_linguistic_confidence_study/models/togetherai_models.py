from together import Together
from omegaconf import DictConfig
import json
import os
import time
import logging
from datetime import datetime

OPENAI_SYSTEM_PROMPT = "You are a helpful assistant."

class TogetherAI():
    # https://docs.together.ai/docs/batch-inference
    def __init__(self, model_cfg: DictConfig):
        self.model_cfg = model_cfg
        self.tasks = []

    def __call__(self, prompts: list[str], task_name: str, batch_job_id: str = None) -> list[str]:
        if batch_job_id is None:
            batch_job_id = self.prepare_batch_task_and_submit(prompts, task_name)
            # check batch job status
            while True:
                batch_job = self.check_batch_job(batch_job_id)
                if batch_job.status != "COMPLETED":
                    logging.info(f"Batch job {batch_job.id} for {task_name} is {batch_job.status}, waiting for 60 seconds...")
                    time.sleep(60)
                else:
                    logging.info(f"Batch job {batch_job.id} for {task_name} is completed")
                    break
        else:
            logging.info(f"Batch job {batch_job_id} is runned before, download the output file now...")

        # retrieve batch job output
        responses = self.retrieve_batch_job_output(batch_job_id)
        return responses
    
    
    def get_output_response_and_logprobs(self, prompts: list[str], task_name: str, batch_job_id: str = None) -> tuple[list, list]:
        """
        For 
        """
        if batch_job_id is None:
            batch_job_id = self.prepare_batch_task_and_submit(prompts, task_name)
            # check batch job status
            while True:
                batch_job = self.check_batch_job(batch_job_id)
                if batch_job.status != "COMPLETED":
                    logging.info(f"Batch job {batch_job.id} for {task_name} is {batch_job.status}, waiting for 60 seconds...")
                    time.sleep(60)
                else:
                    logging.info(f"Batch job {batch_job.id} for {task_name} is completed")
                    break
        else:
            logging.info(f"Batch job {batch_job_id} is runned before, download the output file now...")

        # retrieve batch job output
        responses, logprobs = self.retrieve_batch_job_output_response_and_logprobs(batch_job_id)
        return responses, logprobs
    

    def get_output_token_logprob(self, prompts: list[str], task_name: str, batch_job_id: str = None) -> list[str]:
        if batch_job_id is None:
            batch_job_id = self.prepare_batch_task_and_submit(prompts, task_name)
            # check batch job status
            while True:
                batch_job = self.check_batch_job(batch_job_id)
                if batch_job.status != "COMPLETED":
                    logging.info(f"Batch job {batch_job.id} for {task_name} is {batch_job.status}, waiting for 60 seconds...")
                    time.sleep(60)
                else:
                    logging.info(f"Batch job {batch_job.id} for {task_name} is completed")
                    break
        else:
            logging.info(f"Batch job {batch_job_id} is runned before, download the output file now...")
            while True:
                batch_job = self.check_batch_job(batch_job_id)
                if batch_job.status != "COMPLETED":
                    logging.info(f"Batch job {batch_job.id} for {task_name} is {batch_job.status}, waiting for 60 seconds...")
                    time.sleep(60)
                else:
                    logging.info(f"Batch job {batch_job.id} for {task_name} is completed")
                    break

        # retrieve batch job output
        tokens, logprobs = self.retrieve_batch_job_output_token_logprobs(batch_job_id)
        return tokens, logprobs
    
    
    def prepare_batch_task_and_submit(self, prompts: list[str], task_name: str) -> str:
        '''
        This function checks the status of a batch job.
        batch_job_id: the id of the batch job
        return: batch job id
        '''
        if "qwen" in self.model_cfg.name.lower(): 
            no_think_prefix = "/no_think\n"
            print("Qwen no-think prefix preppended.")
        else:
            no_think_prefix = ""
        tasks = []
        for idx, prompt in enumerate(prompts):
            task = {
                "custom_id": task_name + "_" + str(idx),
                "body": {
                    "model": self.model_cfg.name,        
                    "messages": [
                        {
                            "role": "system",
                            "content": OPENAI_SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": no_think_prefix + prompt
                        }
                    ],
                    "reasoning_effort": "low",
                    "logprobs": 5,
                }
            }
            tasks.append(task)
        self.tasks = tasks

        task_file_path = f'llm_linguistic_confidence_study/batch_tasks/{task_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl'
        os.makedirs(os.path.dirname(task_file_path), exist_ok=True)
        with open(task_file_path, 'w', encoding='utf-8') as f:
            for t in tasks:
                f.write(json.dumps(t, ensure_ascii=False) + '\n')

        logging.info(f"Batch task file saved to {task_file_path}")
        
        client = Together(api_key=os.getenv("TOGETHERAI_API_KEY"))

        file_resp = client.files.upload(
            file=task_file_path,
            purpose='batch-api'
        )
        file_id = file_resp.id

        # Create the batch job 
        batch = client.batches.create_batch(file_id, endpoint="/v1/chat/completions")

        # return batch id
        return batch.id
    
    
    def check_batch_job(self, batch_job_id):
        '''
        This function retrieves the output file of a batch job.
        batch_job_id: the id of the batch job
        return: BatchJob Object
        '''
        client = Together(api_key=os.getenv("TOGETHERAI_API_KEY"))
        batch_job = client.batches.get_batch(batch_job_id)
        return batch_job


    
    def retrieve_batch_job_output(self, batch_job_id: str):
        client = Together(api_key=os.getenv("TOGETHERAI_API_KEY"))
        file_path = f"batch_tasks/tmp_together_{batch_job_id}_batch_output.jsonl"

        batch_stat = client.batches.get_batch(batch_job_id)

        responses = {}

        if self.check_batch_job(batch_job_id).status == "COMPLETED":
            # Download result file
            client.files.retrieve_content(id=batch_stat.output_file_id, output=file_path)
            with open(file_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    custom_id = entry["custom_id"]
                    answer = entry["response"]["body"]["choices"][0]["message"]["content"].strip()
                    responses[custom_id] = answer

            # Download error file if available
            if batch_stat.error_file_id:
                error_path = f"batch_tasks/tmp_together_{batch_job_id}_batch_errors.jsonl"
                client.files.retrieve_content(id=batch_stat.error_file_id, output=error_path)
                with open(error_path, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        entry = json.loads(line)
                        custom_id = entry["custom_id"]
                        if custom_id not in responses:
                            responses[custom_id] = None

        # Return list sorted by custom_id
        return [
            responses[k]
            for k in sorted(responses.keys(), key=lambda x: int(x.rsplit("_", 1)[-1]))
        ]
    

    def retrieve_batch_job_output_response_and_logprobs(self, batch_job_id: str) -> tuple[list[str], list[list[float]]]:
        client = Together(api_key=os.getenv("TOGETHERAI_API_KEY"))
        file_path = f"batch_tasks/tmp_together_{batch_job_id}_batch_output.jsonl"

        batch_stat = client.batches.get_batch(batch_job_id)

        responses = {}
        logprobs = {}

        try:

            if self.check_batch_job(batch_job_id).status == "COMPLETED":
                # Download result file
                client.files.retrieve_content(id=batch_stat.output_file_id, output=file_path)
                with open(file_path, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        entry = json.loads(line)
                        custom_id = entry["custom_id"]
                        responses[custom_id] = entry["response"]["body"]["choices"][0]["message"]["content"].strip()
                        logprobs[custom_id] = [d["logprob"] for d in entry["response"]["body"]["choices"][0]["logprobs"]["content"]]

                # Download error file if available
                if batch_stat.error_file_id:
                    error_path = f"batch_tasks/tmp_together_{batch_job_id}_batch_errors.jsonl"
                    client.files.retrieve_content(id=batch_stat.error_file_id, output=error_path)
                    with open(error_path, "r") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            entry = json.loads(line)
                            custom_id = entry["custom_id"]
                            if custom_id not in responses:
                                responses[custom_id] = ""
                                logprobs[custom_id] = []

            # Return list sorted by custom_id
            return [responses[k] for k in sorted(responses.keys(), key=lambda x: int(x.rsplit("_", 1)[-1]))], [logprobs[k] for k in sorted(logprobs.keys(), key=lambda x: int(x.rsplit("_", 1)[-1]))]
        except:
            if self.check_batch_job(batch_job_id).status == "COMPLETED":
                # Download result file
                client.files.retrieve_content(id=batch_stat.output_file_id, output=file_path)
                with open(file_path, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        entry = json.loads(line)
                        custom_id = entry["custom_id"]
                        responses[custom_id] = entry["response"]["body"]["choices"][0]["message"]["content"].strip()
                        logprobs[custom_id] = entry["response"]["body"]["choices"][0]["logprobs"]["token_logprobs"]

                # Download error file if available
                if batch_stat.error_file_id:
                    error_path = f"batch_tasks/tmp_together_{batch_job_id}_batch_errors.jsonl"
                    client.files.retrieve_content(id=batch_stat.error_file_id, output=error_path)
                    with open(error_path, "r") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            entry = json.loads(line)
                            custom_id = entry["custom_id"]
                            if custom_id not in responses:
                                responses[custom_id] = ""
                                logprobs[custom_id] = []

            # Return list sorted by custom_id
            return [responses[k] for k in sorted(responses.keys(), key=lambda x: int(x.rsplit("_", 1)[-1]))], [logprobs[k] for k in sorted(logprobs.keys(), key=lambda x: int(x.rsplit("_", 1)[-1]))]
    
    def retrieve_batch_job_output_token_logprobs(self, batch_job_id: str) -> tuple[list[str], list[list[float]]]:
        client = Together(api_key=os.getenv("TOGETHERAI_API_KEY"))
        file_path = f"batch_tasks/tmp_together_{batch_job_id}_batch_output.jsonl"

        batch_stat = client.batches.get_batch(batch_job_id)

        tokens = {}
        logprobs = {}

        try: # qwen
            if self.check_batch_job(batch_job_id).status == "COMPLETED":
                # Download result file
                client.files.retrieve_content(id=batch_stat.output_file_id, output=file_path)
                with open(file_path, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        entry = json.loads(line)
                        custom_id = entry["custom_id"]
                        tokens[custom_id] = entry["response"]["body"]["choices"][0]["logprobs"]["tokens"]
                        logprobs[custom_id] = entry["response"]["body"]["choices"][0]["logprobs"]["token_logprobs"]

                # Download error file if available
                if batch_stat.error_file_id:
                    error_path = f"batch_tasks/tmp_together_{batch_job_id}_batch_errors.jsonl"
                    client.files.retrieve_content(id=batch_stat.error_file_id, output=error_path)
                    with open(error_path, "r") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            entry = json.loads(line)
                            custom_id = entry["custom_id"]
                            if custom_id not in tokens.keys():
                                tokens[custom_id] = entry["response"]["body"]["choices"][0]["logprobs"]["tokens"]
                                logprobs[custom_id] = entry["response"]["body"]["choices"][0]["logprobs"]["token_logprobs"]

            # Return list sorted by custom_id
            return [tokens[k] for k in sorted(tokens.keys(), key=lambda x: int(x.rsplit("_", 1)[-1]))], [logprobs[k] for k in sorted(tokens.keys(), key=lambda x: int(x.rsplit("_", 1)[-1]))]
        except: # openai/gpt-oss
            if self.check_batch_job(batch_job_id).status == "COMPLETED":
                # Download result file
                client.files.retrieve_content(id=batch_stat.output_file_id, output=file_path)
                with open(file_path, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        entry = json.loads(line)
                        target_start = ['<|start|>', 'assistant', '<|channel|>', 'final', '<|message|>']
                        target_stop = '<|return|>'

                        entry_tokens = [d["token"] for d in entry["response"]["body"]["choices"][0]["logprobs"]["content"]]
                        entry_logprobs = [d["logprob"] for d in entry["response"]["body"]["choices"][0]["logprobs"]["content"]]

                        def find_sequence(entry_tokens, target):
                            for i in range(len(entry_tokens) - len(target) + 1):
                                if entry_tokens[i:i+len(target)] == target:
                                    return i + len(target)  # index after the match
                            return 0

                        # Find start
                        start_idx = find_sequence(entry_tokens, target_start)

                        # Find stop (first occurrence after start)
                        stop_idx = None
                        if start_idx is not None:
                            for j in range(start_idx, len(entry_tokens)):
                                if entry_tokens[j] == target_stop:
                                    stop_idx = j
                                    break

                        # Slice tokens
                        if start_idx is not None:
                            if stop_idx is not None:
                                filtered_tokens = entry_tokens[start_idx:stop_idx]
                                filtered_logprobs = entry_logprobs[start_idx:stop_idx]
                            else:
                                filtered_tokens = entry_tokens[start_idx:] 
                                filtered_logprobs = entry_logprobs[start_idx:]
                        else:
                            filtered_tokens = []  # no start found
                            filtered_logprobs = []
                        custom_id = entry["custom_id"]
                        tokens[custom_id] = filtered_tokens
                        logprobs[custom_id] = filtered_logprobs
                # Download error file if available
                if batch_stat.error_file_id:
                    error_path = f"batch_tasks/tmp_together_{batch_job_id}_batch_errors.jsonl"
                    client.files.retrieve_content(id=batch_stat.error_file_id, output=error_path)
                    with open(error_path, "r") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            entry = json.loads(line)
                            custom_id = entry["custom_id"]
                            if custom_id not in tokens.keys():
                                tokens[custom_id] = []
                                logprobs[custom_id] = []

            # Return list sorted by custom_id
            return [tokens[k] for k in sorted(tokens.keys(), key=lambda x: int(x.rsplit("_", 1)[-1]))], [logprobs[k] for k in sorted(tokens.keys(), key=lambda x: int(x.rsplit("_", 1)[-1]))]
        
        
