import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from omegaconf import DictConfig
import json
import os
import time
import logging


OPENAI_SYSTEM_PROMPT = "You are a helpful assistant."

class Claude():
    # https://docs.anthropic.com/en/docs/build-with-claude/batch-processing
    def __init__(self, model_cfg: DictConfig):
        self.model_cfg = model_cfg
        self.tasks = []

    def __call__(self, prompts: list[str], task_name: str, batch_job_id: str = None) -> list[str]:
        if batch_job_id is None:
            batch_job_id = self.prepare_batch_task_and_submit(prompts, task_name)
            # check batch job status
            while True:
                batch_job = self.check_batch_job(batch_job_id)
                if batch_job.processing_status != "ended":
                    logging.info(f"Batch job {batch_job.id} for {task_name} is {batch_job.processing_status}, waiting for 60 seconds...")
                    time.sleep(60)
                else:
                    logging.info(f"Batch job {batch_job.id} for {task_name} is completed")
                    break
        else:
            logging.info(f"Batch job {batch_job_id} is runned before, download the output file now...")

        # retrieve batch job output
        responses = self.retrieve_batch_job_output(batch_job_id)
        return responses

    
    def prepare_batch_task_and_submit(self, prompts: list[str], task_name: str) -> str:
        tasks = []
        for idx, prompt in enumerate(prompts):
            task = Request(
                custom_id = task_name + "_" + str(idx),      
                params = MessageCreateParamsNonStreaming(
                    model = self.model_cfg.name,  
                    system = OPENAI_SYSTEM_PROMPT,
                    messages = [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=1000,
                    thinking={
                        "type": "disabled"
                    },
                )
            )
            tasks.append(task)
        self.tasks = tasks
        # Write the tasks to a JSONL file
        # with open(f'batch_tasks/{task_name}.jsonl', 'w', encoding='utf-8') as f:
        #     for t in tasks:
        #         f.write(json.dumps(t, ensure_ascii=False) + '\n')

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        message_batch = client.messages.batches.create(requests=tasks)        

        # return batch id
        return message_batch.id


    def check_batch_job(self, batch_job_id):
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        message_batch = client.messages.batches.retrieve(batch_job_id)
        return message_batch


    def retrieve_batch_job_output(self, batch_job_id):
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        responses = []
        results = list(client.messages.batches.results(batch_job_id))
        # Sort results by custom_id
        results.sort(key=lambda r: int(r.custom_id.split("_")[-1]))

        for result in results:
            match result.result.type:
                case "succeeded":
                    print(f"Success! {result.custom_id}")
                    responses.append(result.result.message.content[0].text)
                case "errored":
                    responses.append(None)
                    if result.result.error.type == "invalid_request":
                        print(f"Validation error {result.custom_id}")
                    else:
                        print(f"Server error {result.custom_id}")
                case "expired":
                    responses.append(None)
                    print(f"Request expired {result.custom_id}")
                case _ :
                    responses.append(None)

        return responses


        
        
        