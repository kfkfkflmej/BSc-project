from omegaconf import DictConfig
from omegaconf import DictConfig, ListConfig
import os
from transformers import AutoTokenizer
import torch
from vllm import LLM, SamplingParams
import csv
from vllm.lora.request import LoRARequest

class LLM:
    def __init__(self, model_cfg: DictConfig):
        self.model_cfg = model_cfg


    def __call__(self, prompts: list[str], task_name: str, batch_job_id: list[str] | str = None) -> list[str]:
        responses = self.model(prompts, task_name, batch_job_id)
        return responses


    def prepare_model(self, model_cfg: DictConfig):
        return Huggingfacemodel(model_cfg)

        
#Changed the name of the class to Huggingfacemodel to avoid confusion with the LLM class from vLLM library, which is also used in this code. The Huggingfacemodel class is a wrapper around the LLM class from vLLM library, and it is responsible for loading the model and generating responses based on the prompts.
class Huggingfacemodel:
    def __init__(self, model_cfg: DictConfig):
        self.model_cfg = model_cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __call__(
        self,
        prompts: list[str],
        task_name: str = None,
        batch_job_id: ListConfig | str = None,
    ) -> list[str]:
        if (batch_job_id == None) or (not os.path.exists(batch_job_id)):
            self.load_model()
            self.sampling_params = self.get_generate_config(self.model_cfg)
            responses = []
            messages = [{"role": "user", "content": prompt} for prompt in prompts]
            inputs = [
                self.tokenizer.apply_chat_template(
                    [msg],
                    tokenize=self.model_cfg.tokenize_output,
                    add_generation_prompt=self.model_cfg.add_generation_prompt,
                    enable_thinking=self.model_cfg.enable_thinking,
                )
                for msg in messages
            ]
            outputs = self.model.generate(
                inputs,
                self.sampling_params,
                lora_request=self.lora_request,
            )
            for output in outputs:
                output_ids = output.outputs[0].token_ids
                if 151668 in output_ids:
                    index = len(output_ids) - output_ids[::-1].index(151668)
                else:
                    index = 0
                text = self.tokenizer.decode(
                    output_ids[index:], skip_special_tokens=True
                ).strip("\n")
                responses.append(text)
            if batch_job_id != None:
                os.makedirs(os.path.dirname(batch_job_id), exist_ok=True)
                with open(batch_job_id, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    for resp in responses:
                        writer.writerow([resp])
        else:
            responses = []
            with open(batch_job_id, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        responses.append(row[0])
        return responses

    def load_model(self):
        self.model = LLM(
            model=self.model_cfg.model_name,
            tokenizer=self.model_cfg.model_name,
            trust_remote_code=self.model_cfg.trust_remote_code,
            dtype=self.model_cfg.dtype,
            tensor_parallel_size=self.model_cfg.tensor_parallel_size,
            gpu_memory_utilization=self.model_cfg.gpu_memory_utilization,
            max_model_len=self.model_cfg.max_tokens,
            disable_log_stats=self.model_cfg.disable_log_stats,
            enable_lora=self.model_cfg.enable_lora,
            max_lora_rank=self.model_cfg.max_lora_rank,
            seed=self.model_cfg.seed, 
        )
        """if self.model_cfg.name in [
            "qwen3-8b-lora-mapper0",
            "qwen3-8b-lora-mapper1",
            "qwen3-8b-lora-mapper2",
            "qwen3-8b-lora-mapper3",
            "qwen3-32b-lora-mapper0",
            "qwen3-32b-lora-mapper1",
            "qwen3-32b-lora-mapper2",
            "qwen3-32b-lora-mapper3",
        ]:
            self.lora_request = self.get_lora_request()
        else:
            self.lora_request = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_cfg.model_name,
            trust_remote_code=self.model_cfg.trust_remote_code,
        )

    def get_lora_request(self):
        return LoRARequest(
            lora_name=self.model_cfg.name,
            lora_path=self.model_cfg.lora_weight_path,
            lora_int_id=1,
        )
    """
    def get_generate_config(self, cfg):
        return SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            min_p=cfg.min_p,
            max_tokens=cfg.max_tokens,
        )
        