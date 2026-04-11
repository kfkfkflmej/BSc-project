from omegaconf import DictConfig
from .openai_models import GPT
from .togetherai_models import TogetherAI
from .xai_models import Grok
from .anthropic_models import Claude
from .huggingface_models import Huggingfacemodel

OPEN_AI_MODEL_LIST = [
    "gpt-5", 
    "gpt-5-mini",
    "gpt-5-nano",
]

# this is a list of models that support batch inference
TOGETHER_AI_MODEL_LIST = [
    "deepseek-ai/DeepSeek-R1-0528-tput",
    "deepseek-ai/DeepSeek-V3",
    "meta-llama/Llama-3-70b-chat-hf",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "moonshotai/Kimi-K2-Instruct",
    "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "Qwen/Qwen3-235B-A22B-fp8-tput",
    "Qwen/QwQ-32B",
    "openai/whisper-large-v3",
    "deepseek-ai/DeepSeek-R1",
    "google/gemma-3n-E4B-it",
    "marin-community/marin-8b-instruct",
    "meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
    "togethercomputer/Refuel-Llm-V2",
    "togethercomputer/Refuel-Llm-V2-Small",
    "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    "openai/gpt-oss-120b",
    "zai-org/GLM-4.5-Air-FP8",
    "Qwen/Qwen3-235B-A22B-Thinking-2507",
    "openai/gpt-oss-20b",
]

ANTHROPIC_AI_MODEL_LIST = [
    "claude-opus-4-1-20250805",
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-haiku-20240307",
]

# Grok does not support batch inference; real-time inference is used
X_AI_MODEL_LIST = [
    "grok-4-0709",
    "grok-3",
    "grok-3-mini",
]

HUGGING_FACE_LIST = [
    "qwen3-8b",
    "qwen3-8b-lora-mapper0",
    "qwen3-8b-lora-mapper1",
    "qwen3-8b-lora-mapper2",
    "qwen3-8b-lora-mapper3",
    "qwen3-32b",
    "qwen3-32b-lora-mapper0",
    "qwen3-32b-lora-mapper1",
    "qwen3-32b-lora-mapper2",
    "qwen3-32b-lora-mapper3",
]

class LLM:
    def __init__(self, model_cfg: DictConfig):
        self.model_cfg = model_cfg
        self.model: GPT | TogetherAI | Grok | Claude | Huggingfacemodel = self.prepare_model(model_cfg)


    def __call__(self, prompts: list[str], task_name: str, batch_job_id: list[str] | str = None) -> list[str]:
        responses = self.model(prompts, task_name, batch_job_id)
        return responses


    def prepare_model(self, model_cfg: DictConfig):
        if model_cfg.name in OPEN_AI_MODEL_LIST:
            return GPT(model_cfg)
        elif model_cfg.name in TOGETHER_AI_MODEL_LIST:
            return TogetherAI(model_cfg)
        elif model_cfg.name in ANTHROPIC_AI_MODEL_LIST:
            return Claude(model_cfg)
        elif model_cfg.name in X_AI_MODEL_LIST:
            return Grok(model_cfg)
        elif model_cfg.name in HUGGING_FACE_LIST:
            return Huggingfacemodel(model_cfg)
        else:
            raise ValueError(f"Invalid model name: {model_cfg.name}")
        