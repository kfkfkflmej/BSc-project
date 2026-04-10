from omegaconf import DictConfig

class LLM:
    def __init__(self, model_cfg: DictConfig):
        self.model_cfg = model_cfg


    def __call__(self, prompts: list[str], task_name: str, batch_job_id: list[str] | str = None) -> list[str]:
        responses = self.model(prompts, task_name, batch_job_id)
        return responses


    def prepare_model(self, model_cfg: DictConfig):
        model_name = model_cfg.name
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto" # Automatically handles device placement (e.g., GPU if available)
        )

        

        # if model_name != "gemma-3-1b-it":
        #     raise ValueError(f"Invalid model name: {model_cfg.name}")
        