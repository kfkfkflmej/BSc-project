from omegaconf import DictConfig



class LLM:
    def __init__(self, model_cfg: DictConfig):
        self.model_cfg = model_cfg


    def __call__(self, prompts: list[str], task_name: str, batch_job_id: list[str] | str = None) -> list[str]:
        responses = self.model(prompts, task_name, batch_job_id)
        return responses


    def prepare_model(self, model_cfg: DictConfig):
        model_name = model_cfg.name


        else:
            raise ValueError(f"Invalid model name: {model_cfg.name}")
        