from omegaconf import DictConfig

class AttemptedCount:
    def __init__(self, metric_cfg: DictConfig):
        self.metric_cfg = metric_cfg

    def evaluate(self, responses_df):
        if self.metric_cfg.format == "simpleqa_like":
            correct_count = (responses_df["accuracies"] == "CORRECT").sum()
            incorrect_count = (responses_df["accuracies"] == "INCORRECT").sum()
            return correct_count + incorrect_count
        else:
            raise ValueError(f"Invalid format: {self.metric_cfg.format}")
    