from omegaconf import DictConfig

class Accuracy:
    def __init__(self, metric_cfg: DictConfig):
        self.metric_cfg = metric_cfg

    def evaluate(self, responses_df):
        if self.metric_cfg.format == "simpleqa_like":
            correct_count = (responses_df["accuracies"] == "CORRECT").sum()
            incorrect_count = (responses_df["accuracies"] == "INCORRECT").sum()
            not_attempted_count = (responses_df["accuracies"] == "NOT_ATTEMPTED").sum()
            if self.metric_cfg.exclude_not_attempted:
                accuracy = correct_count / (correct_count + incorrect_count)
            else:
                accuracy = correct_count / (correct_count + incorrect_count + not_attempted_count)
            return accuracy
        else:
            raise ValueError(f"Invalid format: {self.metric_cfg.format}")
    