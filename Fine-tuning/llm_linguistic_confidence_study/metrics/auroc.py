from sklearn.metrics import roc_auc_score
from omegaconf import DictConfig

class AUROC:
    def __init__(self, metric_cfg: DictConfig):
        self.metric_cfg = metric_cfg

    def evaluate(self, responses_df):
        responses_df = responses_df.query(""" accuracies == accuracies & confidences == confidences """)
        if self.metric_cfg.format == "simpleqa_like":
            if self.metric_cfg.exclude_not_attempted:
                filtered_responses_df = responses_df[responses_df["accuracies"] != "NOT_ATTEMPTED"]
                accuracies = (filtered_responses_df["accuracies"] == "CORRECT").astype(int).to_numpy()
                confidences = filtered_responses_df["confidences"].to_numpy()
            else:
                accuracies = (responses_df["accuracies"] == "CORRECT").astype(int).to_numpy()
                confidences = responses_df["confidences"].to_numpy()
            return roc_auc_score(accuracies, confidences)
        else:
            raise ValueError(f"Invalid format: {self.metric_cfg.format}")