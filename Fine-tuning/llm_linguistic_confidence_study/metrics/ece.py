from omegaconf import DictConfig
import numpy as np

class ECE:
    def __init__(self, metric_cfg: DictConfig):
        self.metric_cfg = metric_cfg
        self.n_bins = metric_cfg.n_bins

    def evaluate(self, responses_df):
        if self.metric_cfg.format == "simpleqa_like":
            if self.metric_cfg.exclude_not_attempted:
                filtered_responses_df = responses_df[responses_df["accuracies"] != "NOT_ATTEMPTED"]
                accuracies = (filtered_responses_df["accuracies"] == "CORRECT").astype(int).to_numpy()
                confidences = filtered_responses_df["confidences"].to_numpy()
            else:
                accuracies = (responses_df["accuracies"] == "CORRECT").astype(int).to_numpy()
                confidences = responses_df["confidences"].to_numpy()
            ece = self.compute_ece(accuracies, confidences)
            return ece
        else:
            raise ValueError(f"Invalid format: {self.metric_cfg.format}")   
        
    def compute_ece(self, accuracies, confidences):
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece