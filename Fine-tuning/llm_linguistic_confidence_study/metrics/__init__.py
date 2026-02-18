from .acc import Accuracy
from .ece import ECE
from .auroc import AUROC
from .attempted_count import AttemptedCount
from omegaconf import DictConfig

class MetricEvaluator:
    def __init__(self, metric_cfg: DictConfig, dataset):
        self.metric_cfg = metric_cfg
        self.dataset = dataset
        self.metric_evaluator = self.get_metric_evaluator(metric_cfg)

    def evaluate(self, responses_df):
        # return the float value of the metric
        if self.metric_cfg.exclude_training:
            with open("llm_linguistic_confidence_study/datasets/problem_list.txt", "r", encoding="utf-8") as f:
                problem_list = [line.strip() for line in f.readlines()]
            responses_df = responses_df[~responses_df["problem"].isin(problem_list)]
        return self.metric_evaluator.evaluate(responses_df)

    def get_metric_evaluator(self, metric_cfg: DictConfig):
        if metric_cfg.name == "acc":
            return Accuracy(metric_cfg)
        elif metric_cfg.name == "ece":
            return ECE(metric_cfg)
        elif metric_cfg.name == "auroc":
            return AUROC(metric_cfg)
        elif metric_cfg.name == "attempted_count":
            return AttemptedCount(metric_cfg)
        else:
            raise ValueError(f"Metric {metric_cfg.name} not found")