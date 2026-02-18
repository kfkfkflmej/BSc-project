from .confidence_extraction_methods import ConfidenceExtractor
from .datasets import load_dataset
from .metrics import MetricEvaluator
from omegaconf import OmegaConf, DictConfig
import hydra
import logging
import os
from hydra.core.hydra_config import HydraConfig
import pandas as pd

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logging.info(OmegaConf.to_yaml(cfg, resolve=True))
    
    # load the dataset
    dataset = load_dataset(cfg.dataset)
    # save the results, responses and config
    save_dir = HydraConfig.get().runtime.output_dir 

    confidence_extractor = ConfidenceExtractor(cfg.confidence_extractor, cfg.qa_model)
    # return a dataframe with the following columns: question, gold_answer, reponse1, reponse2, reponse3, ..., confidence, accuracy
    responses_df = confidence_extractor(dataset, cfg.pre_runned_batch)
    responses_df.to_csv(os.path.join(save_dir, "responses.csv"), index=False)

    # evaluate the responses
    results = pd.DataFrame()
    for _, metric_cfg in cfg.metrics.items():
        metric_evaluator = MetricEvaluator(metric_cfg, dataset)
        score = metric_evaluator.evaluate(responses_df)
        logging.info(f"{metric_cfg}: {score}")
        results = pd.concat([results, pd.DataFrame({"metric": [OmegaConf.to_yaml(metric_cfg, resolve=True).replace("\n", "; ")], "score": [score]})])
    
    responses_df.to_csv(os.path.join(save_dir, "responses.csv"), index=False)
    results.to_csv(os.path.join(save_dir, "results.csv"), index=False)
    OmegaConf.save(cfg, os.path.join(save_dir, "config.yaml"))
    logging.info(f"Outputs saved to {save_dir}")
    
if __name__ == "__main__":
    main()