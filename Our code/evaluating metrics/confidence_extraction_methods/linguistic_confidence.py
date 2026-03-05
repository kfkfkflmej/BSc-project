import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from ..models import LLM
import pandas as pd
from omegaconf import DictConfig, OmegaConf, ListConfig, MissingMandatoryValue
import re
import logging

SIMPLE_QA_EVAL_VANILLA_TEMPLATE = """
Answer the following question using a succinct (at most one sentence) and full answer.

Question: {question}
Answer:
""".strip()

SIMPLE_QA_EVAL_VANILLA_UNCERTAINTY_TEMPLATE = """
Answer the following question using a succinct (at most one sentence) and full answer. If you are uncertain about your answer to the question, convey this uncertainty linguistically by precisely hedging this answer.

Question: {question}
Answer:
""".strip()

MMLU_PRO_EVAL_VANILLA_TEMPLATE = """
Answer the following question using a succinct (at most one sentence) and full answer.

Question: {question}
Answer:
""".strip()

MMLU_PRO_EVAL_VANILLA_UNCERTAINTY_TEMPLATE = """
Answer the following question using a succinct (at most one sentence) and full answer. If you are uncertain about your answer to the question, convey this uncertainty linguistically by precisely hedging this answer.

Question: {question}
Answer:
""".strip()

NQ_OPEN_EVAL_VANILLA_TEMPLATE = """
Answer the following question using a succinct (at most one sentence) and full answer.

Question: {question}
Answer:
""".strip()

NQ_OPEN_EVAL_VANILLA_UNCERTAINTY_TEMPLATE = """
Answer the following question using a succinct (at most one sentence) and full answer. If you are uncertain about your answer to the question, convey this uncertainty linguistically by precisely hedging this answer.

Question: {question}
Answer:
""".strip()

POP_QA_EVAL_VANILLA_TEMPLATE = """
Answer the following question using a succinct (at most one sentence) and full answer.

Question: {question}
Answer:
""".strip()

POP_QA_EVAL_VANILLA_UNCERTAINTY_TEMPLATE = """
Answer the following question using a succinct (at most one sentence) and full answer. If you are uncertain about your answer to the question, convey this uncertainty linguistically by precisely hedging this answer.

Question: {question}
Answer:
""".strip()

DECISIVENESS_PROMPT = """You will be given a question and a proposed answer. Your task is to extract any assertions made by the proposed answer; for each assertion, you should output a
decisiveness score: a number in [0,1] representing how likely you think this assertion is to be true, judged solely based on the provided answer. You should pay
special attention to the usage of any hedging modifiers in the original answer, used to convey uncertainty in the truthfulness of the assertion. If the proposed
answer punts the question, the extracted assertion should be the empty string and the decisiveness score should be 1.0.

Here are some examples:

Question: criminal minds episode where jj becomes a profiler
Proposed answer: I think JJ officially becomes a profiler in Criminal Minds season 6, episode 2.
Extracted assertion: JJ officially becomes a profiler in Criminal Minds season 6, episode 2.
Decisiveness score: 0.8.

Question: criminal minds episode where jj becomes a profiler
Proposed answer: JJ officially becomes a profiler in Criminal Minds season 6, in either the first or second episode.
Extracted assertion: JJ officially becomes a profiler in Criminal Minds season 6, episode 1.
Decisiveness score: 0.5.
Extracted assertion: JJ officially becomes a profiler in Criminal Minds season 6, episode 2.
Decisiveness score: 0.5.

Question: criminal minds episode where jj becomes a profiler
Proposed answer: I'm not really sure about this, but I think the episode in which JJ officially becomes a profiler in Criminal Minds may be episode 2 in season 6.
Extracted assertion: JJ officially becomes a profiler in Criminal Minds season 6, episode 2.
Decisiveness score: 0.6.

Question: criminal minds episode where jj becomes a profiler
Proposed answer: I don't know which episode you're referring to.
Extracted assertion:
Decisiveness score: 1.0

Question: {question}
Proposed answer: {response}
""".strip()

DIRECT_PROMPT = """
Please provide only a confidence score between 0 and 100, based solely on the degree of confidence expressed in the tone of the following sentence (without using any external or prior knowledge): {sentence}
""".strip()


class LinguisticConfidenceExtractor():
    def __init__(self, confidence_extraction_method_cfg, qa_model_cfg):
        self.confidence_extraction_method_cfg = confidence_extraction_method_cfg
        self.qa_model_cfg = qa_model_cfg
        self.qa_model = self.get_qa_model(self.qa_model_cfg)
        self.confidence_mapper = self.get_confidence_mapper(self.confidence_extraction_method_cfg)
        
    def get_qa_model(self, qa_model_cfg):
        return LLM(qa_model_cfg)
    
    def get_confidence_mapper(self, confidence_extraction_method_cfg):
        if confidence_extraction_method_cfg.mapper_name == "self-trained":
            confidence_estimator = LinguisticConfidenceEstimator(confidence_extraction_method_cfg)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            confidence_estimator.reg_head.load_state_dict(torch.load(confidence_extraction_method_cfg.state_dict_path))
            return confidence_estimator.to(device)
        elif confidence_extraction_method_cfg.mapper_name == "decisiveness":
            return DecisivenessEstimator(confidence_extraction_method_cfg)
        elif confidence_extraction_method_cfg.mapper_name == "direct-prompt":
            ece_model_config = OmegaConf.create(
                {
                    "name":"gpt-5-mini",
                }
            )
            return DirectPromptEstimator(ece_model_config)
        else:
            raise ValueError(f"Invalid confidence extraction method: {confidence_extraction_method_cfg}")
        
    def __call__(self, dataset, pre_runned_batch_info: DictConfig):
        qa_batch_job_id = pre_runned_batch_info.qa_batch_id
        grader_batch_job_id = pre_runned_batch_info.grader_batch_id
        try:
            ece_batch_job_id = pre_runned_batch_info.ece_batch_id
        except (AttributeError, MissingMandatoryValue):
            ece_batch_job_id = None
        
        task_model_name = self.qa_model_cfg.name.split("/")[-1] if "/" in self.qa_model_cfg.name else self.qa_model_cfg.name
        if dataset.name == "simple_qa" or dataset.name == "mini_simple_qa":
            qa_responses = self.generate_qa_responses(dataset.df, self.confidence_extraction_method_cfg, task_name=f"simple_qa_{task_model_name}_lc_qa", qa_batch_job_id=qa_batch_job_id)
            # combine qa_responses and dataset_df
            response_df = dataset.df.copy()
            response_df["responses"] = qa_responses
            # confidence estimation
            confidences = self.confidence_mapper(response_df, batch_job_id=ece_batch_job_id, task_name=f"simple_qa_{task_model_name}_lc_ece")
            response_df["confidences"] = confidences
            # grade the accuracy of the confidence scores
            accuracies = dataset.grade_responses(response_df["responses"], grader_batch_job_id=grader_batch_job_id, task_name=f"simple_qa_{task_model_name}_lc_grader")
            response_df["accuracies"] = accuracies
        elif dataset.name == "nq_open" or dataset.name == "mini_nq_open":
            qa_responses = self.generate_qa_responses(dataset.df, self.confidence_extraction_method_cfg, task_name=f"nq_open_{task_model_name}_lc_qa", qa_batch_job_id=qa_batch_job_id)
            # combine qa_responses and dataset_df
            response_df = dataset.df.copy()
            response_df["responses"] = qa_responses
            # confidence estimation
            confidences = self.confidence_mapper(response_df, batch_job_id=ece_batch_job_id, task_name=f"simple_qa_{task_model_name}_lc_ece")
            response_df["confidences"] = confidences
            # grade the accuracy of the confidence scores
            accuracies = dataset.grade_responses(response_df["responses"], grader_batch_job_id=grader_batch_job_id, task_name=f"nq_open_{task_model_name}_lc_grader")
            response_df["accuracies"] = accuracies
        elif dataset.name == "mmlu_pro" or dataset.name == "mini_mmlu_pro":
            qa_responses = self.generate_qa_responses(dataset.df, self.confidence_extraction_method_cfg, task_name=f"mmlu_pro_{task_model_name}_lc_qa", qa_batch_job_id=qa_batch_job_id)
            # combine qa_responses and dataset_df
            response_df = dataset.df.copy()
            response_df["responses"] = qa_responses
            # confidence estimation
            confidences = self.confidence_mapper(response_df, batch_job_id=ece_batch_job_id, task_name=f"simple_qa_{task_model_name}_lc_ece")
            response_df["confidences"] = confidences
            # grade the accuracy of the confidence scores
            accuracies = dataset.grade_responses(response_df["responses"], grader_batch_job_id=grader_batch_job_id, task_name=f"mmlu_pro_{task_model_name}_lc_grader")
            response_df["accuracies"] = accuracies
        elif dataset.name == "pop_qa" or dataset.name == "mini_pop_qa":
            qa_responses = self.generate_qa_responses(dataset.df, self.confidence_extraction_method_cfg, task_name=f"pop_qa_{task_model_name}_lc_qa", qa_batch_job_id=qa_batch_job_id)
            # combine qa_responses and dataset_df
            response_df = dataset.df.copy()
            response_df["responses"] = qa_responses
            # confidence estimation
            confidences = self.confidence_mapper(response_df, batch_job_id=ece_batch_job_id, task_name=f"simple_qa_{task_model_name}_lc_ece")
            response_df["confidences"] = confidences
            # grade the accuracy of the confidence scores
            accuracies = dataset.grade_responses(response_df["responses"], grader_batch_job_id=grader_batch_job_id, task_name=f"pop_qa_{task_model_name}_lc_grader")
            response_df["accuracies"] = accuracies
        else:
            raise ValueError(f"Invalid dataset name: {dataset.name}")
        # return the response_df
        return response_df


    def generate_qa_responses(self, dataset_df: pd.DataFrame, confidence_extraction_method_cfg: DictConfig, task_name: str, qa_batch_job_id: str = None) -> list[str]:
        if "simple_qa" in task_name:
            # prepare prompts
            if confidence_extraction_method_cfg.qa_template == "vanilla":
                prompt_template = SIMPLE_QA_EVAL_VANILLA_TEMPLATE
            elif confidence_extraction_method_cfg.qa_template == "vanilla_uncertainty":
                prompt_template = SIMPLE_QA_EVAL_VANILLA_UNCERTAINTY_TEMPLATE
            else:
                raise ValueError(f"Invalid qa template: {confidence_extraction_method_cfg.qa_template}")
            qa_prompts = [prompt_template.format(question=row["problem"]) for _, row in dataset_df.iterrows()]
            # generate responses
            responses = self.qa_model(qa_prompts, task_name=task_name, batch_job_id=qa_batch_job_id)
            # post-process the responses if needed
            return responses
        elif "mmlu_pro" in task_name:
            # prepare prompts
            if confidence_extraction_method_cfg.qa_template == "vanilla":
                prompt_template = MMLU_PRO_EVAL_VANILLA_TEMPLATE
            elif confidence_extraction_method_cfg.qa_template == "vanilla_uncertainty":
                prompt_template = MMLU_PRO_EVAL_VANILLA_UNCERTAINTY_TEMPLATE
            else:
                raise ValueError(f"Invalid qa template: {confidence_extraction_method_cfg.qa_template}")
            qa_prompts = [prompt_template.format(question=row["problem"]) for _, row in dataset_df.iterrows()]
            # generate responses
            responses = self.qa_model(qa_prompts, task_name=task_name, batch_job_id=qa_batch_job_id)
            # post-process the responses if needed
            return responses
        elif "nq_open" in task_name:
            # prepare prompts
            if confidence_extraction_method_cfg.qa_template == "vanilla":
                prompt_template = NQ_OPEN_EVAL_VANILLA_TEMPLATE
            elif confidence_extraction_method_cfg.qa_template == "vanilla_uncertainty":
                prompt_template = NQ_OPEN_EVAL_VANILLA_UNCERTAINTY_TEMPLATE
            else:
                raise ValueError(f"Invalid qa template: {confidence_extraction_method_cfg.qa_template}")
            qa_prompts = [prompt_template.format(question=row["problem"]) for _, row in dataset_df.iterrows()]
            # generate responses
            responses = self.qa_model(qa_prompts, task_name=task_name, batch_job_id=qa_batch_job_id)
            # post-process the responses if needed
            return responses
        elif "pop_qa" in task_name:
            # prepare prompts
            if confidence_extraction_method_cfg.qa_template == "vanilla":
                prompt_template = POP_QA_EVAL_VANILLA_TEMPLATE
            elif confidence_extraction_method_cfg.qa_template == "vanilla_uncertainty":
                prompt_template = POP_QA_EVAL_VANILLA_UNCERTAINTY_TEMPLATE
            else:
                raise ValueError(f"Invalid qa template: {confidence_extraction_method_cfg.qa_template}")
            qa_prompts = [prompt_template.format(question=row["problem"]) for _, row in dataset_df.iterrows()]
            # generate responses
            responses = self.qa_model(qa_prompts, task_name=task_name, batch_job_id=qa_batch_job_id)
            # post-process the responses if needed
            return responses
        else:
            raise ValueError("Invalid task name.")

        
class LinguisticConfidenceEstimator(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = AutoModel.from_pretrained(cfg.model_name)
        self.reg_head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 1),
            nn.Sigmoid()  # output range in [0, 1]
        )
        
    def __call__(self, response_df: pd.DataFrame, task_name: str, batch_size: int = 32, batch_job_id: ListConfig | str = None) -> list[float]:
        logging.info(f"Batch job {batch_job_id} for {task_name} is computing by self-trained.")
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        responses = response_df["responses"].fillna("").tolist()
        all_confidences = []

        for i in range(0, len(responses), batch_size):
            batch_responses = responses[i:i + batch_size]

            inputs = tokenizer(
                batch_responses,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output = self.encoder(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )
                cls_hidden = output.last_hidden_state[:, 0]
                confidence_scores = self.reg_head(cls_hidden).squeeze(-1).detach().cpu().numpy()
                all_confidences.extend(confidence_scores.tolist())

        return all_confidences
    
class DecisivenessEstimator():
    def __init__(self, cfg: DictConfig):
        self.decisiveness_prompt_template = DECISIVENESS_PROMPT
        self.model = LLM(cfg.model)

    def __call__(self, dataset_df: pd.DataFrame, task_name: str, batch_job_id: ListConfig | str = None) -> list[float]:
        logging.info(f"Batch job {batch_job_id} for {task_name} is computing by decisiveness.")
        decisiveness_prompts = []
        for i in range(len(dataset_df)):
            decisiveness_prompt = self.decisiveness_prompt_template.format(question=dataset_df.iloc[i]["question"], response=dataset_df.iloc[i]["response"])
            decisiveness_prompts.append(decisiveness_prompt)
        decisiveness_responses = self.model(decisiveness_prompts, task_name="decisiveness")
        decisiveness_scores = self.extract_confidence_scores(decisiveness_responses)
        return decisiveness_scores
    
    def extract_confidence_scores(self, decisiveness_responses: list[str]) -> list[float]:
        # use regex to extract the confidence score from the response
        # decisiveness_responses: list of strings
        # return a list of confidence scores in [0, 1]
        confidence_scores = []
        for response in decisiveness_responses:
            confidence_scores.append(float(re.search(r"Decisiveness score: (\d+\.\d+)", response).group(1)))
        return confidence_scores
    
class DirectPromptEstimator():
    def __init__(self, cfg):
        self.ece_cfg = cfg

    def __call__(self, response_df: pd.DataFrame, task_name: str, batch_job_id: ListConfig | str = None) -> list[float]:
        logging.info(f"Batch job {batch_job_id} for {task_name} is computing by direct prompt.")
        responses = response_df["responses"].tolist()
        prompts = [DIRECT_PROMPT.format(sentence=resp) for resp in responses]
        ece_model = LLM(self.ece_cfg)
        scores = ece_model(prompts, task_name=task_name, batch_job_id=batch_job_id)
        try:
            scores = [0.5 if score.strip() == '' else (int(score) / 100 if score.strip().isdigit() and 0 <= int(score) <= 100 else (float(score) if (score.replace('.', '', 1).isdigit() and 0 <= float(score) <= 1) else 0.5)) for score in scores]
        except Exception as e:
            logging.error(f"{scores[2767]}, {response_df.at[2767, "responses"]}")
            raise(e)
        return scores