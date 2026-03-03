import pandas as pd
import re
from ..datasets import MMLUProDataset, SimpleQADataset
from ..models import LLM
from omegaconf import DictConfig
import numpy as np

# Simple QA Original Paper Prompt: https://arxiv.org/pdf/2411.04368
SIMPLE_QA_QA_VANILLA_TEMPLATE = """
Answer the following question using a succinct (at most one sentence) and full answer.

Question: {question}
Answer:
""".strip()

# Simple QA Original Paper Prompt: https://arxiv.org/pdf/2411.04368
MMLU_PRO_QA_VANILLA_TEMPLATE = """
Answer the following question using a succinct (at most one sentence) and full answer.

Question: {question}
Answer:
""".strip()

# Simple QA Original Paper Prompt: https://arxiv.org/pdf/2411.04368
NQ_OPEN_QA_VANILLA_TEMPLATE = """
Answer the following question using a succinct (at most one sentence) and full answer.

Question: {question}
Answer:
""".strip()

# Simple QA Original Paper Prompt: https://arxiv.org/pdf/2411.04368
POP_QA_QA_VANILLA_TEMPLATE = """
Answer the following question using a succinct (at most one sentence) and full answer.

Question: {question}
Answer:
""".strip()



class LogitsPerplexityConfidenceExtractor():
    def __init__(self, confidence_extraction_method_cfg, qa_model_cfg):
        self.confidence_extraction_method_cfg = confidence_extraction_method_cfg
        self.qa_model_cfg = qa_model_cfg
        self.qa_model = self.get_qa_model(self.qa_model_cfg)


    def get_qa_model(self, qa_model_cfg):
        return LLM(qa_model_cfg)
    

    def __call__(self, dataset: MMLUProDataset | SimpleQADataset, pre_runned_batch_info: DictConfig):
        qa_batch_job_id = pre_runned_batch_info.qa_batch_id
        grader_batch_job_id = pre_runned_batch_info.grader_batch_id
        
        task_model_name = self.qa_model_cfg.name.split("/")[-1] if "/" in self.qa_model_cfg.name else self.qa_model_cfg.name

        def calculate_logits_perplexity(logprobs: list):
            try:
                return float(np.exp(np.array(logprobs).mean()))
            except:
                return None

        if dataset.name == "simple_qa" or dataset.name == "mini_simple_qa":
            qa_responses, qa_logprobs = self.generate_qa_responses(dataset.df, self.confidence_extraction_method_cfg, task_name=f"simple_qa_{task_model_name}_perplexity_qa", qa_batch_job_id=qa_batch_job_id)
            # combine qa_responses and dataset_df
            response_df = dataset.df.copy()
            response_df["responses"] = qa_responses
            response_df["logprobs"] = qa_logprobs
            response_df["confidences"] =  response_df["logprobs"].apply(calculate_logits_perplexity)
            # grade the accuracy of the confidence scores
            accuracies = dataset.grade_responses(response_df["responses"], grader_batch_job_id=grader_batch_job_id, task_name=f"simple_qa_{task_model_name}_perplexity_grader")
            response_df["accuracies"] = accuracies 

        elif dataset.name == "mmlu_pro" or dataset.name == "mini_mmlu_pro":
            qa_responses, qa_logprobs = self.generate_qa_responses(dataset.df, self.confidence_extraction_method_cfg, task_name=f"mmlu_pro_{task_model_name}_perplexity_qa", qa_batch_job_id=qa_batch_job_id)
            # combine qa_responses and dataset_df
            response_df = dataset.df.copy()
            response_df["responses"] = qa_responses
            response_df["logprobs"] = qa_logprobs
            response_df["confidences"] =  response_df["logprobs"].apply(calculate_logits_perplexity)
            # grade the accuracy of the confidence scores
            accuracies = dataset.grade_responses(response_df["responses"], grader_batch_job_id=grader_batch_job_id, task_name=f"mmlu_pro_{task_model_name}_perplexity_grader")
            response_df["accuracies"] = accuracies 
        elif dataset.name == "nq_open" or dataset.name == "mini_nq_open":
            qa_responses, qa_logprobs = self.generate_qa_responses(dataset.df, self.confidence_extraction_method_cfg, task_name=f"nq_open_{task_model_name}_perplexity_qa", qa_batch_job_id=qa_batch_job_id)
            # combine qa_responses and dataset_df
            response_df = dataset.df.copy()
            response_df["responses"] = qa_responses
            response_df["logprobs"] = qa_logprobs
            response_df["confidences"] =  response_df["logprobs"].apply(calculate_logits_perplexity)
            # grade the accuracy of the confidence scores
            accuracies = dataset.grade_responses(response_df["responses"], grader_batch_job_id=grader_batch_job_id, task_name=f"nq_open_{task_model_name}_perplexity_grader")
            response_df["accuracies"] = accuracies 
        elif dataset.name == "pop_qa" or dataset.name == "mini_pop_qa":
            qa_responses, qa_logprobs = self.generate_qa_responses(dataset.df, self.confidence_extraction_method_cfg, task_name=f"pop_qa_{task_model_name}_perplexity_qa", qa_batch_job_id=qa_batch_job_id)
            # combine qa_responses and dataset_df
            response_df = dataset.df.copy()
            response_df["responses"] = qa_responses
            response_df["logprobs"] = qa_logprobs
            response_df["confidences"] =  response_df["logprobs"].apply(calculate_logits_perplexity)
            # grade the accuracy of the confidence scores
            accuracies = dataset.grade_responses(response_df["responses"], grader_batch_job_id=grader_batch_job_id, task_name=f"pop_qa_{task_model_name}_perplexity_grader")
            response_df["accuracies"] = accuracies 
        else:
            raise ValueError(f"Invalid dataset name: {dataset.name}")
        # return the response_df
        return response_df
        

    def generate_qa_responses(self, dataset_df: pd.DataFrame, confidence_extraction_method_cfg: DictConfig, task_name: str, qa_batch_job_id: str = None) -> list[str]:
        if "simple_qa" in task_name:
            # prepare prompts
            if confidence_extraction_method_cfg.qa_template == "vanilla":
                prompt_template = SIMPLE_QA_QA_VANILLA_TEMPLATE
            else:
                raise ValueError(f"Invalid QA template: {confidence_extraction_method_cfg.qa_template}")
            qa_prompts = [prompt_template.format(question=row["problem"]) for _, row in dataset_df.iterrows()]
            # generate responses
            responses, logprobs = self.qa_model.model.get_output_response_and_logprobs(qa_prompts, task_name=task_name, batch_job_id=qa_batch_job_id)
            # post-process the responses if needed
            return responses, logprobs
        elif "mmlu_pro" in task_name:
            # prepare prompts
            if confidence_extraction_method_cfg.qa_template == "vanilla":
                prompt_template = MMLU_PRO_QA_VANILLA_TEMPLATE
            else:
                raise ValueError(f"Invalid QA template: {confidence_extraction_method_cfg.qa_template}")
            qa_prompts = [prompt_template.format(question=row["problem"]) for _, row in dataset_df.iterrows()]
            # generate responses
            responses, logprobs = self.qa_model.model.get_output_response_and_logprobs(qa_prompts, task_name=task_name, batch_job_id=qa_batch_job_id)
            # post-process the responses if needed
            return responses, logprobs
        elif "nq_open" in task_name:
            # prepare prompts
            if confidence_extraction_method_cfg.qa_template == "vanilla":
                prompt_template = NQ_OPEN_QA_VANILLA_TEMPLATE
            else:
                raise ValueError(f"Invalid QA template: {confidence_extraction_method_cfg.qa_template}")
            qa_prompts = [prompt_template.format(question=row["problem"]) for _, row in dataset_df.iterrows()]
            # generate responses
            responses, logprobs = self.qa_model.model.get_output_response_and_logprobs(qa_prompts, task_name=task_name, batch_job_id=qa_batch_job_id)
            # post-process the responses if needed
            return responses, logprobs
        elif "pop_qa" in task_name:
            # prepare prompts
            if confidence_extraction_method_cfg.qa_template == "vanilla":
                prompt_template = POP_QA_QA_VANILLA_TEMPLATE
            else:
                raise ValueError(f"Invalid QA template: {confidence_extraction_method_cfg.qa_template}")
            qa_prompts = [prompt_template.format(question=row["problem"]) for _, row in dataset_df.iterrows()]
            # generate responses
            responses, logprobs = self.qa_model.model.get_output_response_and_logprobs(qa_prompts, task_name=task_name, batch_job_id=qa_batch_job_id)
            # post-process the responses if needed
            return responses, logprobs
        else:
            raise ValueError("Invalid task name.")