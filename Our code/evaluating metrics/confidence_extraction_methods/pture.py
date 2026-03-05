import pandas as pd
import re
from ..datasets import MMLUProDataset, SimpleQADataset
from ..models import LLM
from omegaconf import DictConfig
from ..models.togetherai_models import TogetherAI
import math

# Simple QA Original Prompt
SIMPLE_QA_P_TRUE_QA_PROMPT = """
Answer the following question using a succinct (at most one sentence) and full answer, here is the question:
{question}
""".strip()

# Simple QA Original Prompt
MMLU_PRO_P_TRUE_QA_PROMPT = """
Answer the following question using a succinct (at most one sentence) and full answer, here is the question:
{question}
""".strip()

# Simple QA Original Prompt
NQ_OPEN_P_TRUE_QA_PROMPT = """
Answer the following question using a succinct (at most one sentence) and full answer, here is the question:
{question}
""".strip()

# Simple QA Original Prompt
POP_QA_P_TRUE_QA_PROMPT = """
Answer the following question using a succinct (at most one sentence) and full answer, here is the question:
{question}
""".strip()

P_TRUE_SELF_EVALUATION_PROMPT = """
Question: {question}
Proposed Answer: {proposed_answer}
Is the proposed answer:
(A) True
(B) False
The proposed answer is: 

Output either A or B with no other text around it.
""".strip()


class PTrueConfidenceExtractor():
    def __init__(self, confidence_extraction_method_cfg, qa_model_cfg):
        self.confidence_extraction_method_cfg = confidence_extraction_method_cfg
        self.qa_model_cfg = qa_model_cfg
        self.qa_model = self.get_qa_model(self.qa_model_cfg)


    def get_qa_model(self, qa_model_cfg):
        return LLM(qa_model_cfg)
    

    def __call__(self, dataset: MMLUProDataset | SimpleQADataset, pre_runned_batch_info: DictConfig):
        qa_batch_job_id = pre_runned_batch_info.qa_batch_id
        grader_batch_job_id = pre_runned_batch_info.grader_batch_id
        try:
            p_true_self_eval_batch_job_id = pre_runned_batch_info.p_true_self_eval_batch_id
        except:
            p_true_self_eval_batch_job_id = None
        
        task_model_name = self.qa_model_cfg.name.split("/")[-1] if "/" in self.qa_model_cfg.name else self.qa_model_cfg.name

        if dataset.name == "simple_qa" or dataset.name == "mini_simple_qa":
            qa_responses = self.generate_qa_responses(dataset.df, self.confidence_extraction_method_cfg, task_name=f"simple_qa_{task_model_name}_ptrue_qa", qa_batch_job_id=qa_batch_job_id)
            # combine qa_responses and dataset_df
            response_df = dataset.df.copy()
            response_df["responses"] = qa_responses 
            response_df["confidences"] = self.generate_self_eval_confidence(response_df, task_name=f"simple_qa_{task_model_name}_ptrue_self_eval", p_true_self_eval_batch_job_id=p_true_self_eval_batch_job_id)
            # grade the accuracy of the confidence scores
            accuracies = dataset.grade_responses(response_df["responses"], grader_batch_job_id=grader_batch_job_id, task_name=f"simple_qa_{task_model_name}_ptrue_grader")
            response_df["accuracies"] = accuracies 
        elif dataset.name == "mmlu_pro" or dataset.name == "mini_mmlu_pro":
            qa_responses = self.generate_qa_responses(dataset.df, self.confidence_extraction_method_cfg, task_name=f"mmlu_pro_{task_model_name}_ptrue_qa", qa_batch_job_id=qa_batch_job_id)
            # combine qa_responses and dataset_df
            response_df = dataset.df.copy()
            response_df["responses"] = qa_responses 
            response_df["confidences"] = self.generate_self_eval_confidence(response_df, task_name=f"mmlu_pro_{task_model_name}_ptrue_self_eval", p_true_self_eval_batch_job_id=p_true_self_eval_batch_job_id)
            # grade the accuracy of the confidence scores
            accuracies = dataset.grade_responses(response_df["responses"], grader_batch_job_id=grader_batch_job_id, task_name=f"mmlu_pro_{task_model_name}_ptrue_grader")
            response_df["accuracies"] = accuracies  
        elif dataset.name == "nq_open" or dataset.name == "mini_nq_open":
            qa_responses = self.generate_qa_responses(dataset.df, self.confidence_extraction_method_cfg, task_name=f"nq_open_{task_model_name}_ptrue_qa", qa_batch_job_id=qa_batch_job_id)
            # combine qa_responses and dataset_df
            response_df = dataset.df.copy()
            response_df["responses"] = qa_responses 
            response_df["confidences"] = self.generate_self_eval_confidence(response_df, task_name=f"nq_open_{task_model_name}_ptrue_self_eval", p_true_self_eval_batch_job_id=p_true_self_eval_batch_job_id)
            # grade the accuracy of the confidence scores
            accuracies = dataset.grade_responses(response_df["responses"], grader_batch_job_id=grader_batch_job_id, task_name=f"nq_open_{task_model_name}_ptrue_grader")
            response_df["accuracies"] = accuracies 
        elif dataset.name == "pop_qa" or dataset.name == "mini_pop_qa":
            qa_responses = self.generate_qa_responses(dataset.df, self.confidence_extraction_method_cfg, task_name=f"pop_qa_{task_model_name}_ptrue_qa", qa_batch_job_id=qa_batch_job_id)
            # combine qa_responses and dataset_df
            response_df = dataset.df.copy()
            response_df["responses"] = qa_responses 
            response_df["confidences"] = self.generate_self_eval_confidence(response_df, task_name=f"pop_qa_{task_model_name}_ptrue_self_eval", p_true_self_eval_batch_job_id=p_true_self_eval_batch_job_id)
            # grade the accuracy of the confidence scores
            accuracies = dataset.grade_responses(response_df["responses"], grader_batch_job_id=grader_batch_job_id, task_name=f"pop_qa_{task_model_name}_ptrue_grader")
            response_df["accuracies"] = accuracies 
        else:
            raise ValueError(f"Invalid dataset name: {dataset.name}")
        # return the response_df
        return response_df
        

    def generate_qa_responses(self, dataset_df: pd.DataFrame, confidence_extraction_method_cfg: DictConfig, task_name: str, qa_batch_job_id: str = None) -> list[str]:
        if "simple_qa" in task_name:
            # prepare prompts
            prompt_template = SIMPLE_QA_P_TRUE_QA_PROMPT
            qa_prompts = [prompt_template.format(question=row["problem"]) for _, row in dataset_df.iterrows()]
            # generate responses
            responses = self.qa_model(qa_prompts, task_name=task_name, batch_job_id=qa_batch_job_id)
            # post-process the responses if needed
            return responses
        elif "mmlu_pro" in task_name:
            # prepare prompts
            prompt_template = MMLU_PRO_P_TRUE_QA_PROMPT
            qa_prompts = [prompt_template.format(question=row["problem"]) for _, row in dataset_df.iterrows()]
            # generate responses
            responses = self.qa_model(qa_prompts, task_name=task_name, batch_job_id=qa_batch_job_id)
            # post-process the responses if needed
            return responses
        elif "nq_open" in task_name:
            # prepare prompts
            prompt_template = NQ_OPEN_P_TRUE_QA_PROMPT
            qa_prompts = [prompt_template.format(question=row["problem"]) for _, row in dataset_df.iterrows()]
            # generate responses
            responses = self.qa_model(qa_prompts, task_name=task_name, batch_job_id=qa_batch_job_id)
            # post-process the responses if needed
            return responses
        elif "pop_qa" in task_name:
            # prepare prompts
            prompt_template = POP_QA_P_TRUE_QA_PROMPT
            qa_prompts = [prompt_template.format(question=row["problem"]) for _, row in dataset_df.iterrows()]
            # generate responses
            responses = self.qa_model(qa_prompts, task_name=task_name, batch_job_id=qa_batch_job_id)
            # post-process the responses if needed
            return responses
        else:
            raise ValueError("Invalid task name.")


    def generate_self_eval_confidence(self, response_df: pd.DataFrame, task_name: str, p_true_self_eval_batch_job_id: str = None):
        prompts = [
            P_TRUE_SELF_EVALUATION_PROMPT.format(question=question, proposed_answer=response)
            for question, response in zip(response_df["problem"], response_df["responses"])
        ]
        tokens, logprobs = self.qa_model.model.get_output_token_logprob(
            prompts, task_name=task_name, batch_job_id=p_true_self_eval_batch_job_id
        )

        true_tokens = {"A", "(A", " (A", "True", "TRUE", "true", " A", " True", " TRUE", " true"}
        true_token_confidence = [None] * len(prompts)

        for i, (token_list, logprob_list) in enumerate(zip(tokens, logprobs)):
            for j, tk in enumerate(token_list):
                if tk in true_tokens:
                    true_token_confidence[i] = math.exp(logprob_list[j])
                    break  # stop at first match

        return true_token_confidence
