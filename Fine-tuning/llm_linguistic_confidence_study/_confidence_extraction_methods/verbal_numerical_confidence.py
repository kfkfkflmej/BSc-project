import pandas as pd
import re
from ..datasets import MMLUProDataset, SimpleQADataset
from ..models import LLM
from omegaconf import DictConfig

# "Can Large Language Models Faithfully Express Their Intrinsic Uncertainty in Words?" vanilla prompt + Simple QA Original Prompt
SIMPLE_QA_VERBAL_NUMERICAL_CONFIDENCE_QA_PROMPT_VANILLA = """
Answer the following question using a succinct (at most one sentence) and full answer, here is the question:
{question}
Please provide a confidence score between 0 and 100 at the end of your answer in the following JSON format:
{{
"answer": Your answer here,
"confidence_score": number
}}
""".strip()

# "Can Large Language Models Faithfully Express Their Intrinsic Uncertainty in Words?" vanilla prompt + Simple QA Original Prompt
MMLU_PRO_VERBAL_NUMERICAL_CONFIDENCE_QA_PROMPT_VANILLA = """
Answer the following question using a succinct (at most one sentence) and full answer, here is the question:
{question}
Please provide a confidence score between 0 and 100 at the end of your answer in the following JSON format:
{{
"answer": Your answer here,
"confidence_score": number
}}
""".strip()

# "Can Large Language Models Faithfully Express Their Intrinsic Uncertainty in Words?" vanilla prompt + Simple QA Original Prompt
NQ_OPEN_VERBAL_NUMERICAL_CONFIDENCE_QA_PROMPT_VANILLA = """
Answer the following question using a succinct (at most one sentence) and full answer, here is the question:
{question}
Please provide a confidence score between 0 and 100 at the end of your answer in the following JSON format:
{{
"answer": Your answer here,
"confidence_score": number
}}
""".strip()

# "Can Large Language Models Faithfully Express Their Intrinsic Uncertainty in Words?" vanilla prompt + Simple QA Original Prompt
POP_QA_VERBAL_NUMERICAL_CONFIDENCE_QA_PROMPT_VANILLA = """
Answer the following question using a succinct (at most one sentence) and full answer, here is the question:
{question}
Please provide a confidence score between 0 and 100 at the end of your answer in the following JSON format:
{{
"answer": Your answer here,
"confidence_score": number
}}
""".strip()



class VerbalNumericalConfidenceExtractor():
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

        def extract_confidence_score(text: str) -> float | None:
            if text is None:
                return None
            patterns = [r'"confidence_score"\s*:\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
                        r'confidence_score\s*:\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',]
            for pat in patterns:
                match = re.search(pat, text)
                if match:
                    return float(match.group(1)) / 100 if match else None
            return None
            
        def extract_answer(text: str) -> float | None:
            if text is None:
                return ""
            patterns = [r'"answer"\s*:\s*"((?:\\.|[^"\\])*)"',
                        r'answer\s*:\s*"((?:\\.|[^"\\])*)"',
                        r'"answer"\s*:\s*([^,}\n]+)',
                        ]
            for pat in patterns:
                match = re.search(pat, text)
                if match:
                    return match.group(1)
            return ""

        if dataset.name == "simple_qa" or dataset.name == "mini_simple_qa":
            qa_responses = self.generate_qa_responses(dataset.df, self.confidence_extraction_method_cfg, task_name=f"simple_qa_{task_model_name}_vnc_qa", qa_batch_job_id=qa_batch_job_id)
            # combine qa_responses and dataset_df
            response_df = dataset.df.copy()
            response_df["raw_responses"] = qa_responses # raw responses, json format
            response_df["responses"] = [extract_answer(r) for r in qa_responses] # clean answers
            response_df["confidences"] = [extract_confidence_score(r) for r in qa_responses] # extracted confidence
            # grade the accuracy of the confidence scores
            accuracies = dataset.grade_responses(response_df["responses"], grader_batch_job_id=grader_batch_job_id, task_name=f"simple_qa_{task_model_name}_vnc_grader")
            response_df["accuracies"] = accuracies 

        elif dataset.name == "mmlu_pro" or dataset.name == "mini_mmlu_pro":
            qa_responses = self.generate_qa_responses(dataset.df, self.confidence_extraction_method_cfg, task_name=f"mmlu_pro_{task_model_name}_vnc_qa", qa_batch_job_id=qa_batch_job_id)
            # combine qa_responses and dataset_df
            response_df = dataset.df.copy()
            response_df["raw_responses"] = qa_responses # raw responses, json format
            response_df["responses"] = [extract_answer(r) for r in qa_responses] # clean answers
            response_df["confidences"] = [extract_confidence_score(r) for r in qa_responses] # extracted confidence
            # grade the accuracy of the confidence scores
            accuracies = dataset.grade_responses(response_df["responses"], grader_batch_job_id=grader_batch_job_id, task_name=f"mmlu_pro_{task_model_name}_vnc_grader")
            response_df["accuracies"] = accuracies 
        elif dataset.name == "nq_open" or dataset.name == "mini_nq_open":
            qa_responses = self.generate_qa_responses(dataset.df, self.confidence_extraction_method_cfg, task_name=f"nq_open_{task_model_name}_vnc_qa", qa_batch_job_id=qa_batch_job_id)
            # combine qa_responses and dataset_df
            response_df = dataset.df.copy()
            response_df["raw_responses"] = qa_responses # raw responses, json format
            response_df["responses"] = [extract_answer(r) for r in qa_responses] # clean answers
            response_df["confidences"] = [extract_confidence_score(r) for r in qa_responses] # extracted confidence
            # grade the accuracy of the confidence scores
            accuracies = dataset.grade_responses(response_df["responses"], grader_batch_job_id=grader_batch_job_id, task_name=f"nq_open_{task_model_name}_vnc_grader")
            response_df["accuracies"] = accuracies 
        elif dataset.name == "pop_qa" or dataset.name == "mini_pop_qa":
            qa_responses = self.generate_qa_responses(dataset.df, self.confidence_extraction_method_cfg, task_name=f"pop_qa_{task_model_name}_vnc_qa", qa_batch_job_id=qa_batch_job_id)
            # combine qa_responses and dataset_df
            response_df = dataset.df.copy()
            response_df["raw_responses"] = qa_responses # raw responses, json format
            response_df["responses"] = [extract_answer(r) for r in qa_responses] # clean answers
            response_df["confidences"] = [extract_confidence_score(r) for r in qa_responses] # extracted confidence
            # grade the accuracy of the confidence scores
            accuracies = dataset.grade_responses(response_df["responses"], grader_batch_job_id=grader_batch_job_id, task_name=f"pop_qa_{task_model_name}_vnc_grader")
            response_df["accuracies"] = accuracies 
        else:
            raise ValueError(f"Invalid dataset name: {dataset.name}")
        # return the response_df
        return response_df
        

    def generate_qa_responses(self, dataset_df: pd.DataFrame, confidence_extraction_method_cfg: DictConfig, task_name: str, qa_batch_job_id: str = None) -> list[str]:
        if "simple_qa" in task_name:
            # prepare prompts
            if confidence_extraction_method_cfg.qa_template == "vanilla":
                prompt_template = SIMPLE_QA_VERBAL_NUMERICAL_CONFIDENCE_QA_PROMPT_VANILLA
            else:
                raise ValueError(f"Invalid QA template: {confidence_extraction_method_cfg.qa_template}")
            qa_prompts = [prompt_template.format(question=row["problem"]) for _, row in dataset_df.iterrows()]
            # generate responses
            responses = self.qa_model(qa_prompts, task_name=task_name, batch_job_id=qa_batch_job_id)
            # post-process the responses if needed
            return responses
        elif "mmlu_pro" in task_name:
            # prepare prompts
            if confidence_extraction_method_cfg.qa_template == "vanilla":
                prompt_template = MMLU_PRO_VERBAL_NUMERICAL_CONFIDENCE_QA_PROMPT_VANILLA
            else:
                raise ValueError(f"Invalid QA template: {confidence_extraction_method_cfg.qa_template}")
            qa_prompts = [prompt_template.format(question=row["problem"]) for _, row in dataset_df.iterrows()]
            # generate responses
            responses = self.qa_model(qa_prompts, task_name=task_name, batch_job_id=qa_batch_job_id)
            # post-process the responses if needed
            return responses
        elif "nq_open" in task_name:
            # prepare prompts
            if confidence_extraction_method_cfg.qa_template == "vanilla":
                prompt_template = NQ_OPEN_VERBAL_NUMERICAL_CONFIDENCE_QA_PROMPT_VANILLA
            else:
                raise ValueError(f"Invalid QA template: {confidence_extraction_method_cfg.qa_template}")
            qa_prompts = [prompt_template.format(question=row["problem"]) for _, row in dataset_df.iterrows()]
            # generate responses
            responses = self.qa_model(qa_prompts, task_name=task_name, batch_job_id=qa_batch_job_id)
            # post-process the responses if needed
            return responses
        elif "pop_qa" in task_name:
            # prepare prompts
            if confidence_extraction_method_cfg.qa_template == "vanilla":
                prompt_template = POP_QA_VERBAL_NUMERICAL_CONFIDENCE_QA_PROMPT_VANILLA
            else:
                raise ValueError(f"Invalid QA template: {confidence_extraction_method_cfg.qa_template}")
            qa_prompts = [prompt_template.format(question=row["problem"]) for _, row in dataset_df.iterrows()]
            # generate responses
            responses = self.qa_model(qa_prompts, task_name=task_name, batch_job_id=qa_batch_job_id)
            # post-process the responses if needed
            return responses
        else:
            raise ValueError("Invalid task name.")