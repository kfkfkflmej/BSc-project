import pandas as pd
from omegaconf import DictConfig
from ..models import LLM
import logging
import math
from concurrent.futures import ThreadPoolExecutor 

OPENAI_SYSTEM_PROMPT = "You are a helpful assistant."

SIMPLE_QA_GRADER_TEMPLATE = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k". 
    - Predicted answers "120k", "124k", and 115k" are all CORRECT. 
    - Predicted answers "100k" and "113k" are INCORRECT. 
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name. 
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip()

MMLU_PRO_GRADER_TEMPLATE = """
Your job is to look at a multiple choice question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
If the predicted answer matches the gold target option's letter or content, the grade will be "CORRECT". 
If the predicted answer does not match the gold target option's letter or content, the grade will be "INCORRECT". 
If the predicted answer punts the question, the grade will be "NOT_ATTEMPTED".

```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C" with no text around it.
""".strip()

POP_QA_GRADER_TEMPLATE = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
A gold target is a list of accepted answers. If any of the accepted answers agrees with the predicted answer, the grade will be "CORRECT".

First, I will give examples of each grade, and then you will grade a new example.

The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k". 
    - Predicted answers "120k", "124k", and 115k" are all CORRECT. 
    - Predicted answers "100k" and "113k" are INCORRECT. 
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name. 
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letter "A", "B", or "C", with no text around it.
""".strip()


def load_dataset(dataset_cfg: DictConfig):
    if dataset_cfg.name == "simple_qa" or dataset_cfg.name == "mini_simple_qa":
        return SimpleQADataset(dataset_cfg)
    elif dataset_cfg.name == "mmlu_pro" or dataset_cfg.name == "mini_mmlu_pro":
        return MMLUProDataset(dataset_cfg)
    elif dataset_cfg.name == "nq_open" or dataset_cfg.name == "mini_nq_open":
        return NQOPENDataset(dataset_cfg)
    elif dataset_cfg.name == "pop_qa" or dataset_cfg.name == "mini_pop_qa":
        return PopQADataset(dataset_cfg)
    if dataset_cfg.name == "simple_qa_finetune" or dataset_cfg.name == "mini_simple_qa_finetune":
        return SimpleQAFinetuneDataset(dataset_cfg)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_cfg.name}")


class SimpleQADataset():
    def __init__(self, dataset_cfg: DictConfig):
        self.name = dataset_cfg.name
        self.dataset_cfg = dataset_cfg
        self.df = pd.read_csv(dataset_cfg.file_path)
        self.grader_model = LLM(dataset_cfg.grader_model)

    def get_dataset(self):
        return self.df

    def grade_responses(self, responses: list[str], grader_batch_job_id: str = None, task_name: str = None) -> list[str]:
        # return correct, incorrect, or not attempted
        # responses is a list of strings, each string is a response to a question, in the same order as the questions in df
        # return a list of strings, each string is the grade for the corresponding response
        # the grade is either "correct", "incorrect", or "not attempted"
        #########################################################
        # prepare request for batch evaluation
        #########################################################
        prompts = []
        for idx, response in enumerate(responses):
            prompts.append(SIMPLE_QA_GRADER_TEMPLATE.format(
                question=self.df.iloc[idx]["problem"], target=self.df.iloc[idx]["answer"], predicted_answer=response))
        grader_results = self.grader_model(
            prompts, task_name=task_name, batch_job_id=grader_batch_job_id)
        if any(grader_result not in ["A", "B", "C", None] for grader_result in grader_results):
            logging.warning("Some grade results are invalid, not in [A, B, C]")
            # exit(1)

        result_map = {
            "A": "CORRECT",
            "B": "INCORRECT",
            "C": "NOT_ATTEMPTED",
            None: "NOT_ATTEMPTED"
        }
        results_list = [result_map.get(grader_result, "NOT_ATTEMPTED")
                        for grader_result in grader_results]
        return results_list



class MMLUProDataset():
    def __init__(self, dataset_cfg: DictConfig):
        self.name = dataset_cfg.name
        self.dataset_cfg = dataset_cfg
        self.df = pd.read_pickle(dataset_cfg.file_path)
        self.grader_model = LLM(dataset_cfg.grader_model)

    def get_dataset(self):
        return self.df

    def grade_responses(self, responses: list[str], grader_batch_job_id: str = None, task_name: str = None) -> list[str]:
        # return correct, incorrect, or not attempted
        # responses is a list of strings, each string is a response to a question, in the same order as the questions in df
        # return a list of strings, each string is the grade for the corresponding response
        # the grade is either "correct", "incorrect", or "not attempted"
        #########################################################
        # prepare request for batch evaluation
        #########################################################
        if len(responses) < 50000:
            prompts = []
            for idx, response in enumerate(responses):
                target = self.df.iloc[idx]["answer"] + ") " + self.df.iloc[idx]["options"][self.df.iloc[idx]["answer_index"]]
                prompts.append(MMLU_PRO_GRADER_TEMPLATE.format(
                    question=self.df.iloc[idx]["problem"], target=target, predicted_answer=response))
            grader_results = self.grader_model(prompts, task_name=task_name, batch_job_id=grader_batch_job_id)
            if any(grader_result not in ["A", "B", "C", None] for grader_result in grader_results):
                logging.warning("Some grade results are invalid, not in [A, B, C, None]")
                # exit(1)
            result_map = {
                "A": "CORRECT",
                "B": "INCORRECT",
                "C": "NOT_ATTEMPTED",
                None: "NOT_ATTEMPTED"
            }
            results_list = [result_map.get(grader_result, "NOT_ATTEMPTED")
                            for grader_result in grader_results]
            return results_list
        else:
            prompts = []
            for idx, response in enumerate(responses):
                target = (
                    self.df.iloc[idx]["answer"] + ") " +
                    self.df.iloc[idx]["options"][self.df.iloc[idx]["answer_index"]]
                )
                prompts.append(MMLU_PRO_GRADER_TEMPLATE.format(
                    question=self.df.iloc[idx]["problem"],
                    target=target,
                    predicted_answer=response
                ))

            # number of batches
            num_batches = 4
            batch_size = math.ceil(len(prompts) / num_batches)

            if grader_batch_job_id is None:
                grader_batch_job_id = [None] * num_batches

            def run_batch(i, batch_prompts, job_id):
                return self.grader_model(
                    batch_prompts,
                    task_name=task_name + f"_grader_batch_{i}",
                    batch_job_id=job_id
                )

            batches = [
                (i, prompts[i*batch_size:(i+1)*batch_size], grader_batch_job_id[i])
                for i in range(num_batches)
                if prompts[i*batch_size:(i+1)*batch_size]
            ]

            grader_results = []
            with ThreadPoolExecutor(max_workers=num_batches) as executor:
                futures = [executor.submit(run_batch, i, batch, job_id) for i, batch, job_id in batches]
                for f in futures:
                    grader_results.extend(f.result())
            result_map = {
                "A": "CORRECT",
                "B": "INCORRECT",
                "C": "NOT_ATTEMPTED",
                None: "NOT_ATTEMPTED"
            }
            results_list = [result_map.get(grader_result, "NOT_ATTEMPTED")
                            for grader_result in grader_results]
            return results_list
        
    

class NQOPENDataset():
    def __init__(self, dataset_cfg: DictConfig):
        self.name = dataset_cfg.name
        self.dataset_cfg = dataset_cfg
        self.df = pd.read_csv(dataset_cfg.file_path)
        self.grader_model = LLM(dataset_cfg.grader_model)

    def get_dataset(self):
        return self.df

    def grade_responses(self, responses: list[str], grader_batch_job_id: str = None, task_name: str = None) -> list[str]:
        # return correct, incorrect, or not attempted
        # responses is a list of strings, each string is a response to a question, in the same order as the questions in df
        # return a list of strings, each string is the grade for the corresponding response
        # the grade is either "correct", "incorrect", or "not attempted"
        #########################################################
        # prepare request for batch evaluation
        #########################################################
        prompts = []
        for idx, response in enumerate(responses):
            prompts.append(SIMPLE_QA_GRADER_TEMPLATE.format(
                question=self.df.iloc[idx]["problem"], target=self.df.iloc[idx]["answer"], predicted_answer=response))
        grader_results = self.grader_model(
            prompts, task_name=task_name, batch_job_id=grader_batch_job_id)
        if any(grader_result not in ["A", "B", "C", None] for grader_result in grader_results):
            logging.warning("Some grade results are invalid, not in [A, B, C]")
            # exit(1)

        result_map = {
            "A": "CORRECT",
            "B": "INCORRECT",
            "C": "NOT_ATTEMPTED",
            None: "NOT_ATTEMPTED"
        }
        results_list = [result_map.get(grader_result, "NOT_ATTEMPTED")
                        for grader_result in grader_results]
        return results_list
    


class PopQADataset():
    def __init__(self, dataset_cfg: DictConfig):
        self.name = dataset_cfg.name
        self.dataset_cfg = dataset_cfg
        self.df = pd.read_csv(dataset_cfg.file_path)
        self.grader_model = LLM(dataset_cfg.grader_model)

    def get_dataset(self):
        return self.df

    def grade_responses(self, responses: list[str], grader_batch_job_id: str = None, task_name: str = None) -> list[str]:
        # return correct, incorrect, or not attempted
        # responses is a list of strings, each string is a response to a question, in the same order as the questions in df
        # return a list of strings, each string is the grade for the corresponding response
        # the grade is either "correct", "incorrect", or "not attempted"
        #########################################################
        # prepare request for batch evaluation
        #########################################################
        prompts = []
        for idx, response in enumerate(responses):
            prompts.append(POP_QA_GRADER_TEMPLATE.format(
                question=self.df.iloc[idx]["problem"], target=self.df.iloc[idx]["answer"], predicted_answer=response))
        grader_results = self.grader_model(
            prompts, task_name=task_name, batch_job_id=grader_batch_job_id)
        if any(grader_result not in ["A", "B", "C", None] for grader_result in grader_results):
            logging.warning("Some grade results are invalid, not in [A, B, C]")
            # exit(1)

        result_map = {
            "A": "CORRECT",
            "B": "INCORRECT",
            "C": "NOT_ATTEMPTED",
            None: "NOT_ATTEMPTED"
        }
        results_list = [result_map.get(grader_result, "NOT_ATTEMPTED")
                        for grader_result in grader_results]
        return results_list