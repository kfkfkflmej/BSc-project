import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSequenceClassification
import tqdm
import os
from ..models import LLM
from omegaconf import DictConfig, OmegaConf, ListConfig, MissingMandatoryValue

sample_times=10

prompt_template="""
Answer the following question using a succinct (at most one sentence) and full answer.

Question: {question}
Answer:
""".strip()


class SemanticUncertaintyExtractor():
    def __init__(self, confidence_extraction_method_cfg, qa_model_cfg):
        self.confidence_extraction_method_cfg = confidence_extraction_method_cfg
        self.qa_model_cfg = qa_model_cfg
        self.qa_model = self.get_qa_model(self.qa_model_cfg)
        
    def get_qa_model(self, qa_model_cfg):
        return LLM(qa_model_cfg)
    
    def __call__(self, dataset, pre_runned_batch_info: DictConfig):
        qa_batch_job_id = pre_runned_batch_info.qa_batch_id
        grader_batch_job_id = pre_runned_batch_info.grader_batch_id
        semantic_ids_path = pre_runned_batch_info.semantic_ids_path
        
        task_model_name = self.qa_model_cfg.name.split("/")[-1] if "/" in self.qa_model_cfg.name else self.qa_model_cfg.name
        
        qa_responses = self.generate_qa_responses(dataset.df, self.confidence_extraction_method_cfg, task_name=f"{dataset.name}_{task_model_name}_lc_qa", qa_batch_job_id=qa_batch_job_id)
        # combine qa_responses and dataset_df
        response_df = dataset.df.copy()
        for i in range(self.confidence_extraction_method_cfg.sample_times):
            response_df[f"response_{i}"] = qa_responses[i]
            response_df[f"response_{i}"] = response_df[f"response_{i}"].fillna("")
        # semantic uncertainty estimation
        if os.path.exists(semantic_ids_path):
            semantic_ids_list_list = np.load(semantic_ids_path)
        else:
            semantic_ids_list_list = self.get_semantic_ids_for_response_df(response_df)
            np.save(semantic_ids_path, semantic_ids_list_list)
        for i in range(self.confidence_extraction_method_cfg.sample_times):
            response_df[f"semantic_id_{i}"] = semantic_ids_list_list[i]
            # confidence estimation
        confidences, predictions = self.calculate_confidence_and_predictions(response_df)
        response_df["confidences"] = confidences
        response_df["responses"] = predictions
        # grade the accuracy of the confidence scores
        response_df["level"] = response_df["confidences"].apply(self.transform_conf)
        return response_df

    
    def generate_qa_responses(self, dataset_df: pd.DataFrame, confidence_extraction_method_cfg: DictConfig, task_name: str, qa_batch_job_id: str = None) -> list[str]:
        if self.uncertainty == False:
            prompt_template = prompt_template
        else:
            raise ValueError(f"Invalid qa template: {confidence_extraction_method_cfg.qa_template}")
        qa_prompts = [prompt_template.format(question=row["problem"]) for _, row in dataset_df.iterrows()]
        # generate responses
        responses = self.qa_model(qa_prompts, task_name=task_name, batch_job_id=qa_batch_job_id)
        # post-process the responses if needed
        return responses
    

    def calculate_confidence_and_predictions(self, response_df: pd.DataFrame):
        confidences = []
        predictions = []
        for idx, row in response_df.iterrows():
            semantic_ids_for_one_question = [row[f"semantic_id_{i}"] for i in range(self.confidence_extraction_method_cfg.sample_times)]
            # find the most frequent semantic id
            most_frequent_semantic_id = max(set(semantic_ids_for_one_question), key=semantic_ids_for_one_question.count)
            confidences.append(semantic_ids_for_one_question.count(most_frequent_semantic_id) / len(semantic_ids_for_one_question))
            index_of_most_frequent_semantic_id = semantic_ids_for_one_question.index(most_frequent_semantic_id)
            # prediction is the response with the most frequent semantic id
            predictions.append(response_df.at[idx, f"response_{index_of_most_frequent_semantic_id}"])
        return confidences, predictions
    
    
    def get_semantic_ids_for_response_df(self, response_df: pd.DataFrame, strict_entailment=False) -> list[list[int]]:
        semantic_ids_list_list = []
        for i in range(self.confidence_extraction_method_cfg.sample_times):
            semantic_ids_list_list.append([])
        for idx, row in tqdm(response_df.iterrows(), total=len(response_df), desc="Getting semantic ids"):
            response_set = [row[f"response_{i}"] for i in range(self.confidence_extraction_method_cfg.sample_times)]
            semantic_ids = self.get_semantic_ids_for_response_set(response_set, strict_entailment)
            for i in range(self.confidence_extraction_method_cfg.sample_times):
                semantic_ids_list_list[i].append(semantic_ids[i])
        return semantic_ids_list_list

    def get_semantic_ids_for_response_set(self, response_set: list[str], strict_entailment=False):
        """Group list of predictions into semantic meaning."""   
        def are_equivalent_batch(text1_list: list[str], text2_list: list[str], strict_entailment: bool = False):
            assert len(text1_list) == len(text2_list), "Both lists must have the same length"

            # batch inference
            test1_list_concat_test2_list = text1_list + text2_list
            test2_list_concat_test1_list = text2_list + text1_list
            implication = self.entailment_model.check_implication_batch(test1_list_concat_test2_list, test2_list_concat_test1_list)
            implication_1 = implication[0:len(text1_list)]
            implication_2 = implication[len(text1_list):]
            # implication_1 = self.entailment_model.check_implication_batch(text1_list, text2_list)
            # implication_2 = self.entailment_model.check_implication_batch(text2_list, text1_list)

            results = []
            i = 0
            for i1, i2 in zip(implication_1, implication_2):
                assert (i1 in [0, 1, 2]) and (i2 in [0, 1, 2])

                if strict_entailment:
                    semantically_equivalent = (i1 == 2) and (i2 == 2)
                else:
                    implications = [i1, i2]
                    semantically_equivalent = (0 not in implications) and ([1, 1] != implications)

                results.append((text1_list[i], text2_list[i], semantically_equivalent))
                i += 1
            return results
        
        def retrieve_equivalent_results(are_equivalent_batch_results, t1, t2):
            for item in are_equivalent_batch_results:
                if item[0] == t1 and item[1] == t2:
                    return item[2]

        # build list of pairs
        text_pair_left = []
        text_pair_right = []
        for i in range(len(response_set)):
            for j in range(i+1, len(response_set)):
                text_pair_left.append(response_set[i])
                text_pair_right.append(response_set[j])
                
                
        # are_equivalent_batch
        are_equivalent_batch_results = are_equivalent_batch(text_pair_left, text_pair_right, strict_entailment)
    
        # Initialise all ids with -1.
        semantic_set_ids = [-1] * len(response_set)
        # Keep track of current id.
        next_id = 0
        for i, string1 in enumerate(response_set):
            # Check if string1 already has an id assigned.
            if semantic_set_ids[i] == -1:
                # If string1 has not been assigned an id, assign it next_id.
                semantic_set_ids[i] = next_id
                for j in range(i+1, len(response_set)):
                    # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                    if retrieve_equivalent_results(are_equivalent_batch_results, string1, response_set[j]):
                        semantic_set_ids[j] = next_id
                next_id += 1

        assert -1 not in semantic_set_ids

        return semantic_set_ids

    
    def transform_conf(self, sem_un_score):
        if sem_un_score > 0.8:
            conf_level = "high level of confidence"
        elif sem_un_score > 0.6:
            conf_level = "moderate level of confidence"
        elif sem_un_score > 0.4:
            conf_level = "low level of confidence"
        elif sem_un_score > 0.2:
            conf_level = "lowest level of confidence"
        else:
            conf_level = "complete uncertainty, reject to reply"

        return conf_level



class EntailmentDeberta():
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-large-mnli").to(device)

    def check_implication(self, text1, text2):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(device)
        # The model checks if text1 -> text2, i.e. if text2 follows from text1.
        # check_implication('The weather is good', 'The weather is good and I like you') --> 1
        # check_implication('The weather is good and I like you', 'The weather is good') --> 2
        outputs = self.model(**inputs)
        logits = outputs.logits
        # Deberta-mnli returns `neutral` and `entailment` classes at indices 1 and 2.
        largest_index = torch.argmax(F.softmax(logits, dim=1))  # pylint: disable=no-member
        prediction = largest_index.cpu().item()
        return prediction

    def check_implication_batch(
        self,
        texts1,
        texts2,
        batch_size: int = 256,
        max_length: int = 256,
    ):
        """
        Batched inference for pairs (premise -> hypothesis).

        Args:
            texts1: List[str] of premises.
            texts2: List[str] of hypotheses (same length as texts1).
            batch_size: Mini-batch size to control memory usage.
            max_length: Truncation length for the tokenizer.
            return_probs: If True, also return entailment probabilities.

        Returns:
            preds: List[int] with MNLI class indices for each pair
                   (0=contradiction, 1=neutral, 2=entailment).
        """
        preds = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            n = len(texts1)
            for i in range(0, n, batch_size):
                batch1 = texts1[i : i + batch_size]
                batch2 = texts2[i : i + batch_size]

                # Tokenize a batch of (premise, hypothesis) pairs.
                enc = self.tokenizer(
                    batch1,
                    batch2,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(device)

                # Forward pass: logits shape [B, 3]
                logits = self.model(**enc).logits

                # Convert logits to probabilities over classes.
                probs = F.softmax(logits, dim=1)   # 0:contra, 1:neutral, 2:entail

                # Predicted class indices.
                pred = torch.argmax(probs, dim=1)  # shape [B]

                preds.extend(pred.cpu().tolist())

        return preds