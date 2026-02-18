import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
import numpy as np
import argparse
from benchmark_mappers.openai_models import GPT


DIRECT_CONFIDENCE_PROMPT = """Please provide only a confidence score between 0 and 100, 
based solely on the degree of confidence expressed in the tone of the following sentence (without using any external or prior knowledge): 
{response}
""".strip()

PAPER_DECISIVENESS_PROMPT = """You will be given a question and a proposed answer. Your task is to extract any assertions made by the proposed answer; for each assertion, you should output a
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

class UncertaintyDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=32):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoded = self.tokenizer(
            item["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "score": torch.tensor(item["score"], dtype=torch.float)
        }

# === linguistic confidence mapper model ===
class LinguisticConfidenceRegressor(nn.Module):
    def __init__(self, model_weights_path):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("distilroberta-base")
        self.reg_head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 1),
            nn.Sigmoid()  # output range is [0, 1]
        )
        self.load_state_dict(torch.load(model_weights_path))

    def forward(self, benchmark_df):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
        inputs = tokenizer(benchmark_df["uncertainty_expression"], return_tensors="pt", truncation=True, padding=True, max_length=32)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = output.last_hidden_state[:, 0]
        return self.reg_head(cls_hidden).squeeze(-1).cpu().numpy().tolist()
    
class LLMConfidenceRegressor(nn.Module):
    def __init__(self, confidence_regressor_type, model_name='gpt-5-mini'):
        super().__init__()
        self.confidence_regressor_type = confidence_regressor_type
        self.model_name = model_name
        
        if self.confidence_regressor_type == "decisiveness_mapper":
            self.prompt_template = PAPER_DECISIVENESS_PROMPT
        elif self.confidence_regressor_type == "llm_direct_prompt_mapper":
            self.prompt_template = DIRECT_CONFIDENCE_PROMPT
        self.llm = GPT(model_name=self.model_name)
        
    def forward(self, benchmark_df):
        if self.confidence_regressor_type == "decisiveness_mapper":
            prompts = [self.prompt_template.format(question=row["problem"], response=row["uncertainty_expression"]) for _, row in benchmark_df.iterrows()]
            responses = self.llm(prompts, task_name="decisiveness_mapper")
        elif self.confidence_regressor_type == "llm_direct_prompt_mapper":
            prompts = [self.prompt_template.format(response=row["uncertainty_expression"]) for _, row in benchmark_df.iterrows()]
            responses = self.llm(prompts, task_name="llm_direct_prompt_mapper")
        confidence_scores = [float(response.split(": ")[1].strip()) for response in responses]
        return confidence_scores


def main(args):
    # === initialize ===
    if args.confidence_regressor == "human_anno_mapper" or args.confidence_regressor == "llm_anno_mapper":
        model = LinguisticConfidenceRegressor(model_weights_path=args.model_weights_path)
    elif args.confidence_regressor == "decisiveness_mapper" or args.confidence_regressor == "llm_direct_prompt_mapper":
        model = LLMConfidenceRegressor(confidence_regressor_type=args.confidence_regressor, model_name=args.llm_name)
    else:
        raise ValueError(f"Invalid confidence regressor: {args.confidence_regressor}")
    
    benchmark_df = pd.read_csv(args.benchmark_set_path)
    confidence_scores = model(benchmark_df)
    mse = np.mean((confidence_scores - benchmark_df["score"]) ** 2)
    print(f"{args.confidence_regressor} MSE: {mse}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--confidence_regressor", type=str, default="linguistic", choices=["human_anno_mapper", "llm_anno_mapper", "decisiveness_mapper", "llm_direct_prompt_mapper"])
    parser.add_argument("--llm_name", type=str, default="gpt-5-mini", help="model name")
    parser.add_argument("--model_weights_path", type=str, default="mapper_weights/human_anno_trained_reg_head.pth", help="model weights path")
    parser.add_argument("--benchmark_set_path", type=str, default="benchmark_mappers/datasets/linguistic_confidence_benchmark.csv", help="benchmark set path")
    args = parser.parse_args()
    main(args)