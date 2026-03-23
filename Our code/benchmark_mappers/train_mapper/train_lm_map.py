import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
import pandas as pd
import pickle
import argparse
import numpy as np

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
class UncertaintyRegressor(nn.Module):
    def __init__(self, model_name='distilroberta-base'):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.reg_head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 1),
            nn.Sigmoid()  # output range is [0, 1]
        )

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = output.last_hidden_state[:, 0]
        return self.reg_head(cls_hidden).squeeze(-1)


def main(args):
    # === initialize ===
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = UncertaintyRegressor(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with open(args.train_set_path, "rb") as f:
        train_set = pickle.load(f)  # load linguistic confidence llm annotation data
        
    dataset = UncertaintyDataset(train_set, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    mse_loss = nn.MSELoss()


    test_set_df = pd.read_csv(args.test_set_path)
    test_set_df['annotation_mean'] = test_set_df['annotation_mean'] / 100
    test_sentences = test_set_df['uncertainty_expression'].tolist()


    test_inputs = tokenizer(test_sentences, return_tensors="pt", truncation=True, padding=True, max_length=args.max_len)
    test_inputs = {k: v.to(device) for k, v in test_inputs.items()}

    best_test_loss = float('inf')
    

    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.reg_head.parameters():
        param.requires_grad = True

    # === training ===
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for index, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            scores = batch["score"].to(device)

            optimizer.zero_grad()
            preds = model(input_ids, attention_mask)
            
            loss = mse_loss(preds, scores)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if index % 100 == 0:
                print(f"Batch {index} - Loss: {total_loss/(index+1):.4f}")


        model.eval()

        with torch.no_grad():
            pred_scores = model(test_inputs["input_ids"], test_inputs["attention_mask"]).squeeze().cpu().numpy()
        test_loss = mse_loss(torch.tensor(pred_scores, dtype=torch.float32), torch.tensor((test_set_df['annotation_mean']).to_numpy(), dtype=torch.float32))
        print(f"Epoch {epoch+1} - Train Loss: {total_loss/len(dataloader):.4f} - Test Loss: {test_loss:.4f} {'(new best loss)' if test_loss < best_test_loss else ''}")
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.reg_head.state_dict(), args.save_path)
            test_set_df['confidence_score_probe_llm_anno_trained'] = pred_scores
            test_set_df.to_csv(args.save_path.replace(".pth", ".csv"), index=False)
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="distilroberta-base", help="model name")
    parser.add_argument("--max_len", type=int, default=128, help="max length of input text")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--train_set_path", type=str, default="train_mapper/datasets/linguistic_confidence_human_anno_training_set.pkl", help="training set path")
    parser.add_argument("--test_set_path", type=str, default="train_mapper/datasets/linguistic_confidence_benchmark.csv", help="test set path")
    parser.add_argument("--save_path", type=str, default="mapper_weights/human_anno_trained_reg_head.pth", help="save path")
    args = parser.parse_args()
    main(args)