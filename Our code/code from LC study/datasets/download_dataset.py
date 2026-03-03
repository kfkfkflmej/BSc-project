import pandas as pd

# splits = {'test': 'data/test-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet'}
# df = pd.read_parquet("hf://datasets/TIGER-Lab/MMLU-Pro/" + splits["test"])

df = pd.read_csv("hf://datasets/akariasai/PopQA/test.tsv", sep="\t")
df.rename(columns={"question": "problem", "possible_answers": "answer"}, inplace=True)

df.sample(1000, random_state=10).to_csv("pop_qa.csv")

mini_df = df.sample(15, random_state=10)
print(mini_df)
mini_df.sample(15, random_state=10).to_csv("mini_pop_qa.csv")