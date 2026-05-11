import pandas as pd
import random

# --- Load data ---
bad = pd.read_csv(r"keep\bad_results.csv")
good = pd.read_csv(r"results_1_google_gemma-3-4b-it(1).csv")

# --- Shuffle + sample ---
bad_sample = bad.sample(n=20, random_state=None)
good_sample = good.sample(n=20, random_state=None)

# Add source label
bad_sample["source"] = "bad"
good_sample["source"] = "good"

# Combine
df = pd.concat([bad_sample, good_sample], ignore_index=True)

# Shuffle combined dataset
df = df.sample(frac=1).reset_index(drop=True)

# Store annotations
annotations = []

# --- CLI loop ---
indices = list(df.index)
random.shuffle(indices)

for idx in indices:
    row = df.loc[idx]

    # print("\n" + "="*80)
    # print(f"Index: {idx} | Source: {row['source']}")
    print("="*80)
    print(f"Question:\n{row['question']}\n")

    print("Responses:\n")

    for i in range(10):
        resp = row.get(f"response_{i}", "")
        sid = row.get(f"semantic_id_{i}", "")
        print(f"[ID {sid}] {resp}\n")

    # --- Annotation input ---
    while True:
        label = input("Consistent clustering? (y/n/q): ").strip().lower()
        if label in ["y", "n", "q"]:
            break

    if label == "q":
        break

    annotations.append({
        "index": idx,
        "label": label,
        "source": row["source"]
    })

# --- Save results ---
annotations_df = pd.DataFrame(annotations)
annotations_df.to_csv("annotations.csv", index=False)

print("\nSaved annotations to annotations.csv")