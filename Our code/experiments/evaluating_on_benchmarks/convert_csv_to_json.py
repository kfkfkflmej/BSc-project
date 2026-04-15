import pandas as pd
import json
import os

DATA_DIR = "data"   # folder where your CSVs are


def parse_choices(val):
    if pd.isna(val):
        return None
    val = str(val)

    try:
        return json.loads(val)
    except:
        pass

    try:
        return eval(val)  # only safe if you trust the data
    except:
        return None


def convert_file(csv_path):
    print(f"\nProcessing: {csv_path}")
    df = pd.read_csv(csv_path)

    if "question" not in df.columns or "answer" not in df.columns:
        print("❌ Skipping (missing required columns)")
        return

    data = {
        "question": df["question"].astype(str).tolist(),
        "answer": df["answer"].astype(str).tolist(),
    }

    # Optional: choices
    if "choices" in df.columns:
        choices = df["choices"].apply(parse_choices).tolist()
        if any(c is not None for c in choices):
            data["choices"] = choices

    # Optional: passage
    if "passage" in df.columns:
        passage = df["passage"].fillna("").astype(str).tolist()
        if any(p.strip() != "" for p in passage):
            data["passage"] = passage

    # Save JSON
    json_path = csv_path.replace(".csv", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved: {json_path} ({len(data['question'])} samples)")


def main():
    for file in os.listdir(DATA_DIR):
        if file.endswith(".csv"):
            convert_file(os.path.join(DATA_DIR, file))


if __name__ == "__main__":
    main()