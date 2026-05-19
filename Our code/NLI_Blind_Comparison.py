import pandas as pd
import random
import argparse
from pathlib import Path

###This script is intended for blind comparison of NLI Clustering examples from 2 different models.

def get_path(arg_value: str | None, prompt_label: str) -> str:
    """Return a valid file path from CLI arg or interactive prompt."""
    while True:
        path = arg_value if arg_value else input(f"Enter path for {prompt_label}: ").strip().strip('"')
        if Path(path).is_file():
            return path
        print(f"  File not found: {path!r}. Please try again.")
        arg_value = None  # clear so we re-prompt on next iteration

def main():
    parser = argparse.ArgumentParser(description="Blind comparison annotation tool for NLI clustering.")
    parser.add_argument("--bad", type=str, default=None, help="Path to the 'bad' model CSV")
    parser.add_argument("--good", type=str, default=None, help="Path to the 'good' model CSV")
    parser.add_argument("--output", type=str, default="annotations.csv", help="Output CSV path (default: annotations.csv)")
    parser.add_argument("--n", type=int, default=20, help="Number of samples per model (default: 20)")
    args = parser.parse_args()

    # --- Resolve paths ---
    bad_path = get_path(args.bad, "bad model CSV")
    good_path = get_path(args.good, "good model CSV")

    # --- Load data ---
    bad = pd.read_csv(bad_path)
    good = pd.read_csv(good_path)

    # --- Shuffle + sample ---
    bad_sample = bad.sample(n=args.n, random_state=None)
    good_sample = good.sample(n=args.n, random_state=None)

    # Add source label
    bad_sample["source"] = "bad"
    good_sample["source"] = "good"

    # Combine and shuffle
    df = pd.concat([bad_sample, good_sample], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)

    # Store annotations
    annotations = []

    # --- CLI loop ---
    indices = list(df.index)
    random.shuffle(indices)

    for idx in indices:
        row = df.loc[idx]

        print("=" * 80)
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
    annotations_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(annotations_df)} annotations to {args.output}")

if __name__ == "__main__":
    main()