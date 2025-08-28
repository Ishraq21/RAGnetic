from datasets import load_dataset
import json
from pathlib import Path

# Load Alpaca
ds = load_dataset("tatsu-lab/alpaca")

# Split into train/eval
split = ds["train"].train_test_split(test_size=0.05, seed=42)
train_data = split["train"]
eval_data = split["test"]

out_dir = Path("../data/prepared_datasets")
out_dir.mkdir(parents=True, exist_ok=True)

def save_jsonl(dataset, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in dataset:
            rec = {
                "instruction": row["instruction"],
                "input": row["input"],
                "output": row["output"]
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

save_jsonl(train_data, out_dir / "alpaca_train.jsonl")
save_jsonl(eval_data, out_dir / "alpaca_eval.jsonl")

print(f"Train size: {len(train_data)} rows")
print(f"Eval size: {len(eval_data)} rows")

print("Saved train/eval splits to", out_dir)
