# src/data_preparation.py

from datasets import load_dataset
import json
from pathlib import Path

def download_and_prepare_alpaca(save_path="data/alpaca_cleaned.json"):
    # yahma/alpaca-cleaned is a Huggingface dataset
    print(" Downloading dataset from Huggingface...")
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")

    dataset = dataset.select(range(10)) 

    print(" Converting dataset to instruction format...")
    formatted = []
    for sample in dataset:
        formatted.append({
            "instruction": sample["instruction"].strip(),
            "input": sample["input"].strip(),
            "output": sample["output"].strip()
        })

    Path("data").mkdir(parents=True, exist_ok=True)

    print(f" Saving {len(formatted)} records to {save_path}")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)

    print(" Dataset preparation complete.")

if __name__ == "__main__":
    download_and_prepare_alpaca()
