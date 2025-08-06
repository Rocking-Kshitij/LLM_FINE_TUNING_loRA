# inference/merge_adapter.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# LORA_DIR = "outputs/lora-tinyllama"
# MERGED_OUTPUT = "outputs/merged-tinyllama"

# Config
# MODEL_NAME = "Qwen/Qwen2.5-0.5B"
LORA_DIR = "outputs/lora-custom-tinyllama"
MERGED_OUTPUT = "outputs/merged-tinyllama"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("🔄 Merging LoRA adapter into base model...")

# Load adapter config
config = PeftConfig.from_pretrained(LORA_DIR)
# Load base + adapter
base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)
base_model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(base_model, LORA_DIR)

# Merge and save
model = model.merge_and_unload()
model.save_pretrained(MERGED_OUTPUT)

# Save tokenizer
tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)
tokenizer.save_pretrained(MERGED_OUTPUT)

print(f"Merged model saved to: {MERGED_OUTPUT}")
