import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType

# Config
#MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# DATA_PATH = "data/alpaca_cleaned.json"
# OUTPUT_DIR = "outputs/lora-tinyllama"

DATA_PATH = "data/custom_data.json"
OUTPUT_DIR = "outputs/lora-custom-tinyllama"
# OUTPUT_DIR = "outputs/lora-custom-tinyqwen"

BATCH_SIZE = 8
NUM_EPOCHS = 200
LEARNING_RATE = 1e-5
MAX_SEQ_LENGTH = 512

#  Define TokenizerWrapper to use as processing_class
class TokenizerWrapper(PreTrainedTokenizerBase):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def pad(self, *args, **kwargs):
        return self.tokenizer.pad(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

def main():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token for consistency

    new_tokens = ["Azerbostideriaza"]
    tokenizer.add_tokens(new_tokens)

    # Load dataset
    raw_dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    # Tokenize dataset
    def tokenize(example):
        prompt = f"### Instruction:\n{example['instruction']}\n\n"
        if example["input"]:
            prompt += f"### Input:\n{example['input']}\n\n"
        prompt += f"### Response:\n{example['output']}"

        tokens = tokenizer(prompt, truncation=True, padding="max_length", max_length=MAX_SEQ_LENGTH)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    print("📚 Tokenizing dataset...")
    tokenized_dataset = raw_dataset.map(tokenize, remove_columns=raw_dataset.column_names)

    #  Load model for GPU (or CPU fallback if needed)
    print(" Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    model.resize_token_embeddings(len(tokenizer))

    #  Apply LoRA
    print(" Applying LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=512,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)


    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        bf16=False,
        fp16=True if device == "cuda" else False,
    )

    #  Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    #  Wrap tokenizer for processing_class
    processing_class = TokenizerWrapper(tokenizer)

    #  Trainer
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    )

    #  Start training
    print("🚀 Starting training...")
    trainer.train()

    #  Save model
    print("💾 Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(" Training complete!")

if __name__ == "__main__":
    main()
