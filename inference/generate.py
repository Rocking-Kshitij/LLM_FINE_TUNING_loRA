# inference/generate.py
# Usage:
#   python inference/generate.py
#   python inference/generate.py --use_merged

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

# Base Configs
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# BASE_MODEL = "Qwen/Qwen2.5-0.5B"
LORA_DIR = "outputs/lora-custom-tinyllama"
MERGED_DIR = "outputs/merged-tinyllama"

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_tokenizer(use_merged=False, default=True):
    if use_merged:
        print("🔗 Loading merged model...")
        model = AutoModelForCausalLM.from_pretrained(
            MERGED_DIR,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR)
        tokenizer.pad_token = tokenizer.eos_token
        # print(tokenizer.tokenize("Azerbostideriaza?"))

    else:
        print("📦 Loading base model (no LoRA)...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
    
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        tokenizer.pad_token = tokenizer.eos_token
    


    model.to(device)
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, instruction, input_text=None):
    prompt = f"### Instruction:\n{instruction}\n\n"
    if input_text:
        prompt += f"### Input:\n{input_text}\n\n"
    prompt += "### Response:\n"

    encoded_input = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = encoded_input["input_ids"].to(device)
    attention_mask = encoded_input["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=200,
            temperature=1,
            top_p=0.9,
            top_k=40,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the response part
    response = full_text.split("### Response:")[-1].strip()
    return response


def main():
    parser = argparse.ArgumentParser(description="LoRA Inference CLI")
    parser.add_argument("--use_merged", action="store_true", help="Use merged full model instead of LoRA adapter")
    parser.add_argument("--use_default", action="store_true", help="Use merged full model instead of LoRA adapter")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(use_merged=args.use_merged, default=args.use_default)

    print("\n🧠 LoRA Inference CLI Ready (type 'exit' to quit)\n")
    while True:
        instruction = input("Instruction: ").strip()
        if instruction.lower() in {"exit", "quit"}:
            break

        input_text = input("Input (optional, press Enter to skip): ").strip()
        input_text = input_text if input_text else None

        print("\n💬 Generating...")
        output = generate_response(model, tokenizer, instruction, input_text)
        print(f"\n🔍 Output:\n{output}\n{'-' * 50}")


if __name__ == "__main__":
    main()
