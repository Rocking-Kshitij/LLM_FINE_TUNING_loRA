# Fine-Tuning TinyLlama-1.1B-Chat with LoRA
Traning a hypothetical word to a llm using LoRA fine tuning

This project demonstrates how to fine-tune the [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) language model using Low-Rank Adaptation (LoRA) on a custom dataset along with `alpaca_cleaned.json`.

## Overview

- Model: `TinyLlama-1.1B-Chat-v1.0`
- Method: Parameter-efficient fine-tuning using LoRA
- Datasets: Custom dataset + `alpaca_cleaned.json`
- Objective: As an experiment we fine tuned the model to learn a word 'Azenrzostideriaza' which hypothetically means "A nerd python programmer"

## Steps

1. **Prepare the Dataset**
   - we create a test dataset `alpaca_cleaned.json` and a custom dataset which contained example of the use of our word.

3. **Fine-Tuning with LoRA**
   - Apply LoRA adapters to reduce training compute and memory requirements.
   - Use frameworks like `PEFT` and `transformers` for LoRA-based fine-tuning.
   - also created qlora for the 4bit quantized version

4. **Merge LoRA Adapters**
   - After training, merge the LoRA weights into the base model to create a standalone fine-tuned model.

5. **Save & Push**
   - Save the merged model locally and compared the results.

## Requirements

transformers
datasets
peft
bitsandbytes
accelerate
scikit-learn
sentencepiece
wandb

Directories
/data -> for raw data
/src -> for traning logic
/inference -> for merging and inference



Normal vanilla llm:
<img width="1604" height="479" alt="image7" src="https://github.com/user-attachments/assets/9505f5db-c77d-4064-a5ff-daddf4acb054" />


After loRA fineTuning:
<img width="1564" height="448" alt="image6" src="https://github.com/user-attachments/assets/433d5546-099b-4c6d-ac44-f0180a4399da" />


