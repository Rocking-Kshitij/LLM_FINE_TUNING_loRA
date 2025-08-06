\# Fine-Tuning TinyLlama-1.1B-Chat with LoRA

Traning a hypothetical word to a llm using LoRA fine tuning



This project demonstrates how to fine-tune the \[TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) language model using Low-Rank Adaptation (LoRA) on a custom dataset along with `alpaca\_cleaned.json`.



\## Overview



\- Model: `TinyLlama-1.1B-Chat-v1.0`

\- Method: Parameter-efficient fine-tuning using LoRA

\- Datasets: Custom dataset + `alpaca\_cleaned.json`

\- Objective: As an experiment we fine tuned the model to learn a word 'Azenrzostideriaza' which hypothetically means "A nerd python programmer"



\## Steps



1\. \*\*Prepare the Dataset\*\*

&nbsp;  - we create a test dataset `alpaca\_cleaned.json` and a custom dataset which contained example of the use of our word.



3\. \*\*Fine-Tuning with LoRA\*\*

&nbsp;  - Apply LoRA adapters to reduce training compute and memory requirements.

&nbsp;  - Use frameworks like `PEFT` and `transformers` for LoRA-based fine-tuning.

&nbsp;  - also created qlora for the 4bit quantized version



4\. \*\*Merge LoRA Adapters\*\*

&nbsp;  - After training, merge the LoRA weights into the base model to create a standalone fine-tuned model.



5\. \*\*Save \& Push\*\*

&nbsp;  - Save the merged model locally and compared the results.



\## Requirements



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





```bash

pip install transformers datasets peft accelerate bitsandbytes



