import os
import json
import argparse
from typing import List
import torch
import torch.nn as nn
import bitsandbytes as bnb  # noqa: F401 (ensure bnb is importable if using 8-bit/4-bit later)
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from datasets import load_dataset



def build_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", device_map="auto") 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def prepare_dataset(tokenizer, eval_size=1000):
    # 1) load + format 
    ## original 50000
    total_size = 50000 + eval_size
    ds = load_dataset("allenai/tulu-3-sft-mixture", split=f"train[:{total_size}]")
    def format_chat(example):
        messages = example["messages"]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": text}
    ds_formatted = ds.map(format_chat, remove_columns=ds.column_names, num_proc=2)

    # 2) Tokenize
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=4096,
            padding=False  # dynamic padding by collator
        )
    ds_tok = ds_formatted.map(tokenize_fn, batched=True, remove_columns=["text"], num_proc=2)

    # 3) Split into train and eval
    split_ds = ds_tok.train_test_split(test_size=eval_size, seed=42)
    train_dataset = split_ds["train"]
    eval_dataset = split_ds["test"]
    # 4) Collator
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    return train_dataset, eval_dataset, collator


def generate_config_recepies(trainer):
    config_dict = {}
    config_dict['learning_rate'] = trainer.args.learning_rate
    if hasattr(trainer.model, 'peft_config'):
        peft_config = list(trainer.model.peft_config.values())[0]
        # Extract LoRA rank
        config_dict['lora_rank'] = peft_config.r
        # Extract LoRA target modules
        config_dict['lora_target_modules'] = peft_config.target_modules
    else:
        # Fallback if model doesn't have PEFT config
        config_dict['lora_rank'] = None
        config_dict['lora_target_modules'] = None
    
    eval_results = trainer.evaluate()
    config_dict['eval_loss'] = eval_results.get('eval_loss', None)
    
    run_name = f"llama1b_lora{config_dict['lora_rank']}_lr{config_dict['learning_rate']:g}" 
    output_dir = os.path.join("runs", run_name)
    os.makedirs(output_dir, exist_ok=True)

    def set_default(obj):
        if isinstance(obj, set):
            return list(obj)
        raise TypeError

    with open(output_dir + "/recepie.json", 'w') as f:
        json.dump(config_dict, indent=4, fp=f, default=set_default)

    return config_dict



def parse_args():
    p = argparse.ArgumentParser(description="LoRA SFT with CLI for LR and LoRA rank.")
    p.add_argument("--output-root", type=str, default="runs", help="Root output dir.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    # Build a single tokenizer for dataset prep (OK; model gets rebuilt per LR)
    model, shared_tokenizer = build_model_and_tokenizer()
    ds_tok, eval_data, collator = prepare_dataset(shared_tokenizer)

    from train import train_model
    assert callable(train_model), "train_model function must exist and be callable"
    
    # Train the model (this will be optimized by Weco)
    print("Training model...")
    trainer = train_model(model = model , train_data =ds_tok, eval_data = eval_data, data_collator = collator)

 
    # Print results for evaluation Loss
    print("Final Evaluation Results:")
    for key, value in trainer.evaluate().items():
        if key == "eval_loss":
            # Print results for evaluation Loss
            print(f"loss: {value:.4f}")

    ## Store the hyperparameter recepies for lora fine-tuning.  
    recepies = generate_config_recepies(trainer)
  
if __name__ == "__main__":
    main()
