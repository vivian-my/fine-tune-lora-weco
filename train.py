# train.py

# ============================================================================
# CRITICAL: DO NOT CHANGE FUNCTION NAMES OR SIGNATURES
# ============================================================================
# The following functions are part of the stable API contract:
# - train_model(model: model)

# These function names and signatures MUST remain unchanged as they are
# imported and called by evaluate.py. Only modify the internal implementation.
# ============================================================================

from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments
import os 

def train_model(model, train_data, eval_data, data_collator):
    """
    Train a model on the training data and return it.
    This function will be optimized by Weco.

    IMPORTANT: This function name and signature must NOT be changed.
    Only the internal implementation should be modified.
    Args:
        model
        train_data
        data_collator

    Returns:
        trained results
    """

    # --- Using Lora for Model fine-tuning (THIS BLOCK WILL BE OPTIMIZED BY WECO) ---
    # WECO will insert/modify code here for:
    # - Hyperparameter Tuning for LoRa configure files: rank, target_set. 
    # - Hyperparameter Tuning for learning rate. 
    # -- We do not change PEFT method, we only optimize hyper-parameter searching for LoRa and learning rate. 
    # --- Example: Current Hyper-Parameters (Weco will replace/enhance this) ---
    target_set = ["q_proj","k_proj","v_proj","o_proj"] 
    rank = 1
    alpha = rank * 2 
    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules=target_set,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    lr = 0.01
    # --- End of WECO Optimizable Block ---

    run_name = f"llama1b_lora{rank}_lr{lr:g}"

    training_args = TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=0,
            num_train_epochs=1,
            learning_rate=lr,
            lr_scheduler_type="constant",
            bf16=True,
            logging_steps=10,
            save_strategy="no",
            run_name=run_name,
        )

    trainer = Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset = eval_data, 
            args=training_args,
            data_collator=data_collator,
        )
    
    trainer.train()

    return trainer


## This function aims to print the trainable parameters. Do not change or modify this function. 
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    pct = 100 * trainable_params / all_param if all_param else 0
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {pct:.4f}")

