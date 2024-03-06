import os
import argparse
import json
import torch

from datasets import Dataset, load_from_disk

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, logging

import bitsandbytes as bnb
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


# Define the create_prompt function
def create_prompt(sample):
    bos_token = "<s>"
    eos_token = "</s>"
    
    instruction = sample['instruction']
    context = sample['context']
    response = sample['response']

    text_row = f"""[INST] Below is the question based on the context. Question: {instruction}. Below is the given the context {context}. Write a response that appropriately completes the request.[/INST]"""
    answer_row = response

    sample["prompt"] = bos_token + text_row
    sample["completion"] = answer_row + eos_token

    return sample


def prepare_datatset(dataset_location):
    dataset = load_from_disk(dataset_location)
    dataset_instruct_format = dataset.map(create_prompt, remove_columns=['instruction','context','response','category'])
    return dataset_instruct_format


def parse_arge():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    parser.add_argument(
        "--model_id",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.1",
        help="Model id to use for training.",
    )
    # loading dataset from SageMaker training job instance local location 
    parser.add_argument(
        "--dataset_path", type=str, default="/opt/ml/input/data/training", help="Path to dataset."
    )
    # saving the merged model to SageMaker training job instance local location 
    parser.add_argument(
        "--model_save_path", type=str, default="/opt/ml/model", help="Path to save pretrained model."
    )
    parser.add_argument(
        "--job_output_path", type=str, default="/opt/ml/output/data", help="Path to model output artifact."
    )
    
    # add training hyperparameters for epochs, batch size, learning rate, and seed
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size to use for training.",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="Learning rate to use for training."
    )

    args, _ = parser.parse_known_args()

    return args


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        text = f"{example['prompt'][i]}\n\n ### Answer: {example['completion'][i]}"
        output_texts.append(text)
    return output_texts





def training_function(args):
    model_id = args.model_id
    
    ################################################################################
    # loading dataset from SageMaker training job instance local location
    ################################################################################
    # SageMaker copy the data from S3 to /opt/ml/input/data/training 
    dataset_location = args.dataset_path
    print(dataset_location)
    dataset_instruct_format = prepare_datatset(dataset_location)
    

    ################################################################################
    # bitsandbytes parameters and configuration, args and fixed parameters
    ################################################################################

    # Activate 4-bit precision base model loading
    use_4bit = True

    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = "bfloat16"

    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = "nf4"

    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant = True
    
    # Load the base model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )
    # Load the entire model on the GPU 0
    device_map = {"": 0}
    #device_map = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ################################################################################
    # Loading the base model from Mistral Huggingface
    ################################################################################
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map
    )

    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1

    # Load MitsralAi tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'


    ################################################################################
    # QLoRA parameters and Lora Config
    ################################################################################

    # LoRA attention dimension
    lora_r = 64

    # Alpha parameter for LoRA scaling
    lora_alpha = 16

    # Dropout probability for LoRA layers
    lora_dropout = 0.1
 

    # Set LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    ################################################################################
    # TrainingArguments parameters
    ################################################################################

    # Output directory where the model predictions and checkpoints will be stored
    # Similarly, SageMaker will copy the local output artifact to S3 
    output_dir = args.job_output_path
    
    # Number of training epochs
    num_train_epochs = args.epochs

    # Enable fp16/bf16 training (set bf16 to True with an A100)
    fp16 = False
    bf16 = False

    # Batch size per GPU for training
    per_device_train_batch_size = args.per_device_train_batch_size

    # Batch size per GPU for evaluation
    per_device_eval_batch_size = args.per_device_train_batch_size

    # Number of update steps to accumulate the gradients for
    gradient_accumulation_steps = 1

    # Enable gradient checkpointing
    gradient_checkpointing = True

    # Maximum gradient normal (gradient clipping)
    max_grad_norm = 0.3

    # Initial learning rate (AdamW optimizer)
    learning_rate = args.lr

    # Weight decay to apply to all layers except bias/LayerNorm weights
    weight_decay = 0.001

    # Optimizer to use
    optim = "paged_adamw_32bit"

    # Learning rate schedule (constant a bit better than cosine)
    lr_scheduler_type = "constant"

    # Number of training steps (overrides num_train_epochs)
    #max_steps = -1

    # Ratio of steps for a linear warmup (from 0 to learning rate)
    warmup_ratio = 0.03

    # Group sequences into batches with same length
    # Saves memory and speeds up training considerably
    group_by_length = True

    # Save checkpoint every X updates steps
    save_steps = 250

    # Log every X updates steps
    logging_steps = 25
    
    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_checkpointing=gradient_checkpointing,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        #max_steps=100, # the total number of training steps to perform
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
    )
    

    response_template = "### Answer:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    
    ################################################################################
    # SFT parameters
    ################################################################################

    # Maximum sequence length to use
    max_seq_length = 2048

    # Pack multiple short examples in the same input sequence to increase efficiency
    packing = False


    # Initialize the SFTTrainer for fine-tuning
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=dataset_instruct_format,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        peft_config=peft_config,
        max_seq_length=max_seq_length,  # You can specify the maximum sequence length here
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing
    )

    # Start the training process
    trainer.train()
    
    
    ################################################################################
    # Save the pretrained model
    ################################################################################    
    sagemaker_save_dir = args.model_save_path
    #set the name of the new model
    new_model = "Mistral-qlora-7B-Instruct-v0.1" 
    
    # Save the fine-tuned model
    trainer.model.save_pretrained(new_model)    

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map=device_map
    )
    
    merged_model= PeftModel.from_pretrained(base_model, new_model)
    
    merged_model= merged_model.merge_and_unload()
    
    merged_model.save_pretrained(sagemaker_save_dir, safe_serialization=True, max_shard_size="2GB")
    # save tokenizer for easy inference
    tokenizer.save_pretrained(sagemaker_save_dir)

def main():
    args = parse_arge()
    training_function(args)


if __name__ == "__main__":
    main()