import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset
import json
from config.model_config import MODEL_CONFIG

def setup_model():
    """Initialize model with 4-bit quantization for DGX Spark efficiency"""
    
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIG["base_model"],
        quantization_config=bnb_config,
        device_map="auto",  # Automatically distribute across GPU/CPU
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["base_model"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # Add LoRA adapters
    lora_config = LoraConfig(
        r=MODEL_CONFIG["lora_r"],
        lora_alpha=MODEL_CONFIG["lora_alpha"],
        target_modules=MODEL_CONFIG["target_modules"],
        lora_dropout=MODEL_CONFIG["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    return model, tokenizer

def format_dataset(training_data: list):
    """Format data for instruction fine-tuning"""
    formatted_data = []
    
    for item in training_data:
        # Create conversation format
        conversation = [
            {"role": "system", "content": "You are a fact-checking assistant specializing in Canadian parliamentary debates. Verify claims using reliable sources and provide evidence-based assessments."},
            {"role": "user", "content": item['prompt']},
            {"role": "assistant", "content": item['completion']}
        ]
        
        formatted_data.append({
            "text": tokenizer.apply_chat_template(conversation, tokenize=False)
        })
    
    return Dataset.from_list(formatted_data)

def train_model(training_data: list):
    """Execute fine-tuning"""
    model, tokenizer = setup_model()
    dataset = format_dataset(training_data)
    
    training_args = TrainingArguments(
        output_dir=MODEL_CONFIG["output_dir"],
        num_train_epochs=MODEL_CONFIG["num_epochs"],
        per_device_train_batch_size=MODEL_CONFIG["batch_size"],
        gradient_accumulation_steps=MODEL_CONFIG["gradient_accumulation_steps"],
        optim="paged_adamw_8bit",
        learning_rate=MODEL_CONFIG["learning_rate"],
        warmup_steps=MODEL_CONFIG["warmup_steps"],
        logging_steps=MODEL_CONFIG["logging_steps"],
        save_steps=MODEL_CONFIG["save_steps"],
        save_total_limit=3,
        fp16=False,
        bf16=True,  # Use bfloat16 on DGX Spark
        gradient_checkpointing=True,
        report_to="wandb",  # Optional: for experiment tracking
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=MODEL_CONFIG["max_length"],
    )
    
    trainer.train()
    
    # Save final model
    model.save_pretrained(f"{MODEL_CONFIG['output_dir']}/final")
    tokenizer.save_pretrained(f"{MODEL_CONFIG['output_dir']}/final")
    
    return model, tokenizer

if __name__ == "__main__":
    # Load your prepared training data
    with open('data/training_pairs.json', 'r') as f:
        training_data = json.load(f)
    
    train_model(training_data)

