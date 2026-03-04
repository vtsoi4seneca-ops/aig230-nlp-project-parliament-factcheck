# config/model_config.py
MODEL_CONFIG = {
    "base_model": "meta-llama/Llama-3.1-8B-Instruct",  # Or Canadian-specific if available
    "max_length": 4096,
    "batch_size": 4,  # Adjust based on DGX Spark memory
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "num_epochs": 3,
    "warmup_steps": 100,
    "logging_steps": 10,
    "save_steps": 500,
    "output_dir": "./models/factcheck-llama-canadian",
    
    # LoRA configuration for efficient fine-tuning
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
}

