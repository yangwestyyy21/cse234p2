# train.py
import os
import json
from datasets import Dataset
from rapidfireai import Experiment
from rapidfireai.automl import List, RFGridSearch, RFModelConfig, RFLoraConfig, RFSFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Schema Serialization Utility
def serialize_schema(db_id: str) -> str:
    """Reads a Spider schema file and converts it into a lean string representation."""
    # Resolve the space-to-underscore mapping mentioned in the spec
    schema_filename = db_id.replace(" ", "_") + ".json"
    schema_path = os.path.join("./schemas", schema_filename)
    
    if not os.path.exists(schema_path):
        return ""
        
    with open(schema_path, "r") as f:
        schema_data = json.load(f)
        
    serialized = []
    # Build a concise representation to save context window tokens
    for table in schema_data.get("table_names_original", []):
        cols = [c[1] for c in schema_data.get("column_names_original", []) if c[0] == schema_data["table_names_original"].index(table)]
        serialized.append(f"Table: {table} | Columns: [{', '.join(cols)}]")
        
    return "\n".join(serialized)

# 2. Data Formatting Function for Llama Architecture
def llama_formatting_function(row):
    """Formats each data instance to match the Llama-3 instruction token space."""
    serialized_db = serialize_schema(row["db_id"])
    
    system_prompt = (
        "You are a strict database parsing utility. Given a database schema and a user question, "
        "identify referenced tables and columns. Return ONLY a valid JSON object mapping "
        "table names to lists of column names. Do not include conversational filler or markdown wrappers."
    )
    user_content = f"Database Schema:\n{serialized_db}\n\nUser Question: {row['question']}"
    
    # Target training output contract
    completion_content = json.dumps(row["schema_links"])
    
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "completion": [
            {"role": "assistant", "content": completion_content}
        ]
    }

# 3. Model Creation Function for RapidFire AI Interface
def create_llama_model(model_config):
    """Loads and configures base model and tokenizer pairs for the engine."""
    model_name = model_config["model_name"]
    model_kwargs = model_config["model_kwargs"]
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Llama requires explicit pad token assignment if it's missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer

if __name__ == "__main__":
    # Initialize the tracking experiment instance
    experiment = Experiment(experiment_name="llama-schema-linking", mode="fit")
    
    # Load your training files
    with open("train.json", "r") as f:
        train_data = json.load(f)
    with open("validation.json", "r") as f:
        val_data = json.load(f)
        
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(val_data)
    
    # 4. Programmatic Generation of 8 Distinct Configurations
    # We sweep across 2 LoRA ranks, 2 learning rates, and 2 batch configurations
    lora_variants = [
        RFLoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"], bias="none"),
        RFLoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"], bias="none")
    ]
    lr_variants = [1e-4, 2e-4]
    batch_variants = [2, 4]
    
    configs_list = []
    run_counter = 0
    
    for lora in lora_variants:
        for lr in lr_variants:
            for bs in batch_variants:
                # Isolate outputs & logs to unique, run-specific folders under ./logs
                run_dir = f"./logs/run_{run_counter}_r{lora.r}_lr{lr}_bs{bs}"
                
                model_config = RFModelConfig(
                    model_name="meta-llama/Llama-3.2-1B-Instruct",
                    peft_config=lora,
                    training_args=RFSFTConfig(
                        learning_rate=lr,
                        per_device_train_batch_size=bs,
                        per_device_eval_batch_size=bs,
                        max_steps=200,
                        logging_steps=10,
                        eval_strategy="steps",
                        eval_steps=50,
                        bf16=True,
                        # --- ENSURE LOG ROUTING ---
                        output_dir=f"{run_dir}/checkpoints",
                        logging_dir=f"{run_dir}/tb_logs",
                        report_to=["tensorboard"]
                    ),
                    model_type="causal_lm",
                    model_kwargs={"device_map": "auto", "torch_dtype": "auto", "use_cache": False},
                    formatting_func=llama_formatting_function,
                    generation_config={"max_new_tokens": 256, "temperature": 0.1, "top_p": 0.9}
                )
                configs_list.append(model_config)
                run_counter += 1
                
    # Launch parallel/chunked grid search execution
    config_group = RFGridSearch(configs=List(configs_list), trainer_type="SFT")
    experiment.run_fit(config_group, create_llama_model, train_dataset, eval_dataset, num_chunks=4, seed=42)
    experiment.end()