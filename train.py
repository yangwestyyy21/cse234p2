# train.py
import os
import json
import torch
from datasets import Dataset
from rapidfireai import Experiment
from rapidfireai.automl import List, RFGridSearch, RFModelConfig, RFLoraConfig, RFSFTConfig

def load_schema(schemas_dir, db_id):
    """Loads a Spider/SNAILS schema JSON into a text summary for the context window."""
    fname = db_id.replace(' ', '_').replace('/', '_') + '.json'
    path = os.path.join(schemas_dir, fname)
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        s = json.load(f)
    table_names = s['table_names_original']
    column_names = s['column_names_original']
    
    schema_lines = []
    for tidx, table_name in enumerate(table_names):
        cols = [cname for t_idx, cname in column_names if t_idx == tidx]
        schema_lines.append(f"Table: {table_name} | Columns: [{', '.join(cols)}]")
    return "\n".join(schema_lines)

def schema_linking_formatting_function(row):
    """Formats raw instances into a Llama-3 instruction-matching prompt template."""
    # Assuming schemas are stored in a standard local directory
    serialized_db = load_schema("./schemas", row["db_id"])
    
    system_prompt = (
        "You are a strict database parsing utility. Given a database schema and a user question, "
        "identify referenced tables and columns. Return ONLY a valid JSON object mapping "
        "table names to lists of column names. Do not include conversational filler or markdown wrappers."
    )
    user_content = f"Database Schema:\n{serialized_db}\n\nUser Question: {row['question']}"
    
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "completion": [
            {"role": "assistant", "content": json.dumps(row["schema_links"])}
        ]
    }

def create_model_and_tokenizer(model_config): 
    """Factory builder for base models and tokenizers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = model_config["model_name"]
    model_kwargs = model_config["model_kwargs"]
 
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return (model, tokenizer)

if __name__ == "__main__":
    # Ensure root log storage directory exists
    os.makedirs("./logs", exist_ok=True)

    # Initialize experiment framework pointing metadata summaries to ./logs
    experiment = Experiment(
        experiment_name="schema-linking-sft", 
        mode="fit", 
        experiments_path="./logs"
    )
    
    # Load dataset splits
    with open("train.json", "r") as f:
        train_dataset = Dataset.from_list(json.load(f))
    with open("validation.json", "r") as f:
        eval_dataset = Dataset.from_list(json.load(f))
    
    # Define variations to generate 8 unique configurations
    lora_options = [
        RFLoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], bias="none"),
        RFLoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], bias="none")
    ]
    learning_rates = [5e-5, 1e-4]
    batch_sizes = [2, 4]
    
    explicit_configs = []
    run_idx = 0
    
    # Programmatically build combinations to enforce separate log folders
    for lora in lora_options:
        for lr in learning_rates:
            for bs in batch_sizes:
                # Isolate target run directories explicitly inside ./logs
                run_folder = f"./logs/run_{run_idx}_r{lora.r}_lr{lr}_bs{bs}"
                
                config = RFModelConfig(
                    model_name="Qwen/Qwen2.5-1.5B-Instruct",
                    peft_config=lora,
                    training_args=RFSFTConfig(
                        learning_rate=lr,
                        per_device_train_batch_size=bs,
                        per_device_eval_batch_size=2,
                        max_steps=150,
                        logging_steps=10,
                        eval_strategy="steps",
                        eval_steps=50,
                        bf16=True,
                        # --- ENSURE SEPARATE LOG TRACKING HERE ---
                        output_dir=f"{run_folder}/checkpoints",
                        logging_dir=f"{run_folder}/tb_logs",
                        report_to=["tensorboard"]
                    ),
                    model_type="causal_lm",
                    model_kwargs={"device_map": "auto", "torch_dtype": torch.bfloat16, "use_cache": False},
                    formatting_func=schema_linking_formatting_function,
                    generation_config={"max_new_tokens": 128, "temperature": 0.1, "top_p": 0.9}
                )
                explicit_configs.append(config)
                run_idx += 1
                
    # Pass the isolated list group into the auto-ml engine
    config_group = RFGridSearch(configs=List(explicit_configs), trainer_type="SFT")
    
    print(f"Starting execution sweep across {len(explicit_configs)} isolated configurations...")
    experiment.run_fit(
        config_group=config_group,
        model_create_fn=create_model_and_tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_chunks=4,
        seed=42
    )
    experiment.end()
    print("Sweep complete. Run histories are fully tracked under unique subfolders in ./logs/")