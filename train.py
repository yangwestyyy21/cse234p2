# train.py
import os
import json
import torch
from datasets import Dataset
from rapidfireai import Experiment
from rapidfireai.automl import List, RFGridSearch, RFModelConfig, RFLoraConfig, RFSFTConfig

# Prevent Hugging Face tokenizers from deadlocking in multiprocess environments
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Optional: helps avoid CUDA context issues across process forks
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def schema_linking_formatting_function(row):
    """Formats raw instances into a Llama-3 instruction-matching prompt template."""
    # Nesting imports and helper functions ensures they serialize perfectly across workers
    import os
    import json

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

    # Now load_schema is safely accessible inside the worker process boundary
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

def sample_create_model(model_config): 
    """Factory function to dynamically create model and tokenizer objects for any given config.
    
    Must return a two-element tuple: (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM
    from transformers import GenerationConfig

    model_name = model_config["model_name"]
    model_type = model_config["model_type"]
    model_kwargs = model_config["model_kwargs"]

    if model_type == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    elif model_type == "seq2seq_lm":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
    elif model_type == "masked_lm":
        model = AutoModelForMaskedLM.from_pretrained(model_name, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
 
    # Configure generation parameters directly on the model object 
    # to avoid passing them into the SFTTrainer constructor
    try:
        model.generation_config = GenerationConfig.from_pretrained(model_name)
    except Exception:
        model.generation_config = GenerationConfig()
        
    model.generation_config.max_new_tokens = 128
    model.generation_config.temperature = 0.1
    model.generation_config.top_p = 0.9
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    return (model, tokenizer)

if __name__ == "__main__":
    # Ensure root log storage directory exists
    os.makedirs("./logs", exist_ok=True)

    # Initialize experiment framework
    experiment = Experiment(
        experiment_name="schema-linking-sft", 
        mode="fit"
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
                        eval_steps=10,
                        bf16=True,
                        gradient_checkpointing=True,
                        output_dir=f"{run_folder}/checkpoints",
                        logging_dir=f"{run_folder}/tb_logs",
                        report_to=["tensorboard"]
                    ),
                    model_type="causal_lm",
                    model_kwargs={"device_map": "auto", "torch_dtype": torch.bfloat16, "use_cache": False},
                    formatting_func=schema_linking_formatting_function,
                )
                explicit_configs.append(config)
                run_idx += 1
                
    # Pass the isolated list group into the auto-ml engine
    config_group = RFGridSearch(configs=List(explicit_configs), trainer_type="SFT")
    
    print(f"Starting execution sweep across {len(explicit_configs)} isolated configurations...")
    experiment.run_fit(
        param_config=config_group,
        create_model_fn=sample_create_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_chunks=4,
        seed=42
    )
    experiment.end()