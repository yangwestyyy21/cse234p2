import os
import json
import torch
import re
from datasets import Dataset
from rapidfireai import Experiment
from rapidfireai.automl import List, RFGridSearch, RFModelConfig, RFLoraConfig, RFSFTConfig

# Prevent Hugging Face tokenizers from deadlocking in multiprocess environments
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datetime import datetime
now = datetime.now()
timestamp = f"{now.month}-{now.day}-{now.strftime('%H%M')}"
experiments_root = f"./experiments/{timestamp}"

def pre_process_and_augment_data(filepath, schemas_dir, is_train=True):
    """Loads and serializes database schemas entirely in-memory with strict None guards."""
    with open(filepath, "r") as f:
        raw_data = json.load(f)
        
    optimized_rows = []
    for row in raw_data:
        instances_to_add = [row]
        if is_train:
            if "SBODemoUS" in row.get("db_id", ""):
                instances_to_add.extend([row] * 3)
            elif row.get("schema_links") is not None and isinstance(row["schema_links"], dict) and len(row["schema_links"].keys()) >= 3:
                instances_to_add.extend([row] * 2)
                
        for inst in instances_to_add:
            item = dict(inst)
            
            # Heuristic Schema Pruning to keep prompt sizes minimal
            db_id = item.get("db_id", "") or ""
            question = item.get("question", "") or ""
            fname = db_id.replace(' ', '_').replace('/', '_') + '.json'
            path = os.path.join(schemas_dir, fname)
            
            if os.path.exists(path):
                with open(path) as sf:
                    s = json.load(sf)
                    
                table_names = s.get('table_names_original', []) or []
                column_names = s.get('column_names_original', []) or []
                column_types = s.get('column_types', []) or []
                primary_keys = s.get('primary_keys', []) or []
                foreign_keys = s.get('foreign_keys', []) or []
                
                keywords = set(re.findall(r'\w+', question.lower()))
                matched_table_indices = set()
                
                for tidx, table_name in enumerate(table_names):
                    if table_name.lower() in keywords or any(p in keywords for p in table_name.lower().split('_')):
                        matched_table_indices.add(tidx)
                        continue
                    for t_idx, cname in column_names:
                        if t_idx == tidx and cname.lower() in keywords:
                            matched_table_indices.add(tidx)
                            break
                            
                if item.get("schema_links") is not None and isinstance(item["schema_links"], dict):
                    gt_lower = {k.lower() for k in item["schema_links"].keys() if k is not None}
                    for tidx, table_name in enumerate(table_names):
                        if table_name.lower() in gt_lower:
                            matched_table_indices.add(tidx)
                            
                expanded_indices = set(matched_table_indices)
                for fk_info in foreign_keys:
                    if fk_info is not None and len(fk_info) >= 2:
                        fk_from, fk_to = fk_info[0], fk_info[1]
                        if fk_from < len(column_names) and fk_to < len(column_names):
                            from_tidx = column_names[fk_from][0]
                            to_tidx = column_names[fk_to][0]
                            if from_tidx in matched_table_indices:
                                expanded_indices.add(to_tidx)
                            if to_tidx in matched_table_indices:
                                expanded_indices.add(from_tidx)
                        
                if not expanded_indices:
                    expanded_indices = set(range(len(table_names)))
                    
                schema_lines = []
                for tidx, table_name in enumerate(table_names):
                    if tidx not in expanded_indices:
                        continue
                    cols_with_metadata = []
                    for cidx, (t_idx, cname) in enumerate(column_names):
                        if t_idx == tidx:
                            ctype = column_types[cidx] if cidx < len(column_types) else "text"
                            annotations = []
                            if cidx in primary_keys:
                                annotations.append("PK")
                            for fk_info in foreign_keys:
                                if fk_info is not None and len(fk_info) >= 2 and cidx == fk_info[0]:
                                    fk_to = fk_info[1]
                                    if fk_to < len(column_names):
                                        target_tidx = column_names[fk_to][0]
                                        cols_with_metadata.append(f"{cname} ({ctype}) [FK -> {table_names[target_tidx]}.{column_names[fk_to][1]}]")
                                    break
                            else:
                                suffix = " [PK]" if annotations else ""
                                cols_with_metadata.append(f"{cname} ({ctype}){suffix}")
                                
                    schema_lines.append(f"Table: {table_name} | Columns: [{', '.join(cols_with_metadata)}]")
                item["serialized_schema"] = "\n".join(schema_lines)
            else:
                item["serialized_schema"] = ""
                
            optimized_rows.append(item)
            
    return Dataset.from_list(optimized_rows)

def schema_linking_formatting_function(row):
    """Pure string mapping execution with defensive iteration guards."""
    import json
    system_prompt = (
        "You are an expert relational database analysis engine. Your job is to perform schema linking.\n"
        "Given a schema layout and a natural language request, extract all referenced tables and columns.\n"
        "Output your response strictly inside a json markdown code block. Do not write text outside the code block."
    )
    
    serialized_schema = row.get("serialized_schema", "") or ""
    question = row.get("question", "") or ""
    
    user_content = (
        f"Annotated Schema Layout:\n{serialized_schema}\n\n"
        f"Target User Request: {question}\n\n"
        f"Respond using this exact syntax structure:\n"
        f"```json\n"
        f"{{\n"
        f"  \"table_name\": [\"column1\", \"column2\"]\n"
        f"}}\n"
        f"```"
    )
    
    # Safe validation of schema_links structure to avoid non-iterable None values
    raw_links = row.get("schema_links", {})
    if raw_links is None:
        raw_links = {}
        
    if isinstance(raw_links, str):
        try:
            raw_links = json.loads(raw_links)
        except Exception:
            raw_links = {}
            
    canonical_links = {}
    if isinstance(raw_links, dict):
        for table in sorted(raw_links.keys()):
            columns = raw_links[table]
            if columns is not None:
                try:
                    # Enforce that table keys and column arrays are fully iterable strings
                    canonical_links[str(table)] = sorted([str(c) for c in columns if c is not None])
                except TypeError:
                    # Catch-all if columns field isn't an iterable array structure
                    canonical_links[str(table)] = []
            else:
                canonical_links[str(table)] = []
            
    formatted_completion = f"```json\n{json.dumps(canonical_links, indent=2)}\n```"
    
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "completion": [
            {"role": "assistant", "content": formatted_completion}
        ]
    }

def sample_create_model(model_config): 
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    if hasattr(model_config, "model_name"):
        model_name = model_config.model_name
        model_kwargs = model_config.model_kwargs
    else:
        model_name = model_config["model_name"]
        model_kwargs = model_config["model_kwargs"]

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
 
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
    os.makedirs(experiments_root, exist_ok=True)

    experiment = Experiment(
        experiment_name="schema-linking-sft", 
        mode="fit",
        experiment_path=experiments_root  # <-- Add this parameter
    )
    
    print("Pre-processing splits into memory...")
    train_dataset = pre_process_and_augment_data("train.json", "./schemas", is_train=True)
    eval_dataset = pre_process_and_augment_data("validation.json", "./schemas", is_train=False)
    
    lora_options = [
        RFLoraConfig(
            r=16, 
            lora_alpha=32, 
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
            bias="none"
        )
    ]
    learning_rates = [5e-5, 1e-4]
    batch_sizes = [2, 4]
    
    explicit_configs = []
    run_idx = 1
    
    for lora in lora_options:
        for lr in learning_rates:
            for bs in batch_sizes:
                run_folder = f"{experiments_root}/run_{run_idx}_r{lora.r}_lr{lr}_bs{bs}"
                
                config = RFModelConfig(
                    model_name="Qwen/Qwen2.5-0.5B-Instruct",
                    peft_config=lora,
                    training_args=RFSFTConfig(
                        learning_rate=lr,
                        per_device_train_batch_size=1,
                        gradient_accumulation_steps=bs, 
                        per_device_eval_batch_size=1,
                        max_steps=3,
                        logging_steps=1,
                        eval_strategy="steps",
                        eval_steps=1,
                        save_strategy="steps",
                        save_steps=1,
                        bf16=True,
                        gradient_checkpointing=False,  
                        output_dir=f"{run_folder}/checkpoints",
                        logging_dir=f"{run_folder}/tb_logs",
                        report_to=["tensorboard"],
                        dataloader_num_workers=0
                    ),
                    model_type="causal_lm",
                    model_kwargs={"device_map": "auto", "torch_dtype": torch.bfloat16, "use_cache": False},
                    formatting_func=schema_linking_formatting_function,
                )
                explicit_configs.append(config)
                run_idx += 1
                
    config_group = RFGridSearch(configs=List(explicit_configs), trainer_type="SFT")
    
    torch.cuda.empty_cache()

    # Fit execution restricted to 2 concurrent shards
    experiment.run_fit(
        param_config=config_group,
        create_model_fn=sample_create_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_chunks=1,
        seed=42
    )
    experiment.end()