# main.py
import os
import re
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description="SLM Schema Linking Inference Entrypoint")
    parser.add_argument("--input", required=True, help="Path to input JSON file containing evaluation questions")
    parser.add_argument("--output", required=True, help="Path to write final schema link target predictions")
    return parser.parse_args()

def load_schema_map(db_id: str):
    """Loads structural ground-truth definitions to protect validation metrics."""
    schema_filename = db_id.replace(" ", "_") + ".json"
    schema_path = os.path.join("./schemas", schema_filename)
    if not os.path.exists(schema_path):
        return {}, {}
        
    with open(schema_path, "r") as f:
        data = json.load(f)
        
    tables = data.get("table_names_original", [])
    columns = data.get("column_names_original", [])
    
    # Map lowercase structures to their native casing counterparts to enable clean downstream repairs
    table_case_map = {t.lower(): t for t in tables}
    table_column_map = {t: [] for t in tables}
    
    for col_idx, (t_idx, col_name) in enumerate(columns):
        if t_idx == -1: # Skip global wildcards
            continue
        parent_table = tables[t_idx]
        table_column_map[parent_table].append(col_name)
        
    return table_case_map, table_column_map

def clean_and_parse_json(raw_string: str, table_case_map, table_column_map):
    """Executes robust defensive processing loops against structured model outputs."""
    # 1. Clear out markdown code block text formatting if appended by the model
    raw_string = raw_string.strip()
    if raw_string.startswith("```json"):
        raw_string = raw_string[7:]
    if raw_string.endswith("```"):
        raw_string = raw_string[:-3]
    raw_string = raw_string.strip()
    
    # 2. Extract JSON payload via regex if conversational filler was generated
    if not (raw_string.startswith("{") and raw_string.endswith("}")):
        match = re.search(r"\{.*\}", raw_string, re.DOTALL)
        if match:
            raw_string = match.group(0)
            
    try:
        parsed_dict = json.loads(raw_string)
    except Exception:
        # Emergency recovery fallback: Emit an empty schema structure to avoid dropping structural evaluation steps
        return {}
        
    sanitized_output = {}
    
    # 3. Validation Filtering Loop Against Genuine Database Ground Truth
    for predicted_table, predicted_cols in parsed_dict.items():
        pred_table_lower = predicted_table.lower()
        
        # Eliminate hallucinated table mappings to protect metric precision
        if pred_table_lower in table_case_map:
            true_table_name = table_case_map[pred_table_lower]
            
            # Enforce structural consistency rules regarding column listings
            if not isinstance(predicted_cols, list):
                predicted_cols = []
                
            valid_columns = []
            true_columns_lower = [c.lower() for c in table_column_map[true_table_name]]
            column_case_map = {c.lower(): c for c in table_column_map[true_table_name]}
            
            for col in predicted_cols:
                if col.lower() in true_columns_lower:
                    # Enforce identifier casing matching the schema files
                    valid_columns.append(column_case_map[col.lower()])
                    
            # Wildcard fallback condition: retain targeted elements without arbitrary columns as empty list []
            sanitized_output[true_table_name] = valid_columns
            
    return sanitized_output

def main():
    args = parse_args()
    
    # Core Model Pipeline Assembly Block
    base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("Initializing Base Language Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    
    # Load the fine-tuned LoRA weights safely from the repository root folder
    adapter_path = "./adapter"
    if os.path.exists(adapter_path):
        print(f"Injecting Fine-Tuned PEFT Adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        print("Warning: No adapter folder located at root. Defaulting to zero-shot base evaluation.")
        model = base_model
        
    model.eval()
    
    # Load Input Evaluation Evaluation File Contexts
    with open(args.input, "r") as f:
        input_data = json.load(f)
        
    output_records = []
    
    # Schema Generation Inference Loop
    with torch.no_grad():
        for record in input_data:
            q_id = record["question_id"]
            db_id = record["db_id"]
            question = record["question"]
            
            # Dynamically parse current structural blueprint metrics
            table_case_map, table_column_map = load_schema_map(db_id)
            
            # Reconstruct identical prompt context bounds used throughout SFT training execution
            serialized_db = ""
            schema_filename = db_id.replace(" ", "_") + ".json"
            schema_path = os.path.join("./schemas", schema_filename)
            if os.path.exists(schema_path):
                with open(schema_path, "r") as sf:
                    s_data = json.load(sf)
                tables = s_data.get("table_names_original", [])
                cols_raw = s_data.get("column_names_original", [])
                s_lines = []
                for t in tables:
                    c_list = [c[1] for c in cols_raw if c[0] == tables.index(t)]
                    s_lines.append(f"Table: {t} | Columns: [{', '.join(c_list)}]")
                serialized_db = "\n".join(s_lines)
                
            messages = [
                {"role": "system", "content": "You are a strict database parsing utility. Given a database schema and a user question, identify referenced tables and columns. Return ONLY a valid JSON object mapping table names to lists of column names. Do not include conversational filler or markdown wrappers."},
                {"role": "user", "content": f"Database Schema:\n{serialized_db}\n\nUser Question: {question}"}
            ]
            
            # Apply chat template encoding structures explicitly
            input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
            
            outputs = model.generate(
                input_ids,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # Strip prompt sequence context lengths from generation bounds
            generated_tokens = outputs[0][len(input_ids[0]):]
            raw_prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Execute validation safety nets on generated text output
            sanitized_links = clean_and_parse_json(raw_prediction, table_case_map, table_column_map)
            
            # Construct submission record element mapping strictly back to specifications
            output_records.append({
                "question_id": q_id,
                "schema_links": sanitized_links
            })
            
    # Serialize output array back to the filesystem target path
    with open(args.output, "w") as f:
        json.dump(output_records, f, indent=2)
    print(f"Successfully processed inference dataset. Results saved to {args.output}")

if __name__ == "__main__":
    main()