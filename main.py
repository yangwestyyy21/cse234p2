# main.py
import argparse
import json
import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Global cache pointers to prevent loading the model repeatedly in the loop
MODEL_LOADED = False
model_pipeline = None
tokenizer_instance = None

def load_schema_case_maps(schemas_dir, db_id):
    """Loads schema files to build mapping rules for case correction and verification."""
    fname = db_id.replace(' ', '_').replace('/', '_') + '.json'
    path = os.path.join(schemas_dir, fname)
    if not os.path.exists(path):
        return {}, {}, ""
        
    with open(path) as f:
        s = json.load(f)
        
    table_names = s['table_names_original']
    column_names = s['column_names_original']
    
    table_case_map = {t.lower(): t for t in table_names}
    column_case_map = {t: {} for t in table_names}
    
    schema_lines = []
    for tidx, table_name in enumerate(table_names):
        cols = []
        for t_idx, cname in column_names:
            if t_idx == tidx:
                cols.append(cname)
                column_case_map[table_name][cname.lower()] = cname
        schema_lines.append(f"Table: {table_name} | Columns: [{', '.join(cols)}]")
        
    return table_case_map, column_case_map, "\n".join(schema_lines)

def post_process_json(raw_text, table_case_map, column_case_map):
    """Cleans up markdown artifacts and applies case matching against ground truth schemas."""
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    if not (text.startswith("{") and text.endswith("}")):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        text = match.group(0) if match else "{}"
        
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
        
    sanitized = {}
    for raw_table, raw_cols in parsed.items():
        table_lc = raw_table.lower()
        if table_lc in table_case_map:
            true_table = table_case_map[table_lc]
            sanitized[true_table] = []
            
            if isinstance(raw_cols, list):
                for col in raw_cols:
                    col_lc = col.lower()
                    if col_lc in column_case_map[true_table]:
                        sanitized[true_table].append(column_case_map[true_table][col_lc])
    return sanitized

def predict_schema_links(question, db_id, schemas_dir):
    """Loads the model once and processes streaming generations per instance entry."""
    global MODEL_LOADED, model_pipeline, tokenizer_instance
    
    base_model_path = "meta-llama/Llama-3.2-1B-Instruct"
    adapter_path = "./adapter"  # Points to your best-performing checkpoint directory
    
    if not MODEL_LOADED:
        print("Initializing production fine-tuned SLM engine...")
        tokenizer_instance = AutoTokenizer.from_pretrained(base_model_path)
        if tokenizer_instance.pad_token is None:
            tokenizer_instance.pad_token = tokenizer_instance.eos_token
            
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        
        if os.path.exists(adapter_path):
            print(f"Injecting fine-tuned weights from {adapter_path}...")
            model_pipeline = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            print("Warning: Missing adapter checkpoint. Operating in zero-shot fallback mode.")
            model_pipeline = base_model
            
        model_pipeline.eval()
        MODEL_LOADED = True

    # Build context structures
    table_case_map, column_case_map, serialized_db = load_schema_case_maps(schemas_dir, db_id)
    
    system_prompt = (
        "You are a strict database parsing utility. Given a database schema and a user question, "
        "identify referenced tables and columns. Return ONLY a valid JSON object mapping "
        "table names to lists of column names. Do not include conversational filler or markdown wrappers."
    )
    user_content = f"Database Schema:\n{serialized_db}\n\nUser Question: {question}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    input_ids = tokenizer_instance.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(model_pipeline.device)
    
    with torch.no_grad():
        outputs = model_pipeline.generate(
            input_ids,
            max_new_tokens=128,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer_instance.pad_token_id,
            eos_token_id=tokenizer_instance.eos_token_id
        )
        
    generated_tokens = outputs[0][len(input_ids[0]):]
    raw_prediction = tokenizer_instance.decode(generated_tokens, skip_special_tokens=True)
    
    return post_process_json(raw_prediction, table_case_map, column_case_map)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',   required=True)
    ap.add_argument('--output',  required=True)
    ap.add_argument('--schemas_dir', default='./schemas')
    args = ap.parse_args()

    with open(args.input) as f:
        items = json.load(f)
        
    preds = []
    for it in items:
        links = predict_schema_links(it['question'], it['db_id'], args.schemas_dir)
        preds.append({'question_id': it['question_id'], 'schema_links': links})
        
    with open(args.output, 'w') as f:
        json.dump(preds, f, indent=2)
    print(f"Wrote {len(preds)} predictions to {args.output}")

if __name__ == '__main__':
    main()