import os
import subprocess
import datetime

def run_pipeline():
    # 1. Ensure the outputs directory exists
    os.makedirs("./outputs", exist_ok=True)

    # 2. Build the timestamp string matching your format (Month-Day-HourMinute)
    # Example: 5-29-1902
    timestamp = datetime.datetime.now().strftime("%m-%d-%H%M")
    output_filename = f"./outputs/{timestamp}.json"

    # Define path variables
    input_file = "validation_input.json"
    gold_file = "validation_gold_schema_links.json"
    schemas_dir = "schemas/"

    # 3. Execute main.py inference pass
    print(f" Running inference: main.py -> {output_filename}")
    main_cmd = [
        "python", "main.py",
        "--input", input_file,
        "--output", output_filename
    ]
    
    result_main = subprocess.run(main_cmd, capture_output=False, text=True)
    if result_main.returncode != 0:
        print("❌ Inference failed with an error. Skipping evaluation step.")
        return

    # 4. Execute eval.py and capture the stdout/stderr stream
    print(f" Running evaluation metrics on predictions...")
    eval_cmd = [
        "python", "eval.py",
        "--predictions", output_filename,
        "--gold", gold_file,
        "--schemas_dir", schemas_dir,
        "--questions_input", input_file
    ]
    
    result_eval = subprocess.run(eval_cmd, capture_output=True, text=True)
    # Combine standard output and any warning/error strings
    eval_terminal_output = result_eval.stdout + result_eval.stderr
    
    # Also print it live to your current terminal session for visibility
    print("\n--- Captured Eval Output ---")
    print(eval_terminal_output)
    print("----------------------------")

    # 5. Read the generated JSON predictions
    if os.path.exists(output_filename):
        with open(output_filename, "r") as f:
            predictions_json = f.read()
    else:
        predictions_json = "[]"

    # 6. Prepend the evaluation summary block to the top of the file
    combined_content = (
        f"========================================================================\n"
        f" TERMINAL EVALUATION SUMMARY: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"========================================================================\n"
        f"{eval_terminal_output.strip()}\n"
        f"========================================================================\n\n"
        f"{predictions_json}"
    )

    # 7. Write the combined hybrid log text back to the output file
    with open(output_filename, "w") as f:
        f.write(combined_content)

    print(f" Finished! Combined metrics + predictions saved to: {output_filename}")

if __name__ == "__main__":
    run_pipeline()