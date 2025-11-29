import sys
import glob
import os
import yaml

# NOTE: This script requires the PyYAML library. You may need to install it:
# pip install PyYAML

def load_all_results(run_dir):
    """
    Loads and processes results from all 'artifacts/results.yml' files 
    within the specified run directory structure.
    
    Args:
        run_dir (str): The name of the top-level run directory (e.g., '3c0807020a456eba').

    Returns:
        list: A list of dictionaries, each representing a single metric result.
    """
    # Use glob to find all results.yml files in subdirectories, assuming the structure:
    # run_dir/run_dir.X/artifacts/results.yml
    search_path = os.path.join(run_dir, '**', 'artifacts', 'results.yml')
    # Use recursive=True to search subdirectories
    result_files = glob.glob(search_path, recursive=True)

    if not result_files:
        print(f"Error: No results.yml files found in '{run_dir}' using pattern '{search_path}'", file=sys.stderr)
        return []

    all_metrics = []
    processed_tasks = set() # Use a set to prevent duplicate task entries across files

    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                # Use safe_load to safely parse the YAML content
                data = yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not read or parse YAML file {file_path}. Skipping. Error: {e}", file=sys.stderr)
            continue
        
        # We focus on the 'tasks' section for detailed metrics
        tasks = data.get('results', {}).get('tasks', {})
        
        for task_name, task_data in tasks.items():
            # Check if we have already processed this task/file (useful for deduplication)
            if task_name in processed_tasks:
                 # In a typical lm-eval run, each task is run once and saved to its own file.
                 # If we encounter it again, we might prefer to skip it to avoid duplicates, 
                 # but for now, we'll allow it and let the set below handle exact duplicates.
                 pass

            metrics = task_data.get('metrics', {})
            for metric_name, metric_data in metrics.items():
                
                # Extract the score (value) and standard error (stderr)
                try:
                    score_data = metric_data.get('scores', {}).get(metric_name, {})
                    value = score_data.get('value')
                    stderr = score_data.get('stats', {}).get('stderr')
                    
                    if value is not None:
                        # Format for printing
                        value_str = f"{value:.4f}"
                        # Only show stderr if it's a number and not 0.0
                        stderr_str = f"{stderr:.4f}" if stderr and stderr != 0.0 else "N/A"

                        metric_entry = {
                            'task': task_name,
                            'metric': metric_name,
                            'value': value_str,
                            'stderr': stderr_str,
                        }
                        
                        # Use the tuple of all relevant data to ensure we don't add duplicate rows
                        # in case results are saved across multiple artifacts files.
                        unique_key = (metric_entry['task'], metric_entry['metric'], metric_entry['value'])
                        if unique_key not in processed_tasks:
                            all_metrics.append(metric_entry)
                            processed_tasks.add(unique_key)
                            
                except Exception as e:
                    print(f"Warning: Failed to process metric {metric_name} in task {task_name} in {file_path}. Error: {e}. Skipping.", file=sys.stderr)
                    continue

    return all_metrics

def format_summary_table(all_metrics):
    """
    Formats the collected metrics into two tables: 
    1. Main Summary Table (All tasks except MGSM sub-tasks)
    2. Multilingual Math (MGSM) Table
    """
    
    # --- 1. Separate MGSM tasks for the specialized table ---
    mgsm_metrics = [m for m in all_metrics if m['task'].startswith('mgsm_')]
    other_metrics = [m for m in all_metrics if not m['task'].startswith('mgsm_')]
    
    output = []
    
    # --- 2. Format the Main Summary Table ---
    if other_metrics:
        output.append("## 📊 Model Evaluation Summary")
        output.append("") # Explicit blank line
        output.append("| Task Name | Metric Name | Score (Value) | Standard Error (Stderr) |")
        # Use spaced separator for maximum compatibility
        output.append("| :--- | :--- | :--- | :--- |")
        
        # Sort for better readability (case-insensitive task name, then metric name)
        sorted_other_metrics = sorted(other_metrics, key=lambda x: (x['task'].lower(), x['metric']))

        for m in sorted_other_metrics:
            output.append(f"| {m['task']} | {m['metric']} | {m['value']} | {m['stderr']} |")
        output.append("") # Empty line for separation

    # --- 3. Format the Multilingual Math (MGSM) Table ---
    if mgsm_metrics:
        output.append("## Multilingual Math (MGSM) Results")
        output.append("") # Explicit blank line
        output.append("This table breaks down results for the MGSM task by language and prompt type.")
        output.append("") # Explicit blank line
        
        mgsm_data = {}
        # Group by language/prompt type (task name)
        for m in mgsm_metrics:
            task = m['task']
            metric = m['metric']
            
            # Identify language and prompt type from task name (e.g., mgsm_en_cot_bn -> bn, mgsm_native_cot_es -> es)
            parts = task.split('_')
            language_code = parts[-1].upper() # BN, ES, etc.
            prompt_type = "Native" if "native" in task else "English"
            
            # The key for grouping should uniquely identify a single row in the final MGSM table
            key = (language_code, prompt_type)
            if key not in mgsm_data:
                mgsm_data[key] = {'language_code': language_code, 'prompt_type': prompt_type}
            
            # Store the flexible and strict scores
            if 'flexible-extract' in metric:
                mgsm_data[key]['flexible'] = m['value']
            elif 'strict-match' in metric:
                mgsm_data[key]['strict'] = m['value']
                
        # Prepare the MGSM table rows
        mgsm_table_rows = []
        for (lang, p_type), data in mgsm_data.items():
            flexible_score = data.get('flexible', 'N/A')
            strict_score = data.get('strict', 'N/A')
            mgsm_table_rows.append((lang, p_type, flexible_score, strict_score))
            
        output.append("| Language | Prompt Type | Flexible Extract (Score) | Strict Match (Score) |")
        # Use spaced separator for maximum compatibility
        output.append("| :--- | :--- | :--- | :--- |")
        
        # Sort the MGSM table by language code and then prompt type
        sorted_mgsm_rows = sorted(mgsm_table_rows, key=lambda x: (x[0], x[1]))
        
        for lang, p_type, flexible, strict in sorted_mgsm_rows:
            output.append(f"| {lang} | {p_type} | {flexible} | {strict} |")

    return "\n".join(output)

def main():
    if len(sys.argv) != 2:
        print("Usage: python print_summary_table.py <run_directory>", file=sys.stderr)
        sys.exit(1)

    # Handle the case where the user might provide the directory with a trailing slash
    run_dir = sys.argv[1].rstrip('/')
    
    # We rely on the user having PyYAML installed in their environment for this script to run.
    try:
        if not os.path.isdir(run_dir):
            print(f"Error: Directory '{run_dir}' not found.", file=sys.stderr)
            sys.exit(1)
            
        print(f"--- Processing results in directory: {run_dir} ---\n")
        
        all_metrics = load_all_results(run_dir)
        
        if not all_metrics:
            print("No metrics were successfully loaded.", file=sys.stderr)
            return

        markdown_table = format_summary_table(all_metrics)
        
        print(markdown_table)
        
        print(f"\n--- End of Summary for {run_dir} ---")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    # Add a check for PyYAML installation if not running in an environment where it's guaranteed.
    try:
        import yaml
    except ImportError:
        print("Error: PyYAML library not found. Please install it using 'pip install PyYAML'.", file=sys.stderr)
        sys.exit(1)
        
    main()