import sys
import glob
import os
import argparse
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
                        
                        unique_key = (metric_entry['task'], metric_entry['metric'], metric_entry['value'])
                        if unique_key not in processed_tasks:
                            all_metrics.append(metric_entry)
                            processed_tasks.add(unique_key)
                            
                except Exception as e:
                    print(f"Warning: Failed to process metric {metric_name} in task {task_name} in {file_path}. Error: {e}. Skipping.", file=sys.stderr)
                    continue

    return all_metrics

def format_comparison_tables(results_map):
    """
    Formats the collected metrics from multiple runs into comparison tables.
    
    Args:
        results_map (dict): A dictionary where keys are run IDs (strings) 
                            and values are lists of metric dictionaries.
    """
    run_ids = sorted(results_map.keys())
    
    # Consolidate all data into a pivot structure: 
    # { task: { metric: { run_id: value } } }
    
    mgsm_tasks = {} # Special handling for MGSM
    other_tasks = {}
    
    all_known_metrics = set()
    
    for run_id, metrics in results_map.items():
        for m in metrics:
            task = m['task']
            metric = m['metric']
            val = m['value']
            
            target_dict = mgsm_tasks if task.startswith('mgsm_') else other_tasks
            
            if task not in target_dict:
                target_dict[task] = {}
            if metric not in target_dict[task]:
                target_dict[task][metric] = {}
            
            target_dict[task][metric][run_id] = val

    output = []
    
    # --- 1. General Summary Table ---
    if other_tasks:
        output.append("## 📊 Model Evaluation Comparison")
        output.append("")
        
        # Build Header
        header = "| Task Name | Metric Name | " + " | ".join(run_ids) + " |"
        output.append(header)
        
        # Build Separator
        separator = "| :--- | :--- | " + " | ".join([":---"] * len(run_ids)) + " |"
        output.append(separator)
        
        # Build Rows
        sorted_tasks = sorted(other_tasks.keys(), key=str.lower)
        for task in sorted_tasks:
            sorted_metrics = sorted(other_tasks[task].keys())
            for metric in sorted_metrics:
                values = [
                    other_tasks[task][metric].get(run_id, '-')
                    for run_id in run_ids
                ]
                row = f"| {task} | {metric} | " + ' | '.join(values) + ' |'
                output.append(row)
        
        output.append("")

    # --- 2. Multilingual Math (MGSM) Table ---
    if mgsm_tasks:
        output.append("## Multilingual Math (MGSM) Comparison")
        output.append("")
        output.append("Breakdown by language, prompt type, and metric.")
        output.append("")
        
        # Group MGSM data
        # Key: (Language, Prompt Type, Metric Type) -> { run_id: value }
        mgsm_rows = {}
        
        for task, metrics_dict in mgsm_tasks.items():
            # Identify language and prompt type
            parts = task.split('_')
            language_code = parts[-1].upper()
            prompt_type = "Native" if "native" in task else "English"
            
            for metric_name, runs_data in metrics_dict.items():
                metric_type = "Flexible" if "flexible-extract" in metric_name else "Strict" if "strict-match" in metric_name else metric_name
                
                key = (language_code, prompt_type, metric_type)
                if key not in mgsm_rows:
                    mgsm_rows[key] = {}
                
                # Merge run data
                for run_id, val in runs_data.items():
                    mgsm_rows[key][run_id] = val

        # Build MGSM Header
        header = "| Language | Prompt Type | Metric | " + " | ".join(run_ids) + " |"
        output.append(header)
        
        # Build MGSM Separator
        separator = "| :--- | :--- | :--- | " + " | ".join([":---"] * len(run_ids)) + " |"
        output.append(separator)
        
        # Build MGSM Rows
        # Sort by Language, then Prompt Type, then Metric Type
        sorted_keys = sorted(mgsm_rows.keys(), key=lambda x: (x[0], x[1], x[2]))
        
        for (lang, p_type, m_type) in sorted_keys:
            values = [
                mgsm_rows[(lang, p_type, m_type)].get(run_id, '-')
                for run_id in run_ids
            ]
            row = f"| {lang} | {p_type} | {m_type} | " + ' | '.join(values) + ' |'
            output.append(row)

    return "\n".join(output)

def main():
    parser = argparse.ArgumentParser(
        description='Compare evaluation results from multiple runs.'
    )
    parser.add_argument(
        'run_dirs',
        nargs='+',
        help='One or more run directories to compare'
    )
    parser.add_argument(
        '--out', '-o',
        type=str,
        default=None,
        help='Output file path for markdown report (optional)'
    )
    args = parser.parse_args()

    run_dirs = args.run_dirs
    results_map = {}
    
    print(f"--- Processing {len(run_dirs)} run(s) ---", file=sys.stderr)
    
    for run_dir in run_dirs:
        # Use directory name as ID, fall back to full path if needed
        run_id = os.path.basename(os.path.normpath(run_dir))
        if not run_id:
            run_id = run_dir
            
        if not os.path.isdir(run_dir):
            print(
                f"Warning: Directory '{run_dir}' not found. Skipping.",
                file=sys.stderr
            )
            continue
            
        print(f"Loading results from: {run_id} ({run_dir})", file=sys.stderr)
        metrics = load_all_results(run_dir)
        if metrics:
            results_map[run_id] = metrics
        else:
            print(f"Warning: No metrics found for {run_id}", file=sys.stderr)

    if not results_map:
        print("No valid results found to display.", file=sys.stderr)
        return

    markdown_table = format_comparison_tables(results_map)
    
    if args.out:
        with open(args.out, 'w') as f:
            f.write(markdown_table)
            f.write('\n')
        print(f"Report written to: {args.out}", file=sys.stderr)
    else:
        print(markdown_table)
    
    print(f"--- End of Summary ---", file=sys.stderr)


if __name__ == '__main__':
    # Add a check for PyYAML installation if not running in an environment where it's guaranteed.
    try:
        import yaml
    except ImportError:
        print("Error: PyYAML library not found. Please install it using 'pip install PyYAML'.", file=sys.stderr)
        sys.exit(1)
        
    main()