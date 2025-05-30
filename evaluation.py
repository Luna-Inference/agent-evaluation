# Requirements:
# - Correctness: Is the answer truthfully correct?
# - Number of steps: How many steps were taken to arrive at the final answer?
# - Failure count: How many times did code execution fail?
# - Misunderstanding count: How many times did the LLM use a tool incorrectly?
# - Token usage: How many prefill and generation tokens were used?
# - Time: What was the total end-to-end time taken?
# - Category: What category does this prompt fall into?
# - Self-confidence: How confident is the model in its own internal knowledge? 
#   If an answer is in the general information domain, the model should rely on its own knowledge.

import os
import json
import re
import glob
from collections import defaultdict
import csv
from openai import AzureOpenAI
import argparse

import time

endpoint = "https://openai-1306.openai.azure.com/"
deployment = "o3-mini"
with open('secrets.json', 'r') as f:
    secrets = json.load(f)
    subscription_key = secrets['azure_openai']

# Initialize the OpenAI client with the API key
def get_openai_client():
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version="2025-01-01-preview",
    )
    return client

# Find all log directories
def find_log_directories():
    return [d for d in glob.glob('log*') if os.path.isdir(d)]

# Get all log files in a directory
def get_log_files(directory):
    return glob.glob(os.path.join(directory, '*.txt'))

# Parse a log file to extract relevant information
def parse_log_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract task information
    task_match = re.search(r'Task \d+/\d+: (.+?)\n', content)
    task = task_match.group(1) if task_match else "Unknown task"
    
    # Extract the final answer
    final_answer_match = re.search(r'Out - Final answer: (.+?)\n', content)
    answer = final_answer_match.group(1) if final_answer_match else "No answer found"
    
    # Count the number of steps
    step_count = len(re.findall(r'━+ Step \d+ ━+', content))
    
    # Count failures
    failure_count = content.count("Error:")
    
    # Extract token usage
    token_matches = re.findall(r'Input tokens: (\d+) \| Output tokens: (\d+)', content)
    input_tokens = sum(int(m[0]) for m in token_matches) if token_matches else 0
    output_tokens = sum(int(m[1]) for m in token_matches) if token_matches else 0
    
    # Extract total time
    time_matches = re.findall(r'Duration (\d+\.\d+) seconds', content)
    total_time = sum(float(t) for t in time_matches) if time_matches else 0
    
    # Count misunderstandings (this requires deeper analysis with Azure OpenAI)
    misunderstanding_count = 0  # Will be evaluated by Azure OpenAI (o3-mini)
    
    return {
        "task": task,
        "answer": answer,
        "step_count": step_count,
        "failure_count": failure_count,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_time": total_time,
        "file_path": file_path,
        "raw_content": content
    }

# Evaluate the log with Azure OpenAI (o3-mini)
def evaluate_with_gpt(log_data, client):
    system_message = "You are an expert evaluator of AI agent performance. Evaluate the provided log objectively and concisely."
    
    user_prompt = f"""
    Evaluate the following agent interaction log based on these criteria:
    
    1. Correctness: Is the answer truthfully correct? (yes/no/partial)
    2. Number of steps: {log_data['step_count']} steps were taken to arrive at the final answer.
    3. Failure count: {log_data['failure_count']} code execution failures were detected.
    4. Misunderstanding count: How many times did the agent use a tool incorrectly? Review the log carefully.
    5. Token usage: {log_data['input_tokens']} input tokens and {log_data['output_tokens']} output tokens were used.
    6. Time: {log_data['total_time']:.2f} seconds total end-to-end time taken.
    7. Category: What category does this prompt fall into? (general knowledge, coding, math, etc.)
    8. Self-confidence: Rate as "high/medium/low" based on the following definition:
       - High: This question is about common knowledge that the model should know without tools (e.g., capitals, basic facts).
         If the model still used tools for this, it shows low self-confidence in its knowledge.
       - Medium: This question might require some external verification but contains elements the model should know.
       - Low: This question legitimately requires external tools or information the model couldn't reasonably know.
    
    Task: {log_data['task']}
    Final answer: {log_data['answer']}
    
    Log excerpt:
    {log_data['raw_content'][:3000]}... (truncated)
    
    Please provide your evaluation in the following JSON format:
    {{"correctness": "yes/no/partial", 
      "misunderstanding_count": number, 
      "category": "category name", 
      "self_confidence": "high/medium/low",
      "explanation": "brief explanation of your evaluation"}}
    """
    
    # Prepare messages in the format expected by Azure OpenAI
    messages = [
        {
            "role": "developer",
            "content": [
                {
                    "type": "text",
                    "text": system_message
                }
            ]
        },
        {
            "role": "developer",
            "content": [
                {
                    "type": "text",
                    "text": user_prompt
                }
            ]
        }
    ]
    
    try:
        # Call Azure OpenAI API with the appropriate parameters
        response = client.chat.completions.create(
            model=deployment,  # Use the deployment name defined at the top
            messages=messages,
            response_format={"type": "json_object"},
            max_completion_tokens=1000,  # Adjust as needed
            stream=False
        )
        
        # Parse the response
        evaluation = json.loads(response.choices[0].message.content)
        return evaluation
    except Exception as e:
        print(f"Error evaluating log with Azure OpenAI: {e}")
        return {
            "correctness": "error",
            "misunderstanding_count": 0,
            "category": "unknown",
            "self_confidence": "unknown",
            "explanation": f"Error during evaluation: {str(e)}"
        }

# Generate per-model evaluation files
def generate_model_evaluation_files(results, eval_dir, incremental=False):
    print("\nGenerating per-model evaluation files...")
    
    if not results:
        print("No results to summarize")
        return
    
    # Group evaluations by model
    model_evaluations = {}
    for result in results:
        model = result["model"]
        if model not in model_evaluations:
            model_evaluations[model] = []
        model_evaluations[model].append(result)
    
    # Process each model separately
    for model, evals in model_evaluations.items():
        # Create model-specific directory if it doesn't exist
        model_dir = os.path.join(eval_dir, model)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Generate detailed evaluation file for this model - one per log folder
        detailed_file = os.path.join(model_dir, "evaluation.txt")
        with open(detailed_file, 'w', encoding='utf-8') as f:
            f.write(f"DETAILED EVALUATIONS FOR {model}\n")
            f.write("=" * 80 + "\n\n")
            
            for i, eval_data in enumerate(evals):
                f.write(f"Evaluation {i+1}/{len(evals)}: {eval_data['task']}\n")
                f.write("-" * 60 + "\n")
                f.write(f"Task: {eval_data['task']}\n")
                f.write(f"Answer: {eval_data['answer']}\n\n")
                f.write(f"Correctness: {eval_data['correctness']}\n")
                f.write(f"Misunderstanding count: {eval_data['misunderstanding_count']}\n")
                f.write(f"Category: {eval_data['category']}\n")
                f.write(f"Self-confidence: {eval_data['self_confidence']}\n\n")
                f.write(f"Step count: {eval_data['step_count']}\n")
                f.write(f"Failure count: {eval_data['failure_count']}\n")
                f.write(f"Total tokens: {eval_data['total_tokens']} (Input: {eval_data['input_tokens']}, Output: {eval_data['output_tokens']})\n")
                f.write(f"Total time: {eval_data['total_time']:.2f} seconds\n\n")
                f.write(f"Explanation: {eval_data['explanation']}\n\n")
                f.write("-" * 80 + "\n\n")
        
        print(f"Evaluation file for {model} written to {detailed_file}")
        
        # Generate summary file for this model - one per log folder
        summary_file = os.path.join(model_dir, "summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            count = len(evals)
            correct = sum(1 for e in evals if e["correctness"] == "yes")
            partial = sum(1 for e in evals if e["correctness"] == "partial")
            incorrect = sum(1 for e in evals if e["correctness"] == "no")
            
            correct_pct = (correct + 0.5 * partial) / count * 100 if count > 0 else 0
            avg_steps = sum(e["step_count"] for e in evals) / count if count > 0 else 0
            avg_failures = sum(e["failure_count"] for e in evals) / count if count > 0 else 0
            avg_misunderstandings = sum(e["misunderstanding_count"] for e in evals) / count if count > 0 else 0
            avg_tokens = sum(e["total_tokens"] for e in evals) / count if count > 0 else 0
            avg_time = sum(e["total_time"] for e in evals) / count if count > 0 else 0
            
            # Count categories
            categories = {}
            for eval_data in evals:
                category = eval_data["category"]
                if category in categories:
                    categories[category] += 1
                else:
                    categories[category] = 1
            
            f.write(f"SUMMARY FOR {model}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total evaluations: {count}\n\n")
            f.write(f"Performance:\n")
            f.write(f"  Correct: {correct} ({correct/count*100:.1f}%)\n" if count > 0 else "  Correct: 0 (0.0%)\n")
            f.write(f"  Partially correct: {partial} ({partial/count*100:.1f}%)\n" if count > 0 else "  Partially correct: 0 (0.0%)\n")
            f.write(f"  Incorrect: {incorrect} ({incorrect/count*100:.1f}%)\n" if count > 0 else "  Incorrect: 0 (0.0%)\n")
            f.write(f"  Overall correctness score: {correct_pct:.1f}%\n\n")
            
            f.write(f"Efficiency:\n")
            f.write(f"  Average steps: {avg_steps:.1f}\n")
            f.write(f"  Average failures: {avg_failures:.1f}\n")
            f.write(f"  Average misunderstandings: {avg_misunderstandings:.1f}\n")
            f.write(f"  Average tokens: {avg_tokens:.1f}\n")
            f.write(f"  Average time: {avg_time:.1f} seconds\n\n")
            
            f.write(f"Categories:\n")
            for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {category}: {count} ({count/len(evals)*100:.1f}%)\n" if len(evals) > 0 else f"  {category}: 0 (0.0%)\n")
            
            # Find interesting patterns and observations
            f.write("\nInteresting Observations:\n")
            
            # Check if model has strengths in certain categories
            if categories:
                strongest_category = max(categories.items(), key=lambda x: x[1])[0]
                f.write(f"1. This model was tested most frequently on '{strongest_category}' tasks.\n")
            
            # Check if the model is particularly good or bad at certain tasks
            correct_by_category = {}
            for eval_data in evals:
                category = eval_data["category"]
                if category not in correct_by_category:
                    correct_by_category[category] = {"yes": 0, "partial": 0, "no": 0, "error": 0, "total": 0}
                
                correct_by_category[category]["total"] += 1
                # Handle the case when correctness might be 'error' or any other unexpected value
                correctness = eval_data["correctness"]
                if correctness in correct_by_category[category]:
                    correct_by_category[category][correctness] += 1
                else:
                    # If we get an unexpected value, count it as error
                    correct_by_category[category]["error"] += 1
            
            category_performance = {}
            for category, counts in correct_by_category.items():
                if counts["total"] >= 2:  # Only consider categories with at least 2 examples
                    score = (counts["yes"] + 0.5 * counts["partial"]) / counts["total"]
                    category_performance[category] = score
            
            if category_performance:
                best_category = max(category_performance.items(), key=lambda x: x[1])
                worst_category = min(category_performance.items(), key=lambda x: x[1])
                
                if best_category[1] > 0.7:  # More than 70% correct
                    f.write(f"2. The model performs particularly well on '{best_category[0]}' tasks ")
                    f.write(f"({best_category[1]*100:.1f}% correctness).\n")
                
                if worst_category[1] < 0.3 and worst_category[0] != best_category[0]:  # Less than 30% correct
                    f.write(f"3. The model struggles with '{worst_category[0]}' tasks ")
                    f.write(f"({worst_category[1]*100:.1f}% correctness).\n")
            
            # Check for correlation between steps and correctness
            correct_steps = [e["step_count"] for e in evals if e["correctness"] == "yes"]
            incorrect_steps = [e["step_count"] for e in evals if e["correctness"] == "no"]
            
            if correct_steps and incorrect_steps:
                avg_correct_steps = sum(correct_steps) / len(correct_steps)
                avg_incorrect_steps = sum(incorrect_steps) / len(incorrect_steps)
                
                if avg_correct_steps < avg_incorrect_steps * 0.7:  # Correct answers use significantly fewer steps
                    f.write(f"4. The model is more efficient when it knows the answer, ")
                    f.write(f"using {avg_correct_steps:.1f} steps for correct answers vs. ")
                    f.write(f"{avg_incorrect_steps:.1f} steps for incorrect answers.\n")
                elif avg_correct_steps > avg_incorrect_steps * 1.3:  # Correct answers use significantly more steps
                    f.write(f"4. The model is more thorough when providing correct answers, ")
                    f.write(f"using {avg_correct_steps:.1f} steps for correct answers vs. ")
                    f.write(f"{avg_incorrect_steps:.1f} steps for incorrect answers.\n")
            
            # Check for self-confidence patterns
            if len(evals) > 3:  # Only if we have enough samples
                # Self-confidence analysis (looking at questions that should have been answered without tools)
                high_knowledge_questions = [e for e in evals if e["self_confidence"] == "high"]
                if high_knowledge_questions:
                    tools_used = sum(1 for e in high_knowledge_questions if e["step_count"] > 1)  # More than 1 step means tools were used
                    
                    # Calculate percentage of high-knowledge questions where tools were unnecessarily used
                    tools_percentage = (tools_used / len(high_knowledge_questions)) * 100 if high_knowledge_questions else 0
                    
                    if tools_percentage > 70:
                        f.write(f"5. Low self-confidence: The model frequently used tools ({tools_percentage:.1f}% of the time) ")
                        f.write(f"for questions it should have known the answers to without external verification.\n")
                    elif tools_percentage < 30:
                        f.write(f"5. High self-confidence: The model appropriately answered {100-tools_percentage:.1f}% of common knowledge ")
                        f.write(f"questions directly without unnecessary tool use.\n")
                    else:
                        f.write(f"5. Medium self-confidence: The model used tools for {tools_percentage:.1f}% of questions ")
                        f.write(f"that could have been answered directly from its knowledge.\n")
        
        print(f"Summary for {model} written to {summary_file}")
    
    # Also create overall comparison file
    comparison_file = os.path.join(eval_dir, "model_comparison.txt")
    with open(comparison_file, 'w', encoding='utf-8') as f:
        f.write("MODEL COMPARISON SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Table of model performance stats
        f.write("Model Performance Metrics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model':<25} {'Tasks':<6} {'Correct %':<10} {'Steps':<8} {'Failures':<10} {'Misund.':<8} {'Tokens':<10} {'Time (s)':<10}\n")
        f.write("-" * 80 + "\n")
        
        # Calculate and display stats for each model
        model_stats = []
        for model, evals in model_evaluations.items():
            count = len(evals)
            correct = sum(1 for e in evals if e["correctness"] == "yes")
            partial = sum(1 for e in evals if e["correctness"] == "partial")
            incorrect = sum(1 for e in evals if e["correctness"] == "no")
            
            correct_pct = (correct + 0.5 * partial) / count * 100 if count > 0 else 0
            avg_steps = sum(e["step_count"] for e in evals) / count if count > 0 else 0
            avg_failures = sum(e["failure_count"] for e in evals) / count if count > 0 else 0
            avg_misunderstandings = sum(e["misunderstanding_count"] for e in evals) / count if count > 0 else 0
            avg_tokens = sum(e["total_tokens"] for e in evals) / count if count > 0 else 0
            avg_time = sum(e["total_time"] for e in evals) / count if count > 0 else 0
            
            model_stats.append({
                "model": model,
                "count": count,
                "correct_pct": correct_pct,
                "avg_steps": avg_steps,
                "avg_failures": avg_failures,
                "avg_misunderstandings": avg_misunderstandings,
                "avg_tokens": avg_tokens,
                "avg_time": avg_time
            })
        
        # Sort models by correctness percentage (descending)
        model_stats.sort(key=lambda x: x["correct_pct"], reverse=True)
        
        # Write the comparison table
        for stats in model_stats:
            f.write(f"{stats['model']:<25} {stats['count']:<6} {stats['correct_pct']:.1f}%      {stats['avg_steps']:.1f}    {stats['avg_failures']:.1f}        {stats['avg_misunderstandings']:.1f}      {stats['avg_tokens']:.1f}     {stats['avg_time']:.1f}\n")
    
    print(f"Model comparison written to {comparison_file}")

# Create or append to an evaluation file for a model
def update_evaluation_file(model_dir, model, eval_data):
    # Create model-specific directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Path to the evaluation file
    eval_file = os.path.join(model_dir, "evaluation.txt")
    
    # Check if file exists to determine if header is needed
    file_exists = os.path.exists(eval_file)
    
    # Open the file in append mode
    with open(eval_file, 'a', encoding='utf-8') as f:
        # Write header if this is a new file
        if not file_exists:
            f.write(f"DETAILED EVALUATIONS FOR {model}\n")
            f.write("=" * 80 + "\n\n")
        
        # Write evaluation data
        f.write(f"Evaluation: {eval_data['task']}\n")
        f.write("-" * 60 + "\n")
        f.write(f"Task: {eval_data['task']}\n")
        f.write(f"Answer: {eval_data['answer']}\n\n")
        f.write(f"Correctness: {eval_data['correctness']}\n")
        f.write(f"Misunderstanding count: {eval_data['misunderstanding_count']}\n")
        f.write(f"Category: {eval_data['category']}\n")
        f.write(f"Self-confidence: {eval_data['self_confidence']}\n\n")
        f.write(f"Step count: {eval_data['step_count']}\n")
        f.write(f"Failure count: {eval_data['failure_count']}\n")
        f.write(f"Total tokens: {eval_data['total_tokens']} (Input: {eval_data['input_tokens']}, Output: {eval_data['output_tokens']})\n")
        f.write(f"Total time: {eval_data['total_time']:.2f} seconds\n\n")
        f.write(f"Explanation: {eval_data['explanation']}\n\n")
        f.write("-" * 80 + "\n\n")

    print(f"Updated evaluation file for {model} at {eval_file}")

# Update the summary file for a model based on current results
def update_summary_file(model_dir, model, model_results):
    # Create model-specific directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    # Generate summary file for this model
    summary_file = os.path.join(model_dir, "summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        count = len(model_results)
        correct = sum(1 for e in model_results if e["correctness"] == "yes")
        partial = sum(1 for e in model_results if e["correctness"] == "partial")
        incorrect = sum(1 for e in model_results if e["correctness"] == "no")
        errors = sum(1 for e in model_results if e["correctness"] == "error")
        
        correct_pct = (correct + 0.5 * partial) / count * 100 if count > 0 else 0
        avg_steps = sum(e["step_count"] for e in model_results) / count if count > 0 else 0
        avg_failures = sum(e["failure_count"] for e in model_results) / count if count > 0 else 0
        avg_misunderstandings = sum(e["misunderstanding_count"] for e in model_results) / count if count > 0 else 0
        avg_tokens = sum(e["total_tokens"] for e in model_results) / count if count > 0 else 0
        avg_time = sum(e["total_time"] for e in model_results) / count if count > 0 else 0
        
        # Count categories
        categories = {}
        for eval_data in model_results:
            category = eval_data["category"]
            if category in categories:
                categories[category] += 1
            else:
                categories[category] = 1
        
        f.write(f"SUMMARY FOR {model}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total evaluations: {count}\n\n")
        f.write(f"Performance:\n")
        f.write(f"  Correct: {correct} ({correct/count*100:.1f}%)\n" if count > 0 else "  Correct: 0 (0.0%)\n")
        f.write(f"  Partially correct: {partial} ({partial/count*100:.1f}%)\n" if count > 0 else "  Partially correct: 0 (0.0%)\n")
        f.write(f"  Incorrect: {incorrect} ({incorrect/count*100:.1f}%)\n" if count > 0 else "  Incorrect: 0 (0.0%)\n")
        if errors > 0:
            f.write(f"  Errors: {errors} ({errors/count*100:.1f}%)\n" if count > 0 else "  Errors: 0 (0.0%)\n")
        f.write(f"  Overall correctness score: {correct_pct:.1f}%\n\n")
        
        f.write(f"Efficiency:\n")
        f.write(f"  Average steps: {avg_steps:.1f}\n")
        f.write(f"  Average failures: {avg_failures:.1f}\n")
        f.write(f"  Average misunderstandings: {avg_misunderstandings:.1f}\n")
        f.write(f"  Average tokens: {avg_tokens:.1f}\n")
        f.write(f"  Average time: {avg_time:.1f} seconds\n\n")
        
        f.write(f"Categories:\n")
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {category}: {count} ({count/len(model_results)*100:.1f}%)\n" if len(model_results) > 0 else f"  {category}: 0 (0.0%)\n")
        
    print(f"Updated summary for {model} at {summary_file}")

# Main function to evaluate all logs
def evaluate_all_logs(skip_evaluated=True):
    # Get OpenAI client
    client = get_openai_client()
    
    # Create evaluation directory if it doesn't exist
    eval_dir = "./evaluation"
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
        print(f"Created evaluation directory: {eval_dir}")
    
    # Find all log directories
    log_dirs = find_log_directories()
    print(f"Found {len(log_dirs)} log directories: {log_dirs}")
    
    # Prepare results storage
    all_results = []          # All results across all models
    model_results = {}        # Results organized by model
    
    # Track progress file to avoid re-evaluating logs
    progress_file = os.path.join(eval_dir, "evaluation_progress.json")
    evaluated_logs = {}
    
    # Load previously evaluated logs if the progress file exists
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                evaluated_logs = json.load(f)
            print(f"Loaded progress from {progress_file}")
            
            # Reconstruct model_results from evaluated_logs
            for model_name, logs in evaluated_logs.items():
                if model_name not in model_results:
                    model_results[model_name] = []
                for log_filename, result in logs.items():
                    model_results[model_name].append(result)
                    all_results.append(result)
                    
                # Update summary file with existing results
                if model_name in model_results and model_results[model_name]:
                    model_dir = os.path.join(eval_dir, model_name)
                    update_summary_file(model_dir, model_name, model_results[model_name])
                
        except Exception as e:
            print(f"Error loading progress file: {e}")
    
    # Process each directory
    for log_dir in log_dirs:
        print(f"\nProcessing directory: {log_dir}")
        log_files = get_log_files(log_dir)
        print(f"Found {len(log_files)} log files")
        
        model_name = os.path.basename(log_dir)
        
        # Create model entry in evaluated logs and model_results if it doesn't exist
        if model_name not in evaluated_logs:
            evaluated_logs[model_name] = {}
        if model_name not in model_results:
            model_results[model_name] = []
        
        # Create model directory for evaluation files
        model_dir = os.path.join(eval_dir, model_name)
        
        # Process each log file
        for i, log_file in enumerate(log_files):
            log_filename = os.path.basename(log_file)
            print(f"  Processing file {i+1}/{len(log_files)}: {log_filename}")
            
            # Skip if already evaluated and skip_evaluated is True
            if skip_evaluated and log_filename in evaluated_logs[model_name]:
                print(f"  Evaluation already exists, skipping: {log_filename}")
                continue
            
            try:
                # Parse the log file
                log_data = parse_log_file(log_file)
                
                # Evaluate with Azure OpenAI
                print("  Evaluating with Azure OpenAI (o3-mini)...")
                evaluation = evaluate_with_gpt(log_data, client)
                
                # Combine log data with evaluation
                result = {
                    "model": model_name,
                    "task": log_data["task"],
                    "answer": log_data["answer"],
                    "step_count": log_data["step_count"],
                    "failure_count": log_data["failure_count"],
                    "input_tokens": log_data["input_tokens"],
                    "output_tokens": log_data["output_tokens"],
                    "total_tokens": log_data["input_tokens"] + log_data["output_tokens"],
                    "total_time": log_data["total_time"],
                    "correctness": evaluation["correctness"],
                    "misunderstanding_count": evaluation["misunderstanding_count"],
                    "category": evaluation["category"],
                    "self_confidence": evaluation["self_confidence"],
                    "explanation": evaluation["explanation"]
                }
                
                # Add to results and track in progress
                all_results.append(result)
                model_results[model_name].append(result)
                evaluated_logs[model_name][log_filename] = result
                
                # Immediately write this evaluation to the model's evaluation file
                update_evaluation_file(model_dir, model_name, result)
                
                # Update the summary file with current results
                update_summary_file(model_dir, model_name, model_results[model_name])
                
                # Update progress file to save our work incrementally
                with open(progress_file, 'w', encoding='utf-8') as f:
                    json.dump(evaluated_logs, f, indent=2)
                
                print(f"  Evaluation complete: {evaluation['correctness']}")
                
                # Add a small delay to avoid hitting API rate limits
                time.sleep(1)
                
            except Exception as e:
                print(f"  Error processing {log_file}: {e}")
    
    # Write aggregate results to CSV
    output_file = os.path.join(eval_dir, "evaluation_results.csv")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        if all_results:
            fieldnames = all_results[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
    
    print(f"\nEvaluation complete. Results written to {output_file}")
    
    # Generate overall model comparison file
    if model_results:
        comparison_file = os.path.join(eval_dir, "model_comparison.txt")
        with open(comparison_file, 'w', encoding='utf-8') as f:
            f.write("MODEL COMPARISON SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            # Table of model performance stats
            f.write("Model Performance Metrics:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Model':<25} {'Tasks':<6} {'Correct %':<10} {'Steps':<8} {'Failures':<10} {'Misund.':<8} {'Tokens':<10} {'Time (s)':<10}\n")
            f.write("-" * 80 + "\n")
            
            # Calculate and display stats for each model
            model_stats = []
            for model, evals in model_results.items():
                count = len(evals)
                correct = sum(1 for e in evals if e["correctness"] == "yes")
                partial = sum(1 for e in evals if e["correctness"] == "partial")
                
                correct_pct = (correct + 0.5 * partial) / count * 100 if count > 0 else 0
                avg_steps = sum(e["step_count"] for e in evals) / count if count > 0 else 0
                avg_failures = sum(e["failure_count"] for e in evals) / count if count > 0 else 0
                avg_misunderstandings = sum(e["misunderstanding_count"] for e in evals) / count if count > 0 else 0
                avg_tokens = sum(e["total_tokens"] for e in evals) / count if count > 0 else 0
                avg_time = sum(e["total_time"] for e in evals) / count if count > 0 else 0
                
                model_stats.append({
                    "model": model,
                    "count": count,
                    "correct_pct": correct_pct,
                    "avg_steps": avg_steps,
                    "avg_failures": avg_failures,
                    "avg_misunderstandings": avg_misunderstandings,
                    "avg_tokens": avg_tokens,
                    "avg_time": avg_time
                })
            
            # Sort models by correctness percentage (descending)
            model_stats.sort(key=lambda x: x["correct_pct"], reverse=True)
            
            # Write the comparison table
            for stats in model_stats:
                f.write(f"{stats['model']:<25} {stats['count']:<6} {stats['correct_pct']:.1f}%      {stats['avg_steps']:.1f}    {stats['avg_failures']:.1f}        {stats['avg_misunderstandings']:.1f}      {stats['avg_tokens']:.1f}     {stats['avg_time']:.1f}\n")
        
        print(f"Model comparison written to {comparison_file}")

# Test function to evaluate a limited number of logs per folder
def test_evaluation(max_logs_per_folder=5):
    """Run a quick evaluation test with a limited number of logs per folder.
    
    Args:
        max_logs_per_folder: Maximum number of log files to process per folder
    """
    # Get OpenAI client
    client = get_openai_client()
    
    # Create evaluation directory if it doesn't exist
    eval_dir = "./evaluation_test"
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
        print(f"Created test evaluation directory: {eval_dir}")
    
    # Find all log directories
    log_dirs = find_log_directories()
    print(f"Found {len(log_dirs)} log directories: {log_dirs}")
    
    # Prepare results storage
    all_results = []          # All results across all models
    model_results = {}        # Results organized by model
    
    # Process each directory
    for log_dir in log_dirs:
        print(f"\nProcessing directory: {log_dir}")
        log_files = get_log_files(log_dir)[:max_logs_per_folder]  # Limit to max_logs_per_folder
        print(f"Testing with {len(log_files)} log files (limited to {max_logs_per_folder})")
        
        model_name = os.path.basename(log_dir)
        
        # Initialize model results if needed
        if model_name not in model_results:
            model_results[model_name] = []
            
        # Create model directory for evaluation files
        model_dir = os.path.join(eval_dir, model_name)
        
        # Process each log file
        for i, log_file in enumerate(log_files):
            log_filename = os.path.basename(log_file)
            print(f"  Processing file {i+1}/{len(log_files)}: {log_filename}")
            
            try:
                # Parse the log file
                log_data = parse_log_file(log_file)
                
                # Evaluate with Azure OpenAI
                print("  Evaluating with Azure OpenAI (o3-mini)...")
                evaluation = evaluate_with_gpt(log_data, client)
                
                # Combine log data with evaluation
                result = {
                    "model": model_name,
                    "task": log_data["task"],
                    "answer": log_data["answer"],
                    "step_count": log_data["step_count"],
                    "failure_count": log_data["failure_count"],
                    "input_tokens": log_data["input_tokens"],
                    "output_tokens": log_data["output_tokens"],
                    "total_tokens": log_data["input_tokens"] + log_data["output_tokens"],
                    "total_time": log_data["total_time"],
                    "correctness": evaluation["correctness"],
                    "misunderstanding_count": evaluation["misunderstanding_count"],
                    "category": evaluation["category"],
                    "self_confidence": evaluation["self_confidence"],
                    "explanation": evaluation["explanation"]
                }
                
                # Add to results
                all_results.append(result)
                model_results[model_name].append(result)
                
                # Immediately write this evaluation to the model's evaluation file
                update_evaluation_file(model_dir, model_name, result)
                
                # Update the summary file with current results
                update_summary_file(model_dir, model_name, model_results[model_name])
                
                print(f"  Evaluation complete: {evaluation['correctness']}")
                
                # Add a small delay to avoid hitting API rate limits
                time.sleep(1)
                
            except Exception as e:
                print(f"  Error processing {log_file}: {e}")
    
    # Write aggregate results to CSV
    output_file = os.path.join(eval_dir, "test_evaluation_results.csv")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        if all_results:
            fieldnames = all_results[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
    
    print(f"\nTest evaluation complete. Results written to {output_file}")
    
    # Generate overall model comparison file
    if model_results:
        comparison_file = os.path.join(eval_dir, "model_comparison.txt")
        with open(comparison_file, 'w', encoding='utf-8') as f:
            f.write("MODEL COMPARISON SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            # Table of model performance stats
            f.write("Model Performance Metrics:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Model':<25} {'Tasks':<6} {'Correct %':<10} {'Steps':<8} {'Failures':<10} {'Misund.':<8} {'Tokens':<10} {'Time (s)':<10}\n")
            f.write("-" * 80 + "\n")
            
            # Calculate and display stats for each model
            model_stats = []
            for model, evals in model_results.items():
                count = len(evals)
                correct = sum(1 for e in evals if e["correctness"] == "yes")
                partial = sum(1 for e in evals if e["correctness"] == "partial")
                
                correct_pct = (correct + 0.5 * partial) / count * 100 if count > 0 else 0
                avg_steps = sum(e["step_count"] for e in evals) / count if count > 0 else 0
                avg_failures = sum(e["failure_count"] for e in evals) / count if count > 0 else 0
                avg_misunderstandings = sum(e["misunderstanding_count"] for e in evals) / count if count > 0 else 0
                avg_tokens = sum(e["total_tokens"] for e in evals) / count if count > 0 else 0
                avg_time = sum(e["total_time"] for e in evals) / count if count > 0 else 0
                
                model_stats.append({
                    "model": model,
                    "count": count,
                    "correct_pct": correct_pct,
                    "avg_steps": avg_steps,
                    "avg_failures": avg_failures,
                    "avg_misunderstandings": avg_misunderstandings,
                    "avg_tokens": avg_tokens,
                    "avg_time": avg_time
                })
            
            # Sort models by correctness percentage (descending)
            model_stats.sort(key=lambda x: x["correct_pct"], reverse=True)
            
            # Write the comparison table
            for stats in model_stats:
                f.write(f"{stats['model']:<25} {stats['count']:<6} {stats['correct_pct']:.1f}%      {stats['avg_steps']:.1f}    {stats['avg_failures']:.1f}        {stats['avg_misunderstandings']:.1f}      {stats['avg_tokens']:.1f}     {stats['avg_time']:.1f}\n")
        
        print(f"Model comparison written to {comparison_file}")

# Run the evaluation if this script is executed directly
if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Evaluate agent logs and generate summaries.')
    parser.add_argument('--reevaluate', action='store_true', help='Reevaluate logs that have already been evaluated')
    parser.add_argument('--test', action='store_true', help='Run in test mode with limited logs')
    parser.add_argument('--max-logs', type=int, default=5, help='Maximum number of logs to process per folder in test mode')
    args = parser.parse_args()
    
    if args.test:
        print(f"Running in test mode with {args.max_logs} logs per folder")
        test_evaluation(args.max_logs)  # Run test with specified number of logs per folder
    else:
        # Run full evaluation with skip_evaluated set according to reevaluate flag
        evaluate_all_logs(skip_evaluated=not args.reevaluate)
    
    # Uncomment to generate evaluation files only
    # generate_model_evaluation_files([], "./evaluation")
