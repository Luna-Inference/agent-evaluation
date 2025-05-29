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
from openai import OpenAI
import time

# Load API key from secrets.json
def load_api_key():
    with open('secrets.json', 'r') as f:
        secrets = json.load(f)
        return secrets['openai']

# Initialize the OpenAI client with the API key
def get_openai_client():
    api_key = load_api_key()
    return OpenAI(api_key=api_key)

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
    
    # Count misunderstandings (this requires deeper analysis with GPT-4.1)
    misunderstanding_count = 0  # Will be evaluated by GPT-4.1
    
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

# Evaluate the log with GPT-4.1
def evaluate_with_gpt(log_data, client):
    prompt = f"""
    Evaluate the following agent interaction log based on these criteria:
    
    1. Correctness: Is the answer truthfully correct? (yes/no/partial)
    2. Number of steps: {log_data['step_count']} steps were taken to arrive at the final answer.
    3. Failure count: {log_data['failure_count']} code execution failures were detected.
    4. Misunderstanding count: How many times did the agent use a tool incorrectly? Review the log carefully.
    5. Token usage: {log_data['input_tokens']} input tokens and {log_data['output_tokens']} output tokens were used.
    6. Time: {log_data['total_time']:.2f} seconds total end-to-end time taken.
    7. Category: What category does this prompt fall into? (general knowledge, coding, math, etc.)
    8. Self-confidence: How confident should the model be in its own internal knowledge for this question? If this is general information, the model should rely on its own knowledge.
    
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
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are an expert evaluator of AI agent performance. Evaluate the provided log objectively and concisely."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        evaluation = json.loads(response.choices[0].message.content)
        return evaluation
    except Exception as e:
        print(f"Error evaluating log with GPT-4.1: {e}")
        return {
            "correctness": "error",
            "misunderstanding_count": 0,
            "category": "unknown",
            "self_confidence": "unknown",
            "explanation": f"Error during evaluation: {str(e)}"
        }

# Generate evaluation summaries
def generate_evaluation_summaries(results, eval_dir):
    print("\nGenerating evaluation summaries...")
    
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
    
    # Create combined evaluations file
    full_eval_file = os.path.join(eval_dir, "all_evaluations.txt")
    with open(full_eval_file, 'w', encoding='utf-8') as f:
        f.write("ALL MODEL EVALUATIONS\n")
        f.write("=" * 80 + "\n\n")
        
        # Write evaluations for each model
        for model, evals in model_evaluations.items():
            f.write(f"MODEL: {model}\n")
            f.write("-" * 80 + "\n\n")
            
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
            
            f.write("=" * 80 + "\n\n")
    
    print(f"All evaluations written to {full_eval_file}")
    
    # Create combined summary file
    summary_file = os.path.join(eval_dir, "model_comparison.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
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
            
            f.write(f"{model:<25} {count:<6} {correct_pct:.1f}%      {avg_steps:.1f}    {avg_failures:.1f}        {avg_misunderstandings:.1f}      {avg_tokens:.1f}     {avg_time:.1f}\n")
            
            # Store stats for detailed analysis
            stats = {
                "model": model,
                "count": count,
                "correct": correct,
                "partial": partial,
                "incorrect": incorrect,
                "correct_pct": correct_pct,
                "avg_steps": avg_steps,
                "avg_failures": avg_failures,
                "avg_misunderstandings": avg_misunderstandings,
                "avg_tokens": avg_tokens,
                "avg_time": avg_time,
                "categories": {}
            }
            
            # Count categories for this model
            for eval_data in evals:
                category = eval_data["category"]
                if category in stats["categories"]:
                    stats["categories"][category] += 1
                else:
                    stats["categories"][category] = 1
            
            model_stats.append(stats)
        
        # Sort models by correctness percentage (descending)
        model_stats.sort(key=lambda x: x["correct_pct"], reverse=True)
        
        # Detailed analysis for each model
        f.write("\n\nDETAILED MODEL ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        for stats in model_stats:
            model = stats["model"]
            count = stats["count"]
            evals = model_evaluations[model]
            
            f.write(f"MODEL: {model}\n")
            f.write("-" * 60 + "\n\n")
            
            f.write(f"Total evaluations: {count}\n\n")
            f.write(f"Performance:\n")
            f.write(f"  Correct: {stats['correct']} ({stats['correct']/count*100:.1f}%)\n" if count > 0 else "  Correct: 0 (0.0%)\n")
            f.write(f"  Partially correct: {stats['partial']} ({stats['partial']/count*100:.1f}%)\n" if count > 0 else "  Partially correct: 0 (0.0%)\n")
            f.write(f"  Incorrect: {stats['incorrect']} ({stats['incorrect']/count*100:.1f}%)\n" if count > 0 else "  Incorrect: 0 (0.0%)\n")
            f.write(f"  Overall correctness score: {stats['correct_pct']:.1f}%\n\n")
            
            f.write(f"Efficiency:\n")
            f.write(f"  Average steps: {stats['avg_steps']:.1f}\n")
            f.write(f"  Average failures: {stats['avg_failures']:.1f}\n")
            f.write(f"  Average misunderstandings: {stats['avg_misunderstandings']:.1f}\n")
            f.write(f"  Average tokens: {stats['avg_tokens']:.1f}\n")
            f.write(f"  Average time: {stats['avg_time']:.1f} seconds\n\n")
            
            f.write(f"Categories:\n")
            for category, count in sorted(stats["categories"].items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {category}: {count} ({count/stats['count']*100:.1f}%)\n" if stats['count'] > 0 else f"  {category}: 0 (0.0%)\n")
            
            # Find interesting patterns and observations
            f.write("\nInteresting Observations:\n")
            
            # Check if model has strengths in certain categories
            if stats["categories"]:
                strongest_category = max(stats["categories"].items(), key=lambda x: x[1])[0]
                f.write(f"1. This model was tested most frequently on '{strongest_category}' tasks.\n")
            
            # Check if the model is particularly good or bad at certain tasks
            correct_by_category = {}
            for eval_data in evals:
                category = eval_data["category"]
                if category not in correct_by_category:
                    correct_by_category[category] = {"correct": 0, "partial": 0, "incorrect": 0, "total": 0}
                
                correct_by_category[category]["total"] += 1
                correct_by_category[category][eval_data["correctness"]] += 1
            
            category_performance = {}
            for category, counts in correct_by_category.items():
                if counts["total"] >= 2:  # Only consider categories with at least 2 examples
                    score = (counts["correct"] + 0.5 * counts["partial"]) / counts["total"]
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
            
            # Check if the model tends to be overconfident
            if count > 3:  # Only if we have enough samples
                high_confidence = [e for e in evals if e["self_confidence"] == "high"]
                if high_confidence:
                    high_conf_correct = sum(1 for e in high_confidence if e["correctness"] == "yes")
                    high_conf_incorrect = sum(1 for e in high_confidence if e["correctness"] == "no")
                    
                    if high_conf_incorrect > high_conf_correct:  # More incorrect than correct with high confidence
                        f.write(f"5. The model shows signs of overconfidence. When highly confident, ")
                        f.write(f"it's incorrect more often than it's correct.\n")
                    elif high_conf_correct > high_conf_incorrect * 3:  # Much more correct than incorrect with high confidence
                        f.write(f"5. The model's confidence is well-calibrated. When highly confident, ")
                        f.write(f"it's usually correct.\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
    
    print(f"Model comparison summary written to {summary_file}")

# Main function to evaluate all logs
def evaluate_all_logs():
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
    results = []
    
    # Track progress file to avoid re-evaluating logs
    progress_file = os.path.join(eval_dir, "evaluation_progress.json")
    evaluated_logs = {}
    
    # Load previously evaluated logs if the progress file exists
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                evaluated_logs = json.load(f)
            print(f"Loaded progress from {progress_file}")
        except Exception as e:
            print(f"Error loading progress file: {e}")
    
    # Process each directory
    for log_dir in log_dirs:
        print(f"\nProcessing directory: {log_dir}")
        log_files = get_log_files(log_dir)
        print(f"Found {len(log_files)} log files")
        
        model_name = os.path.basename(log_dir)
        
        # Create model entry in evaluated logs if it doesn't exist
        if model_name not in evaluated_logs:
            evaluated_logs[model_name] = {}
        
        # Process each log file
        for i, log_file in enumerate(log_files):
            log_filename = os.path.basename(log_file)
            print(f"  Processing file {i+1}/{len(log_files)}: {log_filename}")
            
            # Skip if already evaluated
            if log_filename in evaluated_logs[model_name]:
                print(f"  Evaluation already exists, skipping: {log_filename}")
                results.append(evaluated_logs[model_name][log_filename])
                continue
            
            try:
                # Parse the log file
                log_data = parse_log_file(log_file)
                
                # Evaluate with GPT-4.1
                print("  Evaluating with GPT-4.1...")
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
                results.append(result)
                evaluated_logs[model_name][log_filename] = result
                
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
        if results:
            fieldnames = results[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    
    print(f"\nEvaluation complete. Results written to {output_file}")
    
    # Generate combined evaluation summaries
    generate_evaluation_summaries(results, eval_dir)

# Run the evaluation if this script is executed directly
if __name__ == "__main__":
    evaluate_all_logs()

