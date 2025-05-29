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
import re
import json
import glob
import argparse
from datetime import datetime
import time
from typing import Dict, List, Any, Optional, Tuple

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI package not installed. Run 'pip install openai' to install it.")

# Load config file
def load_config(config_path: str = "secrets.json") -> Dict[str, str]:
    """Load API keys from secrets.json file"""
    config = {}
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"Config loaded successfully from {config_path}")
        else:
            print(f"Config file {config_path} not found")
    except Exception as e:
        print(f"Error loading config: {str(e)}")
    
    return config

# OpenAI client will be initialized later when API key is available

class LogEvaluator:
    """Agent for evaluating logs based on specified requirements"""
    
    def __init__(self, log_dir: str):
        """Initialize the evaluator with the path to log files"""
        self.log_dir = log_dir
        self.log_files = self._get_log_files()
        self.results = {}
        
    def _get_log_files(self) -> List[str]:
        """Get all log files from the directory"""
        return glob.glob(os.path.join(self.log_dir, "task_*.txt"))
    
    def parse_log(self, log_file: str) -> Dict[str, Any]:
        """Parse a log file to extract relevant information including task metadata"""
        with open(log_file, 'r') as f:
            content = f.read()
            
        # Extract task ID and question
        task_match = re.search(r"Task (\d+)/\d+: (.+?)\n", content)
        task_id = task_match.group(1) if task_match else None
        question = task_match.group(2) if task_match else None
        
        # Extract task metadata (category, task type, difficulty)
        metadata_match = re.search(r"Category: (.+?) \| Type: (.+?) \| Difficulty: (.+?)\n", content)
        category = metadata_match.group(1) if metadata_match else "unknown"
        task_type = metadata_match.group(2) if metadata_match else "unknown"
        difficulty = metadata_match.group(3) if metadata_match else "unknown"
        
        # Try to extract task type from the filename if not found in content
        if task_type == "unknown":
            filename_match = re.search(r"task_\d+_([a-zA-Z_]+)_\d+", os.path.basename(log_file))
            if filename_match:
                task_type = filename_match.group(1)
        
        # Extract answer
        answer_match = re.search(r"Output:\s*\n(.+?)\n\nTask \d+ complete\.\n", content, re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else None
        
        # Extract number of steps
        step_count = len(re.findall(r"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step (\d+) ━━━", content))
        
        # Extract token usage
        token_matches = re.findall(r"Input tokens: (\d+) \| Output tokens: (\d+)", content)
        input_tokens = sum(int(match[0]) for match in token_matches) if token_matches else 0
        output_tokens = sum(int(match[1]) for match in token_matches) if token_matches else 0
        
        # Extract time info
        duration_matches = re.findall(r"Duration (\d+\.\d+) seconds", content)
        total_time = sum(float(match) for match in duration_matches) if duration_matches else 0
        
        # Check for failures
        failure_count = len(re.findall(r"Error:|Failed to execute", content))
        
        # Analyze for misunderstanding (tool usage errors)
        misunderstanding_count = len(re.findall(r"tool not found|invalid argument|TypeError:|AttributeError:", content))
        
        return {
            "task_id": task_id,
            "question": question,
            "answer": answer,
            "category": category,
            "task_type": task_type,
            "difficulty": difficulty,
            "step_count": step_count,
            "failure_count": failure_count,
            "misunderstanding_count": misunderstanding_count,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "total_time": total_time,
            "raw_content": content
        }
    
    def evaluate_correctness(self, question: str, answer: str, client=None, model_name="gpt-4") -> Dict[str, Any]:
        """Use the specified OpenAI model to evaluate the correctness of an answer"""
        # If no OpenAI client is available, return a dummy evaluation
        if not OPENAI_AVAILABLE or client is None:
            print(f"Warning: Skipping OpenAI evaluation for question: '{question}'")
            return {
                "correctness": None,
                "category": "unknown",
                "self_confidence": None,
                "explanation": "No OpenAI evaluation available"
            }

        prompt = f"""
        You are an expert evaluator. Please analyze the following question and answer pair:
        
        Question: {question}
        Answer: {answer}
        
        Please evaluate the following aspects and provide your assessment in JSON format:
        
        1. Correctness: Is the answer truthfully correct? (true/false)
        2. Category: What category does this question fall into? (general knowledge, math, coding, science, history, etc.)
        3. Self-confidence: Should the model have relied on its own knowledge for this question? (true/false)
        
        Provide your response in the following JSON format:
        {{"correctness": true/false, "category": "category_name", "self_confidence": true/false, "explanation": "brief explanation"}}
        """
        
        try:
            # Define parameters without response_format
            params = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are an expert evaluator that judges the quality and correctness of answers. You MUST respond with ONLY valid JSON in the format {\"correctness\": true/false, \"category\": \"category_name\", \"self_confidence\": true/false, \"explanation\": \"brief explanation\"}"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1
            }
            
            # Make the API call without response_format
            response = client.chat.completions.create(**params)
            
            evaluation = json.loads(response.choices[0].message.content)
            return evaluation
        except Exception as e:
            print(f"Error during OpenAI evaluation: {str(e)}")
            return {
                "correctness": None,
                "category": "unknown",
                "self_confidence": None,
                "explanation": f"Evaluation error: {str(e)}"
            }
    
    def evaluate_all_logs(self, openai_api_key=None, model_name="gpt-4", skip_eval=False) -> Dict[str, Dict[str, Any]]:
        """Evaluate all log files and return structured results"""
        results = {}
        client = None
        
        # Initialize OpenAI client if API key is provided and evaluation is not skipped
        if OPENAI_AVAILABLE and openai_api_key and not skip_eval:
            try:
                client = OpenAI(api_key=openai_api_key)
                print(f"OpenAI client initialized successfully (using model: {model_name})")
            except Exception as e:
                print(f"Failed to initialize OpenAI client: {str(e)}")
        elif skip_eval:
            print("Skipping LLM evaluations as requested")
        elif not OPENAI_AVAILABLE:
            print("OpenAI package not available, skipping LLM evaluations")
        elif not openai_api_key:
            print("No API key provided, skipping LLM evaluations")
        
        for log_file in self.log_files:
            try:
                print(f"Processing {os.path.basename(log_file)}...")
                log_data = self.parse_log(log_file)
                
                if log_data["question"] and log_data["answer"]:
                    # Get LLM evaluation
                    evaluation = self.evaluate_correctness(
                        log_data["question"],
                        log_data["answer"],
                        client,
                        model_name=model_name
                    )
                    
                    # Combine data
                    result = {
                        "question": log_data["question"],
                        "answer": log_data["answer"],
                        "correctness": evaluation.get("correctness"),
                        "category": evaluation.get("category", "unknown"),
                        "self_confidence": evaluation.get("self_confidence"),
                        "explanation": evaluation.get("explanation", ""),
                        "step_count": log_data["step_count"],
                        "failure_count": log_data["failure_count"],
                        "misunderstanding_count": log_data["misunderstanding_count"],
                        "input_tokens": log_data["input_tokens"],
                        "output_tokens": log_data["output_tokens"],
                        "total_tokens": log_data["total_tokens"],
                        "total_time": log_data["total_time"]
                    }
                    
                    task_id = log_data["task_id"]
                    results[task_id] = result
                    
            except Exception as e:
                print(f"Error processing {log_file}: {str(e)}")
                
        self.results = results
        return results
    
    def save_results(self, output_path: str = "evaluation_results.json") -> None:
        """Save evaluation results to a JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {output_path}")
        
    def generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of all evaluation results with detailed analytics"""
        if not self.results:
            return {}
            
        total_tasks = len(self.results)
        correct_answers = sum(1 for r in self.results.values() if r["correctness"])
        avg_steps = sum(r["step_count"] for r in self.results.values()) / total_tasks if total_tasks else 0
        avg_failures = sum(r["failure_count"] for r in self.results.values()) / total_tasks if total_tasks else 0
        avg_misunderstandings = sum(r["misunderstanding_count"] for r in self.results.values()) / total_tasks if total_tasks else 0
        avg_tokens = sum(r["total_tokens"] for r in self.results.values()) / total_tasks if total_tasks else 0
        avg_time = sum(r["total_time"] for r in self.results.values()) / total_tasks if total_tasks else 0
        
        # Categorize results by category, task type, and difficulty
        categories = {}
        task_types = {}
        difficulties = {}
        
        # Create more detailed analytics
        category_correctness = {}
        difficulty_correctness = {}
        task_type_correctness = {}
        
        for r in self.results.values():
            # Basic categorization
            cat = r.get("category", "unknown")
            task_type = r.get("task_type", "unknown")
            difficulty = r.get("difficulty", "unknown")
            
            # Count by category
            if cat not in categories:
                categories[cat] = 0
            categories[cat] += 1
            
            # Count by task type
            if task_type not in task_types:
                task_types[task_type] = 0
            task_types[task_type] += 1
            
            # Count by difficulty
            if difficulty not in difficulties:
                difficulties[difficulty] = 0
            difficulties[difficulty] += 1
            
            # Correctness by category
            if cat not in category_correctness:
                category_correctness[cat] = {"correct": 0, "total": 0}
            category_correctness[cat]["total"] += 1
            if r.get("correctness", False):
                category_correctness[cat]["correct"] += 1
                
            # Correctness by difficulty
            if difficulty not in difficulty_correctness:
                difficulty_correctness[difficulty] = {"correct": 0, "total": 0}
            difficulty_correctness[difficulty]["total"] += 1
            if r.get("correctness", False):
                difficulty_correctness[difficulty]["correct"] += 1
                
            # Correctness by task type
            if task_type not in task_type_correctness:
                task_type_correctness[task_type] = {"correct": 0, "total": 0}
            task_type_correctness[task_type]["total"] += 1
            if r.get("correctness", False):
                task_type_correctness[task_type]["correct"] += 1
        
        # Calculate accuracy percentages
        for cat in category_correctness:
            if category_correctness[cat]["total"] > 0:
                category_correctness[cat]["accuracy"] = category_correctness[cat]["correct"] / category_correctness[cat]["total"]
            else:
                category_correctness[cat]["accuracy"] = 0
                
        for diff in difficulty_correctness:
            if difficulty_correctness[diff]["total"] > 0:
                difficulty_correctness[diff]["accuracy"] = difficulty_correctness[diff]["correct"] / difficulty_correctness[diff]["total"]
            else:
                difficulty_correctness[diff]["accuracy"] = 0
                
        for tt in task_type_correctness:
            if task_type_correctness[tt]["total"] > 0:
                task_type_correctness[tt]["accuracy"] = task_type_correctness[tt]["correct"] / task_type_correctness[tt]["total"]
            else:
                task_type_correctness[tt]["accuracy"] = 0
        
        summary = {
            "total_tasks": total_tasks,
            "correct_answers": correct_answers,
            "accuracy": correct_answers / total_tasks if total_tasks else 0,
            "average_steps": avg_steps,
            "average_failures": avg_failures,
            "average_misunderstandings": avg_misunderstandings,
            "average_tokens": avg_tokens,
            "average_time": avg_time,
            "categories": categories,
            "task_types": task_types,
            "difficulties": difficulties,
            "category_performance": category_correctness,
            "difficulty_performance": difficulty_correctness,
            "task_type_performance": task_type_correctness
        }
        
        return summary

def main():
    """Main function to run the evaluation"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate agent logs using OpenAI models")
    parser.add_argument('--log_dir', default="/root/agent-evaluation/log", 
                        help="Directory containing log files (default: /root/agent-evaluation/log)")
    parser.add_argument('--api_key', default=None,
                        help="OpenAI API key for evaluation (if not provided, will check secrets.json or OPENAI_API_KEY env variable)")
    parser.add_argument('--config', default="secrets.json",
                        help="Path to the config file containing API keys (default: secrets.json)")
    parser.add_argument('--model', default="gpt-4",
                        help="OpenAI model to use for evaluations (default: gpt-4)")
    parser.add_argument('--output', default="evaluation_results.json",
                        help="Output file path for detailed results (default: evaluation_results.json)")
    parser.add_argument('--summary', default="evaluation_summary.json",
                        help="Output file path for summary results (default: evaluation_summary.json)")
    parser.add_argument('--sample', type=int, default=None,
                        help="Evaluate only a sample of N log files (default: all files)")
    parser.add_argument('--skip-eval', action='store_true',
                        help="Skip OpenAI evaluation even if API key is available (for faster processing without LLM judgments)")

    
    args = parser.parse_args()
    
    # Load config file for API keys
    config = load_config(args.config)
    
    # Priority for API key: command line > config file > environment variable
    openai_api_key = args.api_key or config.get("openai") or os.environ.get("OPENAI_API_KEY")
    
    if openai_api_key and OPENAI_AVAILABLE and not args.skip_eval:
        print(f"OpenAI API key found and will be used for evaluations with model: {args.model}")
    elif args.skip_eval:
        print("OpenAI evaluations will be skipped as requested")
    elif not openai_api_key and OPENAI_AVAILABLE:
        print("Warning: No OpenAI API key found. Evaluations will be limited.")
        print("To use OpenAI for evaluations, add your key to secrets.json or set it via --api_key")

    
    # Initialize evaluator
    evaluator = LogEvaluator(args.log_dir)
    
    # Limit sample size if requested
    if args.sample and args.sample > 0 and args.sample < len(evaluator.log_files):
        import random
        random.shuffle(evaluator.log_files)
        evaluator.log_files = evaluator.log_files[:args.sample]
        print(f"Evaluating a sample of {args.sample} log files")
    
    print("Starting evaluation of logs...")
    results = evaluator.evaluate_all_logs(openai_api_key, model_name=args.model, skip_eval=args.skip_eval)
    
    # Save detailed results
    evaluator.save_results(args.output)
    
    # Generate and save summary
    summary = evaluator.generate_summary()
    with open(args.summary, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {args.summary}")
    
    # Print summary to console
    print("\nEvaluation Summary:")
    print(f"Total tasks evaluated: {summary.get('total_tasks', 0)}")
    
    if 'correct_answers' in summary and 'total_tasks' in summary and summary['total_tasks'] > 0:
        print(f"Correct answers: {summary['correct_answers']} ({summary['accuracy']:.2%})")
    
    print(f"Average steps per task: {summary.get('average_steps', 0):.2f}")
    print(f"Average failures per task: {summary.get('average_failures', 0):.2f}")
    print(f"Average misunderstandings per task: {summary.get('average_misunderstandings', 0):.2f}")
    print(f"Average tokens per task: {summary.get('average_tokens', 0):.2f}")
    print(f"Average time per task: {summary.get('average_time', 0):.2f} seconds")
    
    # Print category distribution
    if 'categories' in summary and summary.get('total_tasks', 0) > 0:
        print("\nCategories:")
        for category, count in sorted(summary["categories"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {category}: {count} tasks ({count / summary['total_tasks']:.2%})")
    
    # Print task type distribution
    if 'task_types' in summary and summary.get('total_tasks', 0) > 0:
        print("\nTask Types:")
        for task_type, count in sorted(summary["task_types"].items(), key=lambda x: x[1], reverse=True)[:10]:  # Top 10
            print(f"  {task_type}: {count} tasks ({count / summary['total_tasks']:.2%})")
        if len(summary["task_types"]) > 10:
            print(f"  ...and {len(summary['task_types']) - 10} more task types")
    
    # Print difficulty distribution
    if 'difficulties' in summary and summary.get('total_tasks', 0) > 0:
        print("\nDifficulty Levels:")
        for diff, count in sorted(summary["difficulties"].items(), key=lambda x: {"easy": 0, "medium": 1, "difficult": 2}.get(x[0], 3)):
            print(f"  {diff}: {count} tasks ({count / summary['total_tasks']:.2%})")
    
    # Print performance by difficulty
    if 'difficulty_performance' in summary:
        print("\nPerformance by Difficulty:")
        for diff, data in sorted(summary["difficulty_performance"].items(), key=lambda x: {"easy": 0, "medium": 1, "difficult": 2}.get(x[0], 3)):
            if data.get("total", 0) > 0:
                print(f"  {diff}: {data.get('correct', 0)}/{data.get('total', 0)} correct ({data.get('accuracy', 0):.2%})")
    
    # Print performance by category (top 5 categories)
    if 'category_performance' in summary:
        print("\nPerformance by Category (top categories):")
        top_categories = sorted(summary["category_performance"].items(), key=lambda x: x[1].get("total", 0), reverse=True)[:5]
        for cat, data in top_categories:
            if data.get("total", 0) > 0:
                print(f"  {cat}: {data.get('correct', 0)}/{data.get('total', 0)} correct ({data.get('accuracy', 0):.2%})")
        if len(summary["category_performance"]) > 5:
            print(f"  ...and {len(summary['category_performance']) - 5} more categories")

    return summary

if __name__ == "__main__":
    main()
