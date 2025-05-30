import os
import datetime
import sys
import traceback
import re
import json
from io import StringIO
from smolagents import CodeAgent, LiteLLMModel

# Disable colored output for most libraries
os.environ["NO_COLOR"] = "1"
os.environ["CLICOLOR"] = "0"
os.environ["PY_COLORS"] = "0"

# Function to strip ANSI color codes from text
def strip_ansi_codes(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

# Define log directory - but don't redirect output programmatically
# Instead, use shell redirection: python loop.py >> /home/thomas/agents/log/output.txt

# Model setup (same as run.py)
model = LiteLLMModel(
    model_id="ollama_chat/mistral",
    api_base="http://localhost:11434",
    api_key="YOUR_API_KEY",
    num_ctx=30000,
)

agent = CodeAgent(tools=[], model=model, add_base_tools=True)
agent.logger.level = 2

# Load tasks from structured task.json file
try:
    with open('task-subset.json', 'r') as f:
        TASK_DATA = json.load(f)
    
    # Extract tasks from structured format
    TASKS = []
    TASK_METADATA = []
    
    for category_key, category_data in TASK_DATA.items():
        category_name = category_data.get('category', category_key)
        
        for example in category_data.get('examples', []):
            task_type = example.get('task', 'unknown')
            difficulty = example.get('difficulty', 'medium')
            prompt = example.get('prompt', '')
            
            if prompt:
                TASKS.append(prompt)
                TASK_METADATA.append({
                    'category': category_name,
                    'task_type': task_type,
                    'difficulty': difficulty
                })
    
    print(f"Successfully loaded {len(TASKS)} tasks from task.json across {len(TASK_DATA)} categories")
except Exception as e:
    print(f"Error loading tasks from task.json: {e}")
    # Fallback to a small set of tasks in case of error
    TASKS = [
        "What is the capital of France?",
        "Who wrote 'Pride and Prejudice'?",
        "What is the square root of 256?"
    ]
    TASK_METADATA = [
        {'category': 'Knowledge Domains', 'task_type': 'general-knowledge', 'difficulty': 'easy'},
        {'category': 'Knowledge Domains', 'task_type': 'general-knowledge', 'difficulty': 'medium'},
        {'category': 'Mathematics & Quantitative Reasoning', 'task_type': 'basic-arithmetic', 'difficulty': 'easy'}
    ]
    print(f"Using {len(TASKS)} fallback tasks instead")


# Create log directory if it doesn't exist
LOG_DIR = "./log19 - mistral"
os.makedirs(LOG_DIR, exist_ok=True)

# Generate a unique log filename for each task
def get_log_filename(task_idx):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    task_type = TASK_METADATA[task_idx].get('task_type', 'unknown').replace('-', '_')
    return os.path.join(LOG_DIR, f"task_{task_idx+1}_{task_type}_{timestamp}.txt")

# Create a log capture class that will capture and clean output
class LogCapture:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = open(log_file, 'w')
        self.string_buffer = StringIO()
        
    def write(self, message):
        self.terminal.write(message)  # Display on terminal
        self.string_buffer.write(message)  # Buffer for cleaning
        
    def flush(self):
        self.terminal.flush()
        
    def close(self):
        # Clean ANSI codes and write to file
        self.log_file.write(strip_ansi_codes(self.string_buffer.getvalue()))
        self.log_file.close()

print("\n" + "*"*80)
print("Starting tasks execution")
print("*"*80)

try:
    for idx, task in enumerate(TASKS):
        # Create a log file for this task
        log_file = get_log_filename(idx)
        # Save original stdout/stderr
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        
        # Set up logging for this task
        log_capture = LogCapture(log_file)
        sys.stdout = log_capture
        sys.stderr = log_capture
        
        try:
            # Print task info with metadata
            metadata = TASK_METADATA[idx]
            print(f"\n{'='*80}")
            print(f"Task {idx+1}/{len(TASKS)}: {task}")
            print(f"Category: {metadata['category']} | Type: {metadata['task_type']} | Difficulty: {metadata['difficulty']}")
            print(f"{'='*80}\n")
            
            # Run the task and capture output
            try:
                output = agent.run(task)
                print(f"\nOutput:\n{output}\n")
            except Exception as e:
                print(f"\nERROR: {e}\n")
                print(traceback.format_exc())
                
            print(f"Task {idx+1} complete.")
            print(f"{'-'*60}")
            
        finally:
            # Restore original stdout/stderr
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            log_capture.close()
            print(f"Task {idx+1} complete. Log saved to {log_file}")
            
except Exception as main_exc:
    print("FATAL ERROR in loop.py:")
    print(traceback.format_exc())