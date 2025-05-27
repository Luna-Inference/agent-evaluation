import os
import datetime
import sys
import traceback
import re
import socket
import uuid
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
    model_id="ollama_chat/qwen3:1.7b",
    api_base="http://localhost:11434",
    api_key="YOUR_API_KEY",
    num_ctx=8192,
)

agent = CodeAgent(tools=[], model=model, add_base_tools=True)
agent.logger.level = 2

# Define your tasks here; you can modify this list
TASKS = [
    "What is the capital of France?",
    "Who wrote 'Pride and Prejudice'?",
    "What is the square root of 256?",
    "Summarize the theory of relativity in one sentence.",
    "List the first 5 prime numbers.",
    "Translate 'Good morning' to Spanish.",
    "What is the chemical symbol for gold?",
    "Who painted the Mona Lisa?",
    "What is the distance from Earth to the Moon?",
    "Give me a haiku about spring.",
    "What is the boiling point of water in Celsius?",
    "Who was the first person to walk on the moon?",
    "What is the largest mammal?",
    "Explain quantum entanglement simply.",
    "What is the capital city of Japan?",
    "Name three programming languages.",
    "What is the tallest mountain in the world?",
    "Who discovered penicillin?",
    "What is the speed of light?",
    "Give a synonym for 'happy'.",
    "What is 15 multiplied by 7?",
    "Write a short poem about the ocean.",
    "What year did World War II end?",
    "Who is known as the father of computers?",
    "What is the currency of Brazil?",
    "Define 'photosynthesis'.",
    "What is the largest planet in our solar system?",
    "Who wrote '1984'?",
    "What is the freezing point of water in Fahrenheit?",
    "Name a famous Greek philosopher.",
    "What is the formula for the area of a circle?",
    "What is the smallest prime number?",
    "Who was the first President of the United States?",
    "What is the chemical formula for water?",
    "Name a mammal that can fly.",
    "What is the capital of Canada?",
    "What is 12 squared?",
    "Who painted the ceiling of the Sistine Chapel?",
    "What is the main ingredient in guacamole?",
    "What is the largest desert in the world?",
    "Who is the author of 'The Hobbit'?",
    "What is the Pythagorean theorem?",
    "What is the value of pi to three decimal places?",
    "Name a country that starts with 'Z'.",
    "What is the longest river in the world?",
    "Who invented the telephone?",
    "What is the capital of Australia?",
    "What is the hardest natural substance?",
    "Name three types of clouds.",
    "What is the atomic number of carbon?",
    "Who is the Greek god of the sea?",
    "What is the national language of Brazil?",
    "What is the sum of the angles in a triangle?",
    "Name a famous composer from the Classical period.",
    "What is the process by which plants make food?",
    "Who was the first female Nobel laureate?",
    "What is the main gas found in the air we breathe?",
    "What is the capital of Egypt?",
    "What is the largest ocean on Earth?",
    "Who wrote 'Hamlet'?",
    "What is the symbol for potassium?",
    "What is the square root of 144?",
    "Name a continent that lies entirely in the Southern Hemisphere.",
    "Who is known as the 'Maid of Orléans'?",
    "What is the chemical symbol for silver?",
    "What is the most abundant element in the universe?",
    "Who painted 'Starry Night'?",
    "What is the capital of South Korea?",
    "What is the main ingredient in hummus?",
    "What is the tallest building in the world?",
    "Who discovered gravity?",
    "What is the capital of Italy?",
    "What is the process of cell division called?",
    "What is the largest island in the world?",
    "Who is the author of 'To Kill a Mockingbird'?",
    "What is the freezing point of water in Celsius?",
    "What is the capital of Russia?",
    "What is the currency of Japan?",
    "Who invented the light bulb?",
    "What is the main language spoken in Egypt?",
    "What is the largest bone in the human body?",
    "Who is the Roman god of war?",
    "What is the capital of Mexico?",
    "What is the main ingredient in sushi?",
    "What is the largest continent?",
    "Who wrote 'The Odyssey'?",
    "What is the chemical formula for table salt?",
    "What is the capital of Germany?",
    "What is the process by which water changes from liquid to gas?",
    "Who is the founder of Microsoft?",
    "What is the capital of India?",
    "What is the hardest known material?",
    "Who is the author of 'Moby Dick'?",
    "What is the currency of the United Kingdom?",
    "What is the capital of Argentina?",
    "What is the main ingredient in paella?",
    "What is the tallest animal in the world?",
    "Who discovered America?",
    "What is the capital of Spain?",
    "What is the chemical symbol for iron?",
    "What is the main ingredient in pizza?",
    "Who is the author of 'Don Quixote'?",
    "What is the capital of Turkey?",
    "What is the largest lake in Africa?",
    "Who is the Greek goddess of wisdom?",
    "What is the boiling point of water in Fahrenheit?",
    "What is the main ingredient in falafel?",
    "What is the capital of Sweden?",
    "What is the process by which ice turns to water?",
    "Who is the author of 'The Divine Comedy'?",
    "What is the capital of Norway?",
    "What is the main ingredient in ratatouille?",
]


# Create a unique log directory for this run
BASE_LOG_DIR = "/home/thomas/agents/log"

# Generate a unique log directory name that's distinctive across machines
def create_unique_log_dir():
    # Get hostname to differentiate between machines
    hostname = socket.gethostname()
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Add a unique ID for extra uniqueness
    unique_id = str(uuid.uuid4())[:8]
    # Create the unique directory name
    log_dir_name = f"{hostname}_{timestamp}_{unique_id}"
    # Full path
    log_dir = os.path.join(BASE_LOG_DIR, log_dir_name)
    # Create the directory
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

# Create the unique log directory for this run
LOG_DIR = create_unique_log_dir()

# Generate a log filename for each task within the unique log directory
def get_log_filename(task_idx):
    return os.path.join(LOG_DIR, f"task_{task_idx+1}.txt")

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
            # Print task info
            print(f"\n{'='*80}")
            print(f"Task {idx+1}/{len(TASKS)}: {task}")
            print(f"{'='*80}\n")
            
            # Run the task and capture output
            response = agent.generate(task)
            
            # Print response
            print(f"\nResponse:")
            print(f"{'-'*80}")
            print(response)
            print(f"{'-'*80}")
            
        except Exception as e:
            print(f"Error executing task: {e}")
            traceback.print_exc()
        finally:
            # Reset stdout/stderr
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            # Close the log capture
            log_capture.close()
            
            # Print progress
            print(f"Completed task {idx+1}/{len(TASKS)}")
            
except KeyboardInterrupt:
    print("\nProcess interrupted by user")
except Exception as e:
    print(f"Fatal error: {e}")
    traceback.print_exc()
finally:
    print("\n" + "*"*80)
    print(f"Execution completed. Logs stored in: {LOG_DIR}")
    print("*"*80)
