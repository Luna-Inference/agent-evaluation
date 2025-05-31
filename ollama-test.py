#!/usr/bin/env python3
import os
import sys
import argparse
from openai import OpenAI

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Sample OpenAI client using Ollama configuration")
    parser.add_argument("--prompt", type=str, default="Explain quantum computing in simple terms", 
                       help="The prompt to send to the model")
    args = parser.parse_args()

    # Ollama configuration from loop-mac-mini.py
    api_base = "http://100.114.56.51:11434"
    model_id = "smollm2:latest"  # Using just the model name without the ollama_chat/ prefix
    
    # Initialize the OpenAI client with the Ollama API base
    client = OpenAI(
        base_url=f"{api_base}/v1",  # OpenAI-compatible endpoint
        api_key="YOUR_API_KEY"      # Ollama typically doesn't require an API key, but the client expects one
    )
    
    print(f"Sending prompt to {model_id} at {api_base}...")
    print(f"Prompt: {args.prompt}")
    print("-" * 50)
    
    try:
        # Create a chat completion
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": args.prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        # Extract and print the response
        assistant_response = response.choices[0].message.content
        print(f"Response from model:\n{assistant_response}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()