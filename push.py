from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import argparse
import os

def push_to_hub(model_path, repo_name, token=None):
    """
    Push a trained model to Hugging Face Hub.
    
    Args:
        model_path: Local path to the trained model
        repo_name: Name for the model on HF Hub (format: username/model-name)
        token: HuggingFace access token (optional, will use HF_TOKEN from .env if not provided)
    """
    # Load token from .env if not provided
    if token is None:
        load_dotenv()
        token = os.getenv('HF_TOKEN')
        if token is None:
            raise ValueError("No token provided and HF_TOKEN not found in .env file")
    
    # Set the token
    os.environ['HUGGINGFACE_TOKEN'] = token
    
    print(f"Loading model and tokenizer from {model_path}")
    
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Configure the model for text generation
    model.config.task_specific_params = {
        "text-generation": {
            "do_sample": True,
            "max_length": 100,
            "temperature": 0.1
        }
    }
    model.config.architectures = ["OPTForCausalLM"]
    model.config.task = "text-generation"
    
    print(f"Pushing to Hugging Face Hub as {repo_name}")
    
    # Push to hub
    model.push_to_hub(repo_name, use_auth_token=token)
    tokenizer.push_to_hub(repo_name, use_auth_token=token)
    
    print("Done! Your model is now available on Hugging Face Hub")
    print(f"View it at: https://huggingface.co/{repo_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Push a trained model to Hugging Face Hub')
    parser.add_argument('--model', type=str, required=True,
                      help='Local path to the trained model (e.g., ./math_model)')
    parser.add_argument('--repo', type=str, required=True,
                      help='Repository name on HF Hub (format: username/model-name)')
    parser.add_argument('--token', type=str,
                      help='Hugging Face access token (optional, will use HF_TOKEN from .env if not provided)')
    
    args = parser.parse_args()
    
    push_to_hub(args.model, args.repo, args.token)
