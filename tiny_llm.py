from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import torch
import os
import argparse

def prepare_data(model_path="./tiny_llm"):
    # Derive training data filename from model path
    data_file = f"{model_path}_data"
    
    # Load conversations from training data file
    try:
        with open(data_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Training data file '{data_file}' not found. Please create it first.")
    
    # Combine User and Assistant lines into conversations
    conversations = []
    for i in range(0, len(lines), 2):
        if i + 1 < len(lines):  # Make sure we have both User and Assistant lines
            user_line = lines[i].strip()
            assistant_line = lines[i + 1].strip()
            conversation = f"{user_line}\n{assistant_line}"
            conversations.append(conversation)
    
    # Create a simple dataset
    dataset = Dataset.from_dict({
        "text": conversations
    })
    return dataset

def train_model(model_path="./tiny_llm"):
    # Print CUDA information
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
        print("CUDA device count:", torch.cuda.device_count())
    
    # Initialize a small model (125M parameters)
    model_name = "facebook/opt-125m"  # A very small but capable model

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # Prepare the dataset
    dataset = prepare_data(model_path)
    
    def tokenize_function(examples):
        outputs = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Set up training arguments with stronger learning parameters
    training_args = TrainingArguments(
        output_dir=model_path,
        num_train_epochs=20,  # Increased from 3 to 10
        per_device_train_batch_size=2,  # Reduced batch size for better learning
        learning_rate=5e-4,  # Increased learning rate
        save_steps=100,
        save_total_limit=2,
        logging_steps=10,
        weight_decay=0.01,  # Added weight decay for regularization
        warmup_steps=100,  # Added warmup steps
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)

def generate_response(prompt, model_path="./tiny_llm"):
    # Load the trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    print("Using: {}".format("GPU" if torch.cuda.is_available() else "CPU"))

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Encode the prompt
    inputs = tokenizer(f"User: {prompt}\nAssistant:", return_tensors="pt", return_attention_mask=True)
    
    # Move inputs to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate response
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=100,
        num_return_sequences=1,
        temperature=0.1,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Assistant:")[-1].strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model before testing')
    parser.add_argument('--model', type=str, default='./tiny_llm', help='Path to save/load the model')
    parser.add_argument('--question', type=str, help='Question to ask the model')
    args = parser.parse_args()

    if args.train:
        print(f"Starting training using data from '{args.model}_data'...")
        train_model(args.model)
    
    if args.question:
        response = generate_response(args.question, args.model)
        print(f"\nQ: {args.question}")
        print(f"A: {response}")
    else:
        test_questions = [
            "What is 15 + 23?",
            "What is 72 ÷ 8?",
            "Calculate 45 + 67",
            "Solve 13 × 12"
        ]
        
        for question in test_questions:
            response = generate_response(question, args.model)
            print(f"\nQ: {question}")
            print(f"A: {response}")
