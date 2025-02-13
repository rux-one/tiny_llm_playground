from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import torch
import argparse

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, tokenizer

def generate_response(prompt, model, tokenizer):
    # Format the prompt
    formatted_prompt = f"User: {prompt}\nAssistant:"
    
    # Encode the prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt", return_attention_mask=True)
    
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

def create_interface(model_path):
    print(f"Loading model from {model_path}")
    model, tokenizer = load_model(model_path)
    
    def predict(message):
        return generate_response(message, model, tokenizer)
    
    # Create the Gradio interface
    iface = gr.Interface(
        fn=predict,
        inputs=gr.Textbox(label="Your Question"),
        outputs=gr.Textbox(label="Answer"),
        title="Math Problem Solver",
        description="Ask me any math question!",
        examples=[
            ["What is 15 + 23?"],
            ["Calculate 7 × 8"],
            ["What is 100 - 45?"]
        ]
    )
    return iface

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./math_model',
                      help='Path to the model')
    parser.add_argument('--share', action='store_true',
                      help='Create a public URL')
    args = parser.parse_args()
    
    iface = create_interface(args.model)
    iface.launch(share=args.share)
