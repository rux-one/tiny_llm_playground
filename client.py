from gradio_client import Client
import argparse

def query_model(server_url, prompt):
    """
    Query the model through Gradio client
    
    Args:
        server_url: URL of the Gradio server
        prompt: The prompt to send to the model
    """
    client = Client(server_url)
    result = client.predict(prompt, api_name="/predict")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Query your model through Gradio client')
    parser.add_argument('--url', type=str, required=True,
                      help='URL of the Gradio server')
    parser.add_argument('--prompt', type=str, required=True,
                      help='Prompt to send to the model')
    
    args = parser.parse_args()
    
    try:
        response = query_model(args.url, args.prompt)
        print(f"\nPrompt: {args.prompt}")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {str(e)}")
