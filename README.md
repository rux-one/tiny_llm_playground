# Tiny LLM Training Example

```bash=
pip install transformers
pip install datasets
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers[torch]
```

This is a simple educational example of training a small language model. It demonstrates the basic concepts of LLM training using the Hugging Face transformers library.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the training script:
```bash
python tiny_llm.py
```

## What's Happening?

The script does the following:

1. Loads a small pre-trained model (GPT-2 Medium)
2. Prepares a tiny dataset with example conversations
3. Fine-tunes the model on this dataset
4. Saves the trained model
5. Tests the model with some example questions

## Notes

- This is a minimal example for educational purposes
- The dataset is very small and the training is basic
- In real applications, you would need much more data and training time
- The model size is relatively small compared to modern LLMs

## Requirements

See `requirements.txt` for the specific versions of required packages.
