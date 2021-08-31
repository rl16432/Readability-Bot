import os
import json
import torch
from transformers import RobertaTokenizerFast, AutoModelForSequenceClassification

MAX_LENGTH = 256

def init():
    global tokenizer, model

    # Initialise model directories
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    model_folder = "Commonlit-RoBERTa-Base"

    # Initialise RoBERTa tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained(os.path.join(model_dir, model_folder, "tokenizer"))

    # Initialise pretrained RoBERTa-Base model
    model = AutoModelForSequenceClassification.from_pretrained(os.path.join(model_dir, model_folder))

def run(request):
    global tokenizer, model

    data = json.loads(request)

    # Load data
    inputs = list(data["data"])

    # Replace new lines with spaces
    inputs = [text.replace('\n', ' ') for text in inputs]

    # Tokenize data using RoBERTa tokenizer
    token_seqs = tokenizer(inputs, padding = "max_length", 
                            max_length = MAX_LENGTH, 
                            truncation = True, 
                            return_tensors = "pt")
    
    # Pass through pre-trained model
    outputs_tensor = model(**token_seqs)

    # Convert model outputs to list
    outputs = outputs_tensor.logits.flatten().tolist()
    
    # Display targets
    print("Results: " + str(outputs))

    return outputs

