from functools import total_ordering
import json
import tensorflow as tf
import random
import os 
import numpy as np
import pandas as pd
from azureml.core import Workspace, Dataset

def init():
  global model, tokenizer, max_length

  # Initialise model path
  model_root = os.getenv('AZUREML_MODEL_DIR')
  tf_model_folder = 'Commonlit-Bi-LSTM'

  # Read training and test sets to calculate maximum length or the amount of padding required
  dataset_train = pd.read_csv(os.path.join(model_root, tf_model_folder, "assets/train.csv"))
  dataset_test = pd.read_csv(os.path.join(model_root, tf_model_folder, "assets/test.csv"))
  
  # Extract excerpts from each dataset into Numpy array from
  train_text = dataset_train.loc[:,'excerpt'].values
  pred_text = dataset_test.loc[:,'excerpt'].values
  
  # Combine text into one array and create word sequences for each text sample
  total_text = np.concatenate(train_text, pred_text)
  total_text_sequences = np.array([tf.keras.preprocessing.text.text_to_word_sequence(text) for text in total_text], dtype = "object")
  
  # Calculate maximum length
  max_length = max([len(sequence) for sequence in total_text_sequences])
  
  # Load tokenizer from the saved tokenizer
  with open(os.path.join(model_root, tf_model_folder, 'assets/tokenizer.json')) as f:
    data = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)
  
  # Load LSTM model
  model = tf.keras.models.load_model(os.path.join(model_root, tf_model_folder))

def run(request):
  global model, tokenizer, max_length

  data = json.loads(request)
 
  # Load data
  inputs = np.array(data["data"], dtype = "object")
  
  # Tokenize data
  tokens = tokenizer.texts_to_sequences(inputs)
  pad_tokens = tf.keras.preprocessing.sequence.pad_sequences(tokens, maxlen = max_length, padding = "pre")

  # Prediction in np_array
  preds_array = model.predict(x = pad_tokens)

  # Prediction in list form
  pred_list = preds_array.flatten().tolist()
  
  print("Result: " + str(pred_list))
  
  return pred_list