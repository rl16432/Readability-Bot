import json
import tensorflow as tf
import os 
import numpy as np
from azureml.core import Workspace, Dataset

def init():
  global model, tokenizer, max_length

  # Initialise model path
  model_root = os.getenv('AZUREML_MODEL_DIR')
  tf_model_folder = 'Commonlit-Bi-LSTM'

  # Load maximum length from max_length.txt
  with open(os.path.join(model_root, tf_model_folder, 'assets/max_length.txt')) as text:
    max_length = int(text.read())
  
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