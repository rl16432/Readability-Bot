import json
import tensorflow as tf
import random
import os 
import numpy as np
import pandas as pd
from azureml.core import Workspace, Dataset
from sklearn.model_selection import train_test_split

def seed_everything():
  tf.random.set_seed(3)
  np.random.seed(3)
  random.seed(3)
  os.environ['PYTHONHASHSEED'] = '3'

def init():
  global model, tokenizer, max_length
  
  # Seed everything
  seed_everything()

  model_root = os.getenv('AZUREML_MODEL_DIR')
  tf_model_folder = 'model1'

  dataset_train = pd.read_csv(os.path.join(model_root, tf_model_folder, "assets/train.csv"))
  dataset_test = pd.read_csv(os.path.join(model_root, tf_model_folder, "assets/test.csv"))
  
  dataset_train, dataset_eval = train_test_split(dataset_train, test_size = 0.2, random_state = 42)
  
  train_text = dataset_train.loc[:,'excerpt'].values
  eval_text = dataset_eval.loc[:,'excerpt'].values
  pred_text = dataset_test.loc[:,'excerpt'].values

  # Establish tokenizer to process text
  tokenizer = tf.keras.preprocessing.text.Tokenizer()
  
  total_text = np.concatenate((train_text, eval_text, pred_text))
  All_text = np.array([tf.keras.preprocessing.text.text_to_word_sequence(s) for s in total_text], dtype = "object")

  # Fit tokenizer on all text
  tokenizer.fit_on_texts(All_text)

  max_length = max([len(s) for s in All_text])

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