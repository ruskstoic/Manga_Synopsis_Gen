## Import
import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import uuid
import base64
from streamlit_cookies_manager import EncryptedCookieManager
import requests
from streamlit_gsheets import GSheetsConnection
import streamlit_analytics
from streamlit.runtime import get_instance
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.http import MediaIoBaseDownload
from google.auth.credentials import Credentials
from google.oauth2 import service_account
from datetime import datetime, timedelta
import pytz
import tensorflow as tf
import pickle
import io
import tempfile
import requests
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# st.write(tf.__version__)

## Streamlit Tracker Start
streamlit_analytics.start_tracking()
unsafe_password = st.secrets['ST_ANALYTICS_PW'] 

## Cookies Manager
cookies = EncryptedCookieManager(prefix='manga-synopsis-gen/',
                                 password=unsafe_password)

if not cookies.ready():
  st.stop()
cookies_user_id = cookies.get('user_id')
if cookies_user_id is None:
  cookies_user_id = str(uuid.uuid4())
  cookies['user_id'] = cookies_user_id

## Functions
#Function to create tab ID
def get_or_create_tab_ID():
  if 'tab_id' not in st.session_state:
    st.session_state.tab_id = str(uuid.uuid4())
  return st.session_state.tab_id

#Function to compile user info
def log_user_info(user_name, user_id, formatted_datetime, tab_id, seed_text, gen_text, num_gen_words, temperature,):
  user_info = {'Name': user_name,
               'User_ID': user_id,
               'Datetime_Entered': formatted_datetime,
               'Tab_ID': tab_id,
               'Seed_Text': seed_text,
               'Gen_Text': gen_text,
               'Num_Gen_Words': num_gen_words,
               'Temp': temperature,
              }
  df_log_entry = pd.DataFrame([user_info])
  return df_log_entry

## Google Drive Connection
SCOPES = ['https://www.googleapis.com/auth/drive']

def authenticate():
  gdrive_auth_secret = st.secrets['GDRIVE_AUTHENTICATION_CREDENTIALS']
  creds = service_account.Credentials.from_service_account_info(gdrive_auth_secret, scopes=SCOPES)
  return creds
credentials = authenticate()
service = build('drive', 'v3', credentials=credentials)

## Streamlit Interface
st.title('Shall We Generate A Never-Before-Seen Manga Synopsis?')
user_name = st.text_input('Hello! What is your name?')

## Start User Pipeline
if user_name:
  st.write(f'Hello {user_name}!')

  # Create User ID and Tab ID
  user_id = cookies['user_id']
  tab_id = get_or_create_tab_ID()

  # Create Datetime Entered
  datetime_format = '%Y-%m-%d %H:%M:%S'
  converted_timezone = pytz.timezone('Asia/Singapore')
  datetime_entered = datetime.now(converted_timezone)
  formatted_datetime = datetime_entered.strftime(datetime_format)

  # User Input Seed Text, Temperature, Num_Gen_Words
  seed_text = str(st.text_area('Input some text here and we will generate a synopsis from this!\n\n'))
  temperature = float(st.slider('Choose the Temperature You Would Like (The higher the temperature, the more random the generated words)', 0.3, 1.0))
  num_gen_words = int(st.slider('Choose the Number of Generated Words You Would Like', 20, 60))

  st.write(seed_text, temperature, num_gen_words)

  ## User has input seed text and click generate button
  if seed_text:
    if st.button('Generate'):
      st.write('Generating...')
      
      def download_file(file_id, destination, filename):
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
          status, done = downloader.next_chunk()
          st.write(f"{filename} Downloaded {int(status.progress() * 100)}%.")
        fh.seek(0)
        # if destination[0:4:-1] == 'keras':
        with open(destination, 'wb') as f:
          f.write(fh.read())
        # if destination[0:5:-1] == 'pickle':
          
      
      #Download, Save, Load 1st Model from Google Drive
      model1_4o200epoch31_id = '1-R3xMvkZMS7fruMxV-zCCABxiWkmHsP4'
      temp_model1_filepath = '/tmp/model1.keras'
      download_file(file_id=model1_4o200epoch31_id, destination=temp_model1_filepath, filename='Model1')
      model1 = tf.keras.models.load_model(temp_model1_filepath)
      st.success('Model1 file loaded successfully!')

      #Get Tokenizer from Google Drive
      tokenizer_id = '1V-UihkeuzhrjXl_qUjisgxJqju65JFCP'
      temp_tokenizer_filepath = '/tmp/tokenizer.pickle'
      download_file(file_id=tokenizer_id, destination=temp_tokenizer_filepath, filename='Tokenizer1')
      with open(temp_tokenizer_filepath, 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)
      st.success('Tokenizer1 file loaded successfully!')
  
      #Simple Generator Function
      def generate_text(model, tokenizer, seq_len, seed_text, num_gen_words, temperature):
        output_text = []
        input_text = seed_text
        prev_pred_word_idx = 0
    
        for i in range(num_gen_words-1):
          encoded_text = tokenizer.texts_to_sequences([input_text])[0]
          pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre', dtype='float32') #expects a list of sequences
          pad_encoded = np.array(pad_encoded)
          pred_distribution = model.predict(pad_encoded, verbose=0)[0][-1]

          assert pred_distribution.ndim == 1, f"Expected 1D array, got {pred_distribution.ndim}D array"
      
          #Temperature parameter
          new_pred_distribution = np.power(pred_distribution, (1/temperature))
          new_pred_distribution[prev_pred_word_idx] = 0 #prevents previous word from being next word
          new_pred_distribution = new_pred_distribution / new_pred_distribution.sum()

          assert pred_distribution.ndim == 1, f"Expected 1D array, got {pred_distribution.ndim}D array"
      
          #Choose word with highest probability as next word
          choices = range(new_pred_distribution.size)
          pred_word_idx = np.random.choice(a=choices, p=new_pred_distribution) #randomly chooses word
          # pred_word_idx = np.argmax(new_pred_distribution) #choose max probability word
          prev_pred_word_idx = pred_word_idx
          pred_word = tokenizer.index_word.get(pred_word_idx, '')
          input_text += ' ' + pred_word
          output_text.append(pred_word)
      
          #Create Index-Word-Probability Dataframe for Data Visualisation
          # if i==num_gen_words - 1:
          #   chosen = pd.DataFrame({'Probability': new_pred_distribution})
          #   chosen['Word'] = chosen.index.map(lambda x: tokenizer.index_word.get(x, ''))
          #   chosen = chosen[['Word', 'Probability']]
          #   chosen_top3 = chosen.nlargest(3, 'Probability')
          #   chosen_bottom3 = chosen.nsmallest(3, 'Probability')
          #   st.write('Top 3 Probabilities:', chosen_top3) ###comment out print for pipeline
          #   st.write('Bottom 3 Probabilities:', chosen_bottom3) ###comment out print for pipeline
        return ' '.join(output_text)
  
      #Generate Synopsis and Show Result
      filter_size = 10 #changeable parameter
      gen_text = generate_text(model = model1,
                                  tokenizer = loaded_tokenizer,
                                  seq_len = filter_size, # why does -1 work also
                                  seed_text = seed_text,
                                  num_gen_words = num_gen_words,
                                  temperature = temperature)
      st.success(f'Seed Text: {seed_text}\n\nModel Generation: {gen_text}...')
  
      ## Save Data to Google Sheet
      log_entry_df = log_user_info(user_name=user_name, user_id=user_id, formatted_datetime=formatted_datetime, tab_id=tab_id, seed_text=seed_text, gen_text=gen_text, num_gen_words=num_gen_words,
                                  temperature=temperature)
      st.write(log_entry_df)
      conn = st.connection('gsheets', type=GSheetsConnection)
      existing_data = conn.read(worksheet='Sheet2', usecols=[0,1,2,3,4,5,6,7], end='A')
      existing_df = pd.DataFrame(existing_data, columns=['Name', 'User_ID', 'Datetime_Entered', 'Tab_ID', 'Seed_Text', 'Gen_Text', 'Num_Gen_Words', 'Temp'])
      combined_df = pd.concat([existing_df, log_entry_df], ignore_index=True)
      conn.update(worksheet='Sheet2', data=combined_df)
      st.cache_data.clear()
      st.write('hello there')

## Streamlit Tracker End
streamlit_analytics.stop_tracking(unsafe_password)
