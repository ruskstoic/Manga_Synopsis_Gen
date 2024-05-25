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
import Levenshtein
import random

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
def log_user_info(user_name, user_id, formatted_datetime, tab_id, seed_text, gen_text1, gen_text2, num_gen_words, temperature, nucleus_threshold, DBS_diversity_rate, beam_drop_rate, simipen_switch,
                  DBS_switch, DBW_switch, beam_width):
  user_info = {'Name': user_name,
               'User_ID': user_id,
               'Datetime_Entered': formatted_datetime,
               'Tab_ID': tab_id,
               'Seed_Text': seed_text,
               'Gen_Text1': gen_text1,
               'Gen_Text2': gen_text2,
               'Num_Gen_Words': num_gen_words,
               'Temp': temperature,
               'Nucleus_Threshold': nucleus_threshold,
               'DBS_Diversity_Rate': DBS_diversity_rate,
               'Beam_Drop_Rate': beam_drop_rate,
               'Similarity_Penalty': simipen_switch,
               'Diverse_Beam_Search': DBW_switch,
               'Dynamic_Beam_Width': DBW_switch,
               'Beam_Width': beam_width
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
  num_gen_words = int(st.slider('Choose the Number of Generated Words You Would Like', 20, 60))
  temperature = float(st.slider('Choose the Temperature You Would Like (The higher the temperature, the more random the generated words)', 0.3, 1.0))
  nucleus_threshold = float(st.slider('Choose the Nucleus Threshold You Would Like (Higher values allow more randomness by considering a larger set of probable next words)', 0.5, 1.5))
  DBS_diversity_rate = float(st.slider('Choose the Diversity Rate You Would Like (Higher values promote diversity by penalizing similar sequences)', 0.3, 1.0))
  beam_drop_rate = float(st.slider('Choose the Beam Drop Rate You Would Like (Introduces randomness by randomly dropping beams to increase diversity)', 0.0, 0.5))
  simipen_switch = st.selectbox('Select the Similarity Penalty You Would Like to Apply',
                                (None,'jaccard', 'levenshtein'))
  DBS_switch = st.toggle('Activate Diverse Beam Search (Promotes generating varied sequences by penalizing similar beams)')
  DBW_switch = st.toggle('Activate Dynamic Beam Width (Automatically adjusts beam width based on prediction confidence to balance quality and diversity)')
  if DBW_switch:
    beam_width = int(st.slider('Choose the Number of Beams You Would Like (More beams means more possibilities, but also longer generation time)', 3, 8))
  else:
    beam_width = 3
  
  st.write(seed_text, num_gen_words, temperature, nucleus_threshold, DBS_diversity_rate, beam_drop_rate, simipen_switch, DBS_switch, DBW_switch, beam_width)

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
      
      #Download, Save, Load 1st Model (4.400epoch21) from Google Drive
      model1_4o200epoch31_id = '1-R3xMvkZMS7fruMxV-zCCABxiWkmHsP4'
      temp_model1_filepath = '/tmp/model1.keras'
      download_file(file_id=model1_4o200epoch31_id, destination=temp_model1_filepath, filename='Model1')
      model1 = tf.keras.models.load_model(temp_model1_filepath)
      st.success('Model1 file loaded successfully!')
        #Get 1st Tokenizer from Google Drive
      tokenizer1_id = '1V-UihkeuzhrjXl_qUjisgxJqju65JFCP'
      temp_tokenizer1_filepath = '/tmp/tokenizer1.pickle'
      download_file(file_id=tokenizer1_id, destination=temp_tokenizer1_filepath, filename='Tokenizer1')
      with open(temp_tokenizer1_filepath, 'rb') as handle:
        loaded_tokenizer1 = pickle.load(handle)
      st.success('Tokenizer1 file loaded successfully!')

      #Download, Save, Load 2nd Model (7.000epoch2) from Google Drive
      model2_7o000epoch2_id = '1vKnFfCDDB2jyryv4XX7_UFHXcvWZeDQK'
      temp_model2_filepath = '/tmp/model2.keras'
      download_file(file_id=model2_7o000epoch2_id, destination=temp_model2_filepath, filename='Model2')
      model2 = tf.keras.models.load_model(temp_model2_filepath)
      st.success('Model2 file loaded successfully!')
        #Get 2nd Tokenizer from Google Drive
      tokenizer2_id = '1vabdlDt7ObTN2Od-T6xokFbFi1rzVi2b'
      temp_tokenizer2_filepath = '/tmp/tokenizer2.pickle'
      download_file(file_id=tokenizer2_id, destination=temp_tokenizer2_filepath, filename='Tokenizer2')
      with open(temp_tokenizer2_filepath, 'rb') as handle:
        loaded_tokenizer2 = pickle.load(handle)
      st.success('Tokenizer2 file loaded successfully!')
      st.write('Generating text for model1 now...')
  
      #Simple Generator Function
      # def generate_text(model, tokenizer, seq_len, seed_text, num_gen_words, temperature):
      #   output_text = []
      #   input_text = seed_text
      #   prev_pred_word_idx = 0
    
      #   for i in range(num_gen_words-1):
      #     encoded_text = tokenizer.texts_to_sequences([input_text])[0]
      #     pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre', dtype='float32') #expects a list of sequences
      #     pad_encoded = np.array(pad_encoded)
      #     pred_distribution = model.predict(pad_encoded, verbose=0)[0][-1]
      
      #     #Temperature parameter
      #     new_pred_distribution = np.power(pred_distribution, (1/temperature))
      #     new_pred_distribution[prev_pred_word_idx] = 0 #prevents previous word from being next word
      #     new_pred_distribution = new_pred_distribution / new_pred_distribution.sum()

      #     #Choose word with highest probability as next word
      #     choices = range(new_pred_distribution.size)
      #     pred_word_idx = np.random.choice(a=choices, p=new_pred_distribution) #randomly chooses word
      #     # pred_word_idx = np.argmax(new_pred_distribution) #choose max probability word
      #     prev_pred_word_idx = pred_word_idx
      #     pred_word = tokenizer.index_word.get(pred_word_idx, '')
      #     input_text += ' ' + pred_word
      #     output_text.append(pred_word)
      #   return ' '.join(output_text)

      #Generate Synopsis and Show Result
      # filter_size = 10 #changeable parameter
      # gen_text = generate_text(model = model1,
      #                             tokenizer = loaded_tokenizer,
      #                             seq_len = filter_size, # why does -1 work also
      #                             seed_text = seed_text,
      #                             num_gen_words = num_gen_words,
      #                             temperature = temperature)
      # st.success(f'Seed Text: {seed_text}\n\nModel Generation: {gen_text}...')

      #Beam Search 1.4 Generator Function
      def join_tokens(tokens):
        """Join tokens ensuring no space before specified punctuation marks."""
        result = []
        for i, token in enumerate(tokens):
            if token in {'.', ',', '!', '?', "'s", "'t", "'nt"} and i > 0:
              result[-1] += token  # Append punctuation mark to the previous token
            else:
              result.append(token)
        return ' '.join(result)
    
      def calculate_levenshtein_distance(previous_beam, current_beam):
        distance = Levenshtein.distance(previous_beam, current_beam)
        similarity = 1 - (distance / max(len(previous_beam), len(current_beam))) #normalise between 0 and 1 and converts from distance to similarity measure because 0 distance = 1 similarity
        return similarity
    
      def calculate_jaccard_similarity(previous_beam, current_beam):
        intersection = len(set(current_beam).intersection(previous_beam))
        union = len(set(current_beam).union(previous_beam))
        similarity = (intersection / union) if union != 0 else 0
        return similarity
      
      def nucleus_sampling(pred_distribution, threshold):
        sorted_indices = np.argsort(pred_distribution)[::-1]
        cumulative_probs = np.cumsum(pred_distribution[sorted_indices])
        selected_indices = sorted_indices[cumulative_probs <= threshold]
        if len(selected_indices) == 0:
          selected_indices = sorted_indices[:1] #fallback to at least one token
        return selected_indices
    
      def random_beam_dropping(beams, drop_rate=beam_drop_rate):
        return [beam for beam in beams if random.random() > drop_rate]
      
      def v1o4_diverse_beam_search_generation(model,
                                             tokenizer,
                                             seq_len,
                                             seed_text,
                                             beam_width,
                                             num_gen_words,
                                             temperature,
                                             nucleus_threshold,
                                             DBW_probability_threshold,
                                              DBS_num_comparisons,
                                             DBS_diversity_rate,
                                             DBW_switch=False,
                                             simipen_switch=False,
                                              DBS_switch=False,
                                              beam_dropping=True):
      
        output_text = []
        previous_beam = [] ##similarity penalty
      
        #Initialize beam search
        if isinstance(seed_text, str):
          beams = [(seed_text.split(), 1.0)] #a list of tuples with a probability
          seed_text_length = len(beams[0][0])
        elif isinstance(seed_text, list):
          beams = [(seed_text, 1.0)]
          seed_text_length = len(beams[0][0])
      
        moving_average = 0.0
        alpha = 0.8
        cooldown = 3
        cooldown_counter = 2
      
        for _ in range(num_gen_words):
          # print('Beam Width', beam_width)
          new_beams = []
          for beam in beams:
            input_text, beam_prob = beam
            encoded_text = tokenizer.texts_to_sequences([input_text])[0]
            pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre', dtype='float32')
            pad_encoded = np.array(pad_encoded)
            pred_distribution = model.predict(pad_encoded, verbose=0)[0][-1] #Add [-1] for LSTM1.1 only
      
            #Temperature Parameter
            adjusted_distribution = np.power(pred_distribution, (1/temperature))
            adjusted_distribution = adjusted_distribution / adjusted_distribution.sum()
      
            #Use Nucleus Sampling and Similarity Penalty
            selected_indices = nucleus_sampling(adjusted_distribution, threshold=nucleus_threshold)
            for index in selected_indices:
              next_word = tokenizer.index_word.get(index, '')
              if next_word != input_text[-1]: #Ensure no same words are consecutive
                new_text = ' '.join(input_text + [next_word])
                new_prob = beam_prob * adjusted_distribution[index]
                if previous_beam and simipen_switch == 'jaccard':
                  similarity = calculate_jaccard_similarity(previous_beam, beam[0]) ##similarity penalty
                  new_prob = new_prob * (1 - similarity) ##similarity penalty
                elif previous_beam and simipen_switch == 'levenshtein':
                  similarity = calculate_levenshtein_distance(previous_beam, beam[0]) ##similarity penalty
                  new_prob = new_prob * (1 - similarity) ##similarity penalty
                else:
                  new_prob = beam_prob * adjusted_distribution[index]
                new_beams.append((new_text.split(), new_prob))
      
            #Random Beam Dropping, shift tabbed in once
            if beam_dropping:
              new_beams = random_beam_dropping(new_beams)
      
          #Diverse Beam Search
          if DBS_switch:
            diverse_beams = []
            num_beams = len(new_beams)
            for i, beam1 in enumerate(new_beams):
              sampled_indices = np.random.sample(min(num_beams, DBS_num_comparisons))
              for j in range(len(sampled_indices)):
                beam2 = new_beams[j]
                similarity = calculate_jaccard_similarity(beam1[0], beam2[0])
                diversity_penalty = DBS_diversity_rate * similarity
                new_prob = beam1[1] * (1 - diversity_penalty)
                diverse_beams.append((beam2[0], new_prob))
            sorted_beams = sorted(diverse_beams, key=lambda x: x[1], reverse=True)[:beam_width]
          else:
            sorted_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
      
          beams = sorted_beams
          if not beams:
            return None
          previous_beam = beams[0][0]
      
          #Dynamic Beam Width
          if DBW_switch:
            average_prob = np.mean([beam[1] for beam in beams])
            moving_average = alpha * moving_average + (1 - alpha) * average_prob
      
            if cooldown_counter == 0:
              if average_prob > DBW_probability_threshold and beam_width > 0:
                beam_width += 1
              elif average_prob < DBW_probability_threshold and beam_width > 3:
                beam_width -= 1
              cooldown_counter = cooldown
              DBW_probability_threshold *= 0.1
            else:
              cooldown_counter -= 1
        return ' '.join(beams[0][0][seed_text_length:])

      #GF1.4 Generate Text for Model1
      filter_size = 10 #changeable parameter
      model1_generated_text = v1o4_diverse_beam_search_generation(model = model1,
                                       tokenizer = loaded_tokenizer1,
                                       seq_len = filter_size,
                                       seed_text = seed_text,
                                       beam_width = beam_width,
                                       num_gen_words = num_gen_words,
                                       temperature = temperature,
                                       nucleus_threshold = nucleus_threshold,
                                       DBW_probability_threshold = 0.0000004,
                                      DBS_num_comparisons = 10, #10 is good because more means more variation that might not make sense
                                       DBS_diversity_rate = DBS_diversity_rate,
                                       DBW_switch=DBW_switch,
                                       simipen_switch=simipen_switch,
                                      DBS_switch=DBS_switch,
                                        beam_dropping=True)
      st.success(seed_text + model1_generated_text,'...')
      st.write('Generating text for model2 now...')

      #GF1.4 Generate Text for Model1
      model2_generated_text = v1o4_diverse_beam_search_generation(model = model2,
                                       tokenizer = loaded_tokenizer2,
                                       seq_len = filter_size,
                                       seed_text = seed_text,
                                       beam_width = beam_width,
                                       num_gen_words = num_gen_words,
                                       temperature = temperature,
                                       nucleus_threshold = nucleus_threshold,
                                       DBW_probability_threshold = 0.0000004,
                                      DBS_num_comparisons = 10, #10 is good because more means more variation that might not make sense
                                       DBS_diversity_rate = DBS_diversity_rate,
                                       DBW_switch=DBW_switch,
                                       simipen_switch=simipen_switch,
                                      DBS_switch=DBS_switch,
                                        beam_dropping=True)
      st.success(seed_text + model2_generated_text,'...')
  
      ## Save Data to Google Sheet
      log_entry_df = log_user_info(user_name=user_name, user_id=user_id, formatted_datetime=formatted_datetime, tab_id=tab_id, seed_text=seed_text, gen_text1=model1_generated_text, gen_text2=model2_generated_text,
                                   num_gen_words=num_gen_words,temperature=temperature, nucleus_threshold=nucleus_threshold, DBS_diversity_rate=DBS_diversity_rate, beam_drop_rate=beam_drop_rate, simipen_switch=simipen_switch,
                                   DBS_switch=DBS_switch, DBW_switch=DBW_switch, beam_width=beam_width)
      conn = st.connection('gsheets', type=GSheetsConnection)
      existing_data = conn.read(worksheet='Sheet2', usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17], end='A')
      existing_df = pd.DataFrame(existing_data, columns=['Name', 'User_ID', 'Datetime_Entered', 'Tab_ID', 'Seed_Text', 'Gen_Text1', 'Gen_Text2','Num_Gen_Words', 'Temp', 'Nucleus_Threshold','DBS_Diversity_Rate',
                                                         'DBS_Diversity_Rate','Beam_Drop_Rate','Similarity_Penalty','Diverse_Beam_Search','Dynamic_Beam_Width','Beam_Width'])
      combined_df = pd.concat([existing_df, log_entry_df], ignore_index=True)
      conn.update(worksheet='Sheet2', data=combined_df)
      st.cache_data.clear()
      
## Streamlit Tracker End
streamlit_analytics.stop_tracking(unsafe_password)
