## Import
import os
import streamlit as st
import pandas as pd
import numpy as np
import uuid
import base64
from streamlit_cookies_manager import EncryptedCookieManager
import tensorflow as tf
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
def get_or_create_tab_ID():
  if 'tab_id' not in st.session_state:
    st.session_state.tab_id = str(uuid.uuid4())
  return st.session_state.tab_id

hi = get_or_create_tab_ID()
st.write(hi)
st.write(st.session_state)
st.session_state

## Streamlit Interface
st.title('Shall We Generate A Never-Before-Seen Manga Synopsis?')

## Streamlit Tracker End
streamlit_analytics.stop_tracking(unsafe_password)
