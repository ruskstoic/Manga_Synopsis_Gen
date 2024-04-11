## Import
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import requests
from streamlit_gsheets import GSheetsConnection
import streamlit_analytics
from streamlit.runtime import get_instance
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
from streamlit_cookies_manager import EncryptedCookieManager
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.http import MediaIoBaseDownload
from google.auth.credentials import Credentials
from google.oauth2 import service_account

## Streamlit Tracker Start
streamlit_analytics.start_tracking()

## Streamlit Interface
st.title('Shall We Generate A Never-Before-Seen Manga Synopsis?')



## Streamlit Tracker End
unsafe_password = st.secrets['ST_ANALYTICS_PW']
streamlit_analytics.stop_tracking(unsafe_password)
