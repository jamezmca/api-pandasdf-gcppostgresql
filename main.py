# %% Import libararies
import os
import numpy as np
import pandas as pd
import requests
import nest_asyncio
import asyncpg
from dotenv import load_dotenv
load_dotenv('.env')
nest_asyncio.apply()

# %% KEYS
API_KEY = os.getenv('API_KEY')
CHANNEL_ID = "UCTckA2i1O6aiqdnsYm7jhnQ"

#make API call
pageToken = ''
url = 'https://www.googleapis.com/youtube/v3/search?key='+API_KEY+"&channelId="+CHANNEL_ID+"&part=snippet,id&order=date&maxResults=10000"+pageToken
response = requests.get(url).json()
response['items']
# %%