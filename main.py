# %% Import libararies
import os
import numpy as np
import pandas as pd
import pytrends
import yfinance as yf
import requests
import nest_asyncio
import asyncpg
from dotenv import load_dotenv
from pytrends.request import TrendReq
load_dotenv('.env')
nest_asyncio.apply()



#%% GET DATA FROM YFINANCE
sp_wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
sp_wiki_df_list = pd.read_html(sp_wiki_url)
sp_df = sp_wiki_df_list[0]
sp_ticker_list = list(sp_df['Symbol'].values)
sp_name_list = list(sp_df['Security'].values)

#%% DOWNLOAD FROM YFINANCE
df_sp_values = yf.download(sp_ticker_list, start="2016-01-01")

#%% PYTRENDS DATA
pytrends = TrendReq(hl='en-us')
dataset = []
all_keywords = sp_ticker_list
for keyword in all_keywords:
    pytrends.build_payload(kw_list, cat=0, timeframe='today 5-y', geo='', gprop='')
    data = pytrends.interest_over_time()
    if not data.empty:
        data = data.drop(labels='isPartial', axis='columns')
        dataset.append(data)
dataset
#%%
adj_close = df_sp_values['Adj Close']
adj_close.columns
len(adj_close['AAPL'])



# %% 
##ASSUMPTIONS
#Intraday losses greater than 20% are unlikely to be a market failure
# and are more probably a single stock failure
# 
def findLocationsOfValueDrop(stock):
    #function tolerances of 5% up on lowest value to be considered part of the same
    #decrease;
    #minimum drop rate of 5% over 15days
    #find start bounds and end bounds
    print('hello')


#function that compares the current search average to the averages
#of the periods say 2-6 months before and after the stock drops
#as a normalized value

#or i could compute the running average or search activity over the last 3-6months 
#and compare that to the current value, and then add the normalised value to 
#the average and compute the next value


# %% KEYS
API_KEY = os.getenv('API_KEY')
CHANNEL_ID = "UCTckA2i1O6aiqdnsYm7jhnQ"

#make API call
pageToken = ''
url = 'https://www.googleapis.com/youtube/v3/search?key='+API_KEY+"&channelId="+CHANNEL_ID+"&part=snippet,id&order=date&maxResults=10000"+pageToken
response = requests.get(url).json()
response['items']

# %%