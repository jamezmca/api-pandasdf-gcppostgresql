# %% Import libararies
import os
import numpy as np
import pandas as pd
import pytrends
import yfinance as yf
import requests
import asyncio
import nest_asyncio
import asyncpg
from dotenv import load_dotenv
from pytrends.request import TrendReq
load_dotenv('.env')
nest_asyncio.apply()

#EXAMPLE make API call
# pageToken = ''
# url = 'https://www.googleapis.com/youtube/v3/search?key='+API_KEY+"&channelId="+CHANNEL_ID+"&part=snippet,id&order=date&maxResults=10000"+pageToken
# response = requests.get(url).json()

#%% GET DATA FROM YFINANCE
def clenseArray(array):
    array = [x.lower().replace(" ", "_")\
        .replace("-","_").replace("?","_").replace(r"/", "_")\
        .replace(")", "").replace(r"(", "").replace("%", "")\
        .replace("?", "").replace("\\", "_").replace("$","").replace('&',"and") for x in array]
    return array

sp_wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
sp_wiki_df_list = pd.read_html(sp_wiki_url)
sp_df = sp_wiki_df_list[0]
sp_ticker_list = list(sp_df['Symbol'].values)
sp_name_list = list(sp_df['Security'].values)
dictionary = dict(zip(clenseArray(sp_name_list), clenseArray(sp_ticker_list)))


#%% DOWNLOAD FROM YFINANCE INTO DATAFRAME
df_sp_values = yf.download(sp_ticker_list, start="2016-01-01")

#%% TAKE ADJ CLOSE VALUES AND TURN INTO DF
df_sp_prices = df_sp_values['Adj Close']
df_sp_prices.columns = [x.lower().replace(" ", "_")\
        .replace("-","_").replace("?","_").replace(r"/", "_")\
        .replace(")", "").replace(r"(", "").replace("%", "")\
        .replace("?", "").replace("\\", "_").replace("$","").replace('&',"and") for x in df_sp_prices.columns]

#%% PYTRENDS DATA
pytrends = TrendReq(hl='en-us')
dataset = []
all_keywords = sp_name_list
# pytrends.build_payload(['Apple'], cat=0, timeframe='today 5-y', geo='', gprop='')
# data = pytrends.interest_over_time()
# data

for keyword in all_keywords:
    pytrends.build_payload([keyword], cat=0, timeframe='today 5-y', geo='', gprop='')
    data = pytrends.interest_over_time()
    if not data.empty:
        data = data.drop(labels='isPartial', axis='columns')
        dataset.append(data)

#%% TRANSFORM DATASET INTO COMPLETE DATAFRAME
df_sp_searches = dataset[0]
for i in range(len(dataset)):
    if i == 0:
        continue
    df_sp_searches = pd.concat([df_sp_searches, dataset[i]], axis=1)
df_sp_searches.columns = [x.lower().replace(" ", "_")\
        .replace("-","_").replace("?","_").replace(r"/", "_").replace('.', '')\
        .replace(")", "").replace(r"(", "").replace("%", "")\
        .replace("?", "").replace("\\", "_").replace("$","").replace('&',"and") for x in df_sp_searches.columns]

#%% A GOOD CHECK TO MAKE SURE DATAFRAMES ALIGN
# df_sp_searches['abbott_laboratories']
df_sp_searches.axes[0][0]
print(f'{df_sp_searches.axes[0][0]}'.split(' ')[0])

df_sp_prices.loc['2019-08-04']
# df_sp_prices.loc[f'{df_sp_searches.axes[0][10]}'.split(' ')[0]]

#%%CREATE TABLE SCHEMA
col_str = 'date DATE, '
for stock_label in df_sp_prices.columns:
    col_str = col_str + f'{stock_label} ' + 'FLOAT, ' 

col_str_searches = 'date DATE, '
for stock_label in df_sp_searches.columns:
    col_str_searches = col_str_searches + f'{stock_label} ' + 'FLOAT, '
col_str


#%% UPLOAD DATA TO POSTGRESQL DATABASE IN GOOGLE CLOUD
#USER AUTH FOR GOOGLE CLOUD DATABASE FROM ENVIRONMENT VARIABLES
user = os.getenv("USERNAME")
password = os.getenv("PASSWORD")
database = os.getenv("DATABASE")
ip = os.getenv("PUBLICIP")

def makeListOfTuples(dataframe):
    listOfTuples = []
    for i in range(len(dataframe)):
        listOfTuples.append(dataframe.loc(i))

async def run():
    conn = await asyncpg.connect(user=user, password=password, database=database, host=ip)
    print('connected')
    await conn.execute(f'DROP TABLE IF EXISTS sp_prices')
    await conn.execute(f'DROP TABLE IF EXISTS sp_searches')
    await conn.execute(f'''
            CREATE TABLE sp_prices (
                {col_str}
            );
        ''')
    print('sp_prices was created successfully')
    await conn.execute(f'''
            CREATE TABLE sp_searches (
                {col_str_searches}
            );
        ''')
    print('sp_searches was created successfully')

    await conn.close() #close the connection
loop = asyncio.get_event_loop() #can also make single line
loop.run_until_complete(run())
print('all tables successfully imported')


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



# %%