# %% Import libararies
from datetime import datetime
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
from datetime import datetime
from pytrends.request import TrendReq
load_dotenv('.env')
nest_asyncio.apply()

#EXAMPLE make API call
# pageToken = ''
# url = 'https://www.googleapis.com/youtube/v3/search?key='+API_KEY+"&channelId="+CHANNEL_ID+"&part=snippet,id&order=date&maxResults=10000"+pageToken
# response = requests.get(url).json()

#%% FETCH DATA FROM YFINANCE & PYTRENDS AND CLEAN OR CSV
# #  -> UPLOAD TO POSTGRESQL
def clenseArray(array):
    array = [x.lower().replace(" ", "_")\
        .replace("-","_").replace("?","_").replace(r"/", "_").replace('.', '')\
        .replace(")", "").replace(r"(", "").replace("%", "").replace('all', 'all_')\
        .replace("?", "").replace("\\", "_").replace("$","").replace('&',"and") for x in array]
    return array

csv_files = []
for file in os.listdir(os.getcwd()):
    if file.endswith('.csv'):
        csv_files.append(file)

if len(csv_files) == 0:
    #DOWNLOAD FROM YFINANCE INTO DATAFRAME
    sp_wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp_wiki_df_list = pd.read_html(sp_wiki_url)
    sp_df = sp_wiki_df_list[0]
    sp_ticker_list = list(sp_df['Symbol'].values)
    sp_name_list = list(sp_df['Security'].values)
    dictionary = dict(zip(clenseArray(sp_name_list), clenseArray(sp_ticker_list)))

    df_sp_values = yf.download(sp_ticker_list, start="2016-01-01")

    #TAKE ADJ CLOSE VALUES AND TURN INTO DF
    df_sp_prices = df_sp_values['Adj Close']
    df_sp_prices.columns = clenseArray(df_sp_prices.columns)
    
    #DOWNLOAD FROM PYTRENDS INTO DF
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

    #TRANSFORM DATASET INTO COMPLETE DATAFRAME
    df_sp_searches = dataset[0]
    for i in range(len(dataset)):
        if i == 0:
            continue
        df_sp_searches = pd.concat([df_sp_searches, dataset[i]], axis=1)
    df_sp_searches.columns = clenseArray(df_sp_searches.columns)

    #SAVE TO CSV FILE
    df_sp_searches.to_csv('df_sp_searches.csv', header=df_sp_searches.columns, index=True , encoding='utf-8')
    df_sp_prices.to_csv('df_sp_prices.csv', header=df_sp_prices.columns, index=True , encoding='utf-8')
else: 
    df_sp_prices = pd.read_csv('df_sp_prices.csv')
    
#CREATE TABLE SCHEMA
col_str = 'date DATE, '
for stock_label in df_sp_prices.columns:
    col_str = col_str + f'{stock_label} ' + 'FLOAT, ' 
col_str = col_str[:-2]

#UPLOAD DATA TO POSTGRESQL DATABASE IN GOOGLE CLOUD
#USER AUTH FOR GOOGLE CLOUD DATABASE FROM ENVIRONMENT VARIABLES
user = os.getenv("USERNAME")
password = os.getenv("PASSWORD")
database = os.getenv("DATABASE")
ip = os.getenv("PUBLICIP")

def typeClean(str):
    str_arr = str.strip().split(',')
    
    for i in range(len(str_arr)):
        if i == 0:
            str_arr[i] = datetime.fromisoformat(str_arr[i])
        elif str_arr[i] == '':
            str_arr[i] = None
        else:
            str_arr[i] = float(str_arr[i])
    return str_arr


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
                {col_str}
            );
        ''')
    print('sp_searches was created successfully')
    # copy prices to table using price header
    values = []
    with open('df_sp_prices.csv', 'r') as f:
        next(f)
        for row in f:
            values.append(tuple(typeClean(row)))
        
    result = await conn.copy_records_to_table(
        'sp_prices', records=values
    )
    print(result, 'import to sp_prices complete')

    # copy searches to table using insert if header
    # equals dictionary translation
    valuesSecond = []
    with open('df_sp_searches.csv', 'r') as f:
        next(f)

    await conn.close() #close the connection
loop = asyncio.get_event_loop() #can also make single line
loop.run_until_complete(run())
print('all tables successfully imported')

#%%
df_sp_prices = pd.read_csv('df_sp_prices.csv')
    
#CREATE TABLE SCHEMA
col_str = 'date DATE, '
for stock_label in df_sp_prices.columns:
    col_str = col_str + f'{stock_label} ' + 'FLOAT, ' 
col_str = col_str[:-2]