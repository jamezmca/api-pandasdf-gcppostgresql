# %% Import libararies
from datetime import datetime
import os
import numpy as np
import pandas as pd
from pandas.core.algorithms import isin
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
    dictionary = dict(zip(clenseArray(sp_ticker_list), clenseArray(sp_name_list)))

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

def dolStrTwo(column): 
    string = ''
    col = column[1:]
    print(col)
    for val in col:
        string = string + f"('{val}'), "
    print(string.strip()[:-1])
    return string.strip()[:-1]

ivd = {v: k for k, v in dictionary.items()}

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
    df_sp_searches = pd.read_csv('df_sp_searches.csv')
    for col in df_sp_searches.columns:
        print(col)
        await conn.execute(f'''
            INSERT INTO sp_searches ({ivd[col] if col != 'date' else 'date'})
            VALUES {dolStrTwo(df_sp_searches[col].values)};
        ''')

    await conn.close() #close the connection
loop = asyncio.get_event_loop() #can also make single line
loop.run_until_complete(run())
print('all tables successfully imported')

#%%

def dolStr(length): 
    string = ''
    for i in range(length):
        if i != length - 1: 
            string = string + f'(${i+1}) '
        else:
            string = string + f'(${i+1})'
    return string

def dolStrTwo(column): 
    string = ''
    col = column[1:]
    print(col)
    for val in col:
        string = string + f"('{val}'), "
    print(string.strip()[:-1])
    return string.strip()[:-1]

def colToTupleList(column):
    col = column[1:]
    colList = []
    if isinstance(column[1], (int, np.integer)):
        for val in col:
            print(val)
            colList.append((float(val)))
    else:
        for val in col:
            colList.append(f'{val}')
    return colList
ivd = {v: k for k, v in dictionary.items()}

async def run():
    conn = await asyncpg.connect(user=user, password=password, database=database, host=ip)
    print('connected')

    valuesSecond = []
    df_sp_searches = pd.read_csv('df_sp_searches.csv')
    for col in df_sp_searches.columns:
        print(col)
        await conn.execute(f'''
            INSERT INTO sp_searches ({ivd[col] if col != 'date' else 'date'})
            VALUES {dolStrTwo(df_sp_searches[col].values)};
        ''')

        
        #create ($n) string for length of column and insert array of tuples for each item 
        # await conn.executemany(f'''
        #     INSERT INTO sp_searches ({dictionary[col] if col != 'date' else 'date'}) VALUES {dolStr(len(df_sp_searches[col]))};
        #     ''', colToTupleList(df_sp_searches[col].values))

    await conn.close() 

asyncio.get_event_loop().run_until_complete(run())

#%%

      

df_sp_searches = pd.read_csv('df_sp_searches.csv')
df_sp_searches['3m']
dictionary['mmm']
# cols = df_sp_searches.columns[0]
# print(df_sp_searches[cols].values[0])
# print(isinstance(df_sp_searches[cols].values[1], (int, np.integer)))
# if '-' in df_sp_searches[cols].values:
#     print('hi')




    
# %% practice table

async def run():
    conn = await asyncpg.connect(user=user, password=password, database=database, host=ip)
    print('connected')

    await conn.execute(f'DROP TABLE IF EXISTS practice')
    await conn.execute(f'''
            CREATE TABLE practice (
                date DATE,
                hello INT,
                bananas INT,
                cookie INT
            );
        ''')
    await conn.execute(f'''
            INSERT INTO practice (date)
            VALUES ('2015-06-08'), ('2015-06-10'), ('2015-06-09');
        ''')

    print('all tables successfully imported')   
    await conn.close() 

asyncio.get_event_loop().run_until_complete(run())

# %%
string = 'banana oact cookie, '
string.strip()[:-1]
# %%
print(datetime.fromisoformat('2014-07-06'))
# %%
