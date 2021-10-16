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
from datetime import datetime
from pytrends.request import TrendReq
load_dotenv('.env')
nest_asyncio.apply()

#EXAMPLE make API call
# pageToken = ''
# url = 'https://www.googleapis.com/youtube/v3/search?key='+API_KEY+"&channelId="+CHANNEL_ID+"&part=snippet,id&order=date&maxResults=10000"+pageToken
# response = requests.get(url).json()

#FETCH DATA FROM YFINANCE & PYTRENDS AND CLEAN OR CSV
# #  -> UPLOAD TO POSTGRESQL
def clenseArray(array):
    array = [x.lower().replace(" ", "_")\
        .replace("-","_").replace("?","_").replace(r"/", "_").replace('.', '').replace("\'s", 's')\
        .replace(")", "").replace(r"(", "").replace("%", "").replace('all', 'all_')\
        .replace("?", "").replace("\\", "_").replace("$","").replace('&',"and").replace("'", '').replace("3m", '"3m"') for x in array]
    return array

csv_files = []
for file in os.listdir(os.getcwd()):
    if file.endswith('.csv'):
        csv_files.append(file)

#CREATE DICTIONARY OF NAMES
sp_wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
sp_wiki_df_list = pd.read_html(sp_wiki_url)
sp_df = sp_wiki_df_list[0]
sp_ticker_list = list(sp_df['Symbol'].values)
sp_name_list = list(sp_df['Security'].values)
dictionary = dict(zip(clenseArray(sp_ticker_list), clenseArray(sp_name_list)))
df = pd.DataFrame.from_dict(dictionary, orient="index")
df.to_csv("dictionary.csv")
#%%DOWNLOAD FROM YFINANCE INTO DATAFRAME
if len(csv_files) == 0 or len(csv_files) == 1:
    
    df_sp_values = yf.download(sp_ticker_list, start="2016-01-01")

    #TAKE ADJ CLOSE VALUES AND TURN INTO DF
    df_sp_prices = df_sp_values['Adj Close']
    df_sp_prices.columns = clenseArray(df_sp_prices.columns)

    #DOWNLOAD FROM PYTRENDS INTO DF
    pytrends = TrendReq(hl='en-us')
    dataset = []
    all_keywords = sp_name_list

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

#%%
#START OF NON-OPTIONAL CODE
data_sets = ['df_sp_prices.csv', 'df_sp_searches.csv']
df_sp_prices = pd.read_csv('df_sp_prices.csv')
df_sp_searches = pd.read_csv('df_sp_searches.csv')

#UPLOAD DATA TO POSTGRESQL DATABASE IN GOOGLE CLOUD
#USER AUTH FOR GOOGLE CLOUD DATABASE FROM ENVIRONMENT VARIABLES
user = os.getenv("USERNAME")
password = os.getenv("PASSWORD")
database = os.getenv("DATABASE")
ip = os.getenv("PUBLICIP")

def createTableSchema(dataf):
    col_str_two = ''
    for stock_label in dataf.columns:
        print(stock_label)
        if stock_label.lower() == 'date':
            print('howdy')
            col_str_two = col_str_two + f'{stock_label.lower()} ' + 'DATE, ' 
        elif stock_label == "'3m'":
            print('hi')
            col_str_two = col_str_two + f'"{stock_label.lower()[1:-1]}" ' + 'FLOAT, '  
        else:
            print('ahoha')
            col_str_two = col_str_two + f'{stock_label} ' + 'FLOAT, ' 

    print('done')
    return col_str_two[:-2]

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

for dataset in data_sets:
    if dataset not in csv_files:
        tblName = dataset[3:-4]
        async def run():
            conn = await asyncpg.connect(user=user, password=password, database=database, host=ip)
            print('connected')
            await conn.execute(f'DROP TABLE IF EXISTS {tblName}')
            await conn.execute(f'''
                    CREATE TABLE {tblName} (
                        {createTableSchema(eval(dataset[0:-4]))}
                    );
                ''')
            print(f'{tblName} was created successfully')
            # copy prices to table using price header
            values = []
            with open(dataset, 'r') as f:
                next(f)
                for row in f:
                    values.append(tuple(typeClean(row)))
                
            result = await conn.copy_records_to_table(
                tblName, records=values
            )
            print(result, f'import to {tblName} complete')

            await conn.close() #close the connection
        loop = asyncio.get_event_loop() #can also make single line
        loop.run_until_complete(run())
        print('all tables successfully imported')

# %%
csv_files
# %%
data_sets = ['df_sp_prices.csv', 'df_sp_searches.csv']

# %%
data_sets[0][:-4]
# %%
eval(data_sets[0][:-4])
# %%
