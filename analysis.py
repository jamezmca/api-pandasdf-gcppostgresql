# %% IMPORT FROM POSTGRESQL AND RUN DATA ANALYSIS
import os
import numpy as np
import pandas as pd
import asyncio
import nest_asyncio
import asyncpg
from dotenv import load_dotenv
load_dotenv('.env')
nest_asyncio.apply()
#START OF NON-OPTIONAL CODE
df_sp_prices = pd.read_csv('df_sp_prices.csv')
df_sp_searches = pd.read_csv('df_sp_searches.csv')
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


##FINAL OUTPUT DATA WILL BE ACCESSED IN A FRONTEND WEB APPLICATION FROM 
#POSTGRESQL DB USING GRAPHQL