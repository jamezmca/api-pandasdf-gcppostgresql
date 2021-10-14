# %% IMPORT FROM POSTGRESQL AND RUN DATA ANALYSIS
import os
import numpy as np
import pandas as pd
import math
import asyncio
import nest_asyncio
import asyncpg
from dotenv import load_dotenv
load_dotenv('.env')
nest_asyncio.apply()
#START OF NON-OPTIONAL CODE
df_sp_prices = pd.read_csv('df_sp_prices.csv')
df_sp_searches = pd.read_csv('df_sp_searches.csv')
print('Successfully imported csv files :)')
##ASSUMPTIONS
#Intraday losses greater than 20% are unlikely to be a market failure
# and are more probably a single stock failure
# 

#function that compares the current search average to the averages
#of the periods say 2-6 months before and after the stock drops
#as a normalized value

#or i could compute the running average or search activity over the last 3-6months 
#and compare that to the current value, and then add the normalised value to 
#the average and compute the next value


##FINAL OUTPUT DATA WILL BE ACCESSED IN A FRONTEND WEB APPLICATION FROM 
#POSTGRESQL DB USING GRAPHQL
# %%
def findLocationsOfValueDrop(stock):
    #function tolerances of 5% up on lowest value to be considered part of the same
    #decrease;
    #minimum drop rate of 5% over 15days
    #find start bounds and end bounds
    print('hello')

tempDf = df_sp_prices['amzn']

# %%
# starts at zero and searches finds the max and min of the adjacent 20%
# for max and min and creates gradient, saves locations to object, and then moves onto
# next value and saves gradient etc until finish
# searches object and eliminates similar entries or upgrades a min or max 
# point if it creates a more significant gradient or a larger pecentage
def twentyPercent(array):
    gradients = {}

    length = len(array)
    print(array[0])
    for i in range(length):
        if i > 0.95*length:
            break
        tempArray = list(array[i:i+math.ceil(0.05*length)])
        maximum = max(tempArray)
        maxIndex = tempArray.index(maximum)
        minimum = min(tempArray) if min(tempArray) != 0 else 0.001
        minIndex = tempArray.index(minimum)
        deltaX = abs(maxIndex - minIndex) if abs(maxIndex - minIndex) > 0 else 0.001
        grad = (maximum - minimum) / deltaX
        gradients[i] = {"max": maximum, "min": minimum, "grad": grad, "perc": maximum/minimum}
    bigGradients = {k:v for (k,v) in gradients.items() if v['perc'] > 1.2}
    i = 0
    while i < len(bigGradients):
        if bigGradients[i]['max'] <  bigGradients[i+1]['max'] and bigGradients[i]['min'] == bigGradients[i+1]['min']:
            

twentyPercent(tempDf)

# %%
bigGradients

# %%
# %%
