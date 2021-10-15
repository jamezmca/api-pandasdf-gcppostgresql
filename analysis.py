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
df_sp_prices = pd.read_csv('df_sp_prices.csv', 'r')
df_sp_searches = pd.read_csv('df_sp_searches.csv', 'r')
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
practiceArray = pd.Series([0, 1, 3, 5, 10, 7, 8, 7, 5, 11, 11, 12.5, 5, 9, 6, 15])
#5 neg grads

def twentyPercent(array, analysisRange, decrement):
    gradients = {}

    length = len(array)
    for i in range(length):
        if i > (1-analysisRange)*length:
            break
        tempArray = list(array[i:i+math.ceil(analysisRange*length)])
        maximum = max(tempArray)
        maxIndex = tempArray.index(maximum)
        minimum = min(tempArray) if min(tempArray) != 0 else 0.001
        minIndex = tempArray.index(0) if minimum == 0.001 else tempArray.index(minimum)
        if maxIndex - minIndex != 0:
            deltaX = maxIndex - minIndex
        else: #can just set it to a positive integer cause I only want negative gradients
            #and max and min are the same value
            deltaX = 10
        grad = (maximum - minimum) / deltaX #possibly should be perc/deltaX
        gradients[i] = {"max": maximum, 
                        "min": minimum, 
                        "grad": grad, 
                        "perc": maximum/minimum,
                        'minIndex': minIndex,
                        'maxIndex': maxIndex}
    bigGradients = {k:v for (k,v) in gradients.items() if v['perc'] > decrement and v['grad'] < 0}
    return bigGradients

def diffInDays(start, end):
    return

def amalgamate(obj):
    length = len(obj.keys())
    finalObj = {}
    def checkLeft(): 
        #check for bigger values on left, considering not making the grad too much less
        pass
    def checkRight(): #check for smaller values on right, not making the grad too much less
        pass
        
    prevMax = None
    prevMin = None
    for i in range(length):
        key = list(obj.keys())[i]
        val = obj[key]
        print(val)




    return

ansisRange = 0.1
dec = 1.2
gradientLimit = 3 #for eg 20% in 2 months so 20%/60days
negGrads = twentyPercent(practiceArray, ansisRange, dec)
finalGrads = amalgamate(negGrads)



# %%
practiceArray = pd.Series([0, 1, 3, 5, 10, 7, 8, 7, 5, 11, 11, 12.5, 5, 9, 6, 15])

# %%
# %%
