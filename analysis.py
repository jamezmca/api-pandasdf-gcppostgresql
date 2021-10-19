# %% IMPORT FROM POSTGRESQL AND RUN DATA ANALYSIS
import os
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats, optimize, interpolate
import math
import asyncio
import nest_asyncio
import asyncpg
from dotenv import load_dotenv
from datetime import date
import plotly.express as px
load_dotenv('.env')
nest_asyncio.apply()

###TODO
#CAMEL CASE FOR GLOBAL VARIABLES AND _ FOR FUNCTION VARIABLES
#CHANGE FROM NEWS TO ALL SEARCHES AND RERUN CURRENT ANALYSIS IN MAIN.PY
#MAKE SUM WORK

#INITIALIZE DATAFRAMES FROM CSV FILES
df_sp_prices = pd.read_csv('df_sp_prices.csv')
df_sp_searches = pd.read_csv('df_sp_searches.csv')
df_sp_price = pd.read_csv('df_sp_price.csv')
dictionary = pd.read_csv('dictionary.csv')
dates = df_sp_prices['Date']
newDict = {}
for row in dictionary.values:
    newDict[row[0]] = row[1]
print('Successfully imported csv files :)')

#REQUIRED INPUT VARIABLES
analysisRange = 0.01 #range determines allowable drop gradient
dec = 1.2 #signifies a 20 percent decrease in value
howManyDaysLater = 60 #2 month return

#LIST OF ALL FUNCTIONS


def initializeWeekBins(datesArray):
    hokeyPokey = {}
    for i in range(len(datesArray)):
        if i == 0:
            hokeyPokey[f'{i} {i+5}'] = [0, {}] #dict contains stock and count of particular stock
        elif i % 5 == 0:
            hokeyPokey[f'{i} {i+5}'] = [0, {}]
    return  hokeyPokey

def findOverlappingWeeks(wklyBinIndexes, maxIndex, minIndex):
    maxOverlap = 0
    bongo = [] #array of bins to increment
    for key in wklyBinIndexes:
        keyStart, keyEnd =  key.split(' ')
        keyStart, keyEnd, maxIndex, minIndex = [int(keyStart), int(keyEnd), int(maxIndex), int(minIndex)]
        overlap = minIndex - keyStart if minIndex < keyEnd else keyEnd - maxIndex
        hasOverlap = (maxIndex >= keyStart and maxIndex < keyEnd) or (minIndex < keyEnd and minIndex >= keyStart)
        #if find bigger overlap, reset overlap and create for new max overlap
        if (maxIndex >= keyStart and minIndex < keyEnd) or (maxIndex <= keyStart and minIndex > keyEnd):
            overlap = 5
            bongo.append(key)
            maxOverlap = 5
        elif hasOverlap and overlap >= maxOverlap:
            maxOverlap = overlap
            bongo.append(key)
    return bongo

def findLocationsOfValueDrop(array, analysisRange, decrement, stack, how_many_days_later, week_bins): #FIND ALL STATS IN HERE AND MAKE IT A BUNCH OF SMALLER FUNCTIONS AND ALSO INCLUDE WHICHEVER DATE BINS IT'S IN
    gradients = {}
    length = len(array)
    for i in range(length - how_many_days_later - math.floor(analysisRange*length)):
        tempArray = list(array[i:i+math.floor(analysisRange*length)])
        maximum = max(tempArray)
        maxIndex = tempArray.index(maximum) + i
        minimum = min(tempArray) if min(tempArray) != 0 else 0.001
        minIndex = tempArray.index(0) + i if minimum == 0.001 else tempArray.index(minimum) + i
        decreaseRatio = maximum / minimum
        if maxIndex - minIndex != 0:
            deltaX = maxIndex - minIndex
        else: #can just set it to a positive integer cause I only want negative gradients and max and min are the same value
            deltaX = 15
        dateBins = findOverlappingWeeks(list(week_bins), maxIndex, minIndex)
        grad = (maximum - minimum) / deltaX #possibly should be perc/deltaX
        multiplier = array[minIndex+howManyDaysLater] / minimum
        completeObj = {"max": maximum, 
                            "min": minimum, 
                            "grad": grad, 
                            "decreaseRatio": decreaseRatio,
                            'minIndex': minIndex,
                            'maxIndex': maxIndex,
                            'dateBins': dateBins,
                            'multiplier': multiplier}
        if grad < 0 and decreaseRatio > decrement and completeObj not in gradients.values(): #ensures no double ups
            gradients[i] = completeObj
            for week in dateBins:
                if stack not in week_bins[week][1]:
                    week_bins[week][0] += 1
                    week_bins[week][1][stack] = multiplier
                elif multiplier > week_bins[week][1][stack]:
                    week_bins[week][1][stack] = multiplier
    return gradients, week_bins


# %% PART 1: FIND LOCATIONS OF NEGATIVE GRADIENT 
yearsToWeeks = 5 * 52
xRange = yearsToWeeks * analysisRange
#ALSO MAKE TO MAKE A DICT THAT SHOWS WEEK -> STOCK -> list of returns
#ALSO MAKE A DICT THAT SHOES THE AVERAGE RETURN FOR EACH STOCK AFTER 20% dip and maybe check how ranking determines return
    #BUT START OFF MAKING A LIST OF RETURNS FOR EACH STOCK IN A DICT AND THEN AVERAGE IT AND MAKE GRAPPHS OF QUARTILES
weekBins = initializeWeekBins(dates)
negGradsForAllStocks = {}
for stock in df_sp_prices:
    if stock != 'date' and stock != 'Date':
        negGradsForAllStocks[stock], weekBins = findLocationsOfValueDrop(df_sp_prices[stock], analysisRange, dec, stock, howManyDaysLater, weekBins)
print(f'finished part 1 for analysis range: {xRange} weeks')
# print(f'The total number of negative gradient segments in the last 5 years\n of every stock in the S&P500 is: {sum} \nWith an analysis range of {xRange} weeks.')


#%% PART 2: FROM WEEKBINS, CREATE A HISTOGRAM DICT THAT LISTS EVERY RETURN, AND A LIST OF ALL THE SHARED NUM OF ENTRIES (interconnectedness)
multiplierNumWeeks = dict()
for veek in weekBins.values():
    for ret in veek[1].values():
        if ret in multiplierNumWeeks:
            multiplierNumWeeks[ret].append(veek[0])
        else:
            multiplierNumWeeks[ret] = [veek[0]]

multiplierNumWeeksAverage = {k:np.mean(np.array(v)) for (k,v) in multiplierNumWeeks.items()}
multiplierNumWeeksAverage = {k: v for k, v in sorted(multiplierNumWeeksAverage.items(), key=lambda item: item[0])}

#CREATE HISTOGRAM FOR THE RANGES FOR EG 0.7 = [...ALL THE DROPS]
    #AND THEN CREATE PERCENTILES FROM THOSE LISTS
multiplerHistogram = dict() #{1.2: [[...list of interconnecteds], numOfinterconnecteds, avgInterconnecteds]}
for multiplier in multiplierNumWeeksAverage:
    prefix = np.around(multiplier, 1)
    if prefix in multiplerHistogram:
        multiplerHistogram[prefix]['list'].append(multiplierNumWeeksAverage[multiplier])
        listItems = np.array(multiplerHistogram[prefix]['list'])
        multiplerHistogram[prefix]['listLength'] = len(listItems)
        multiplerHistogram[prefix]['mean'] = np.mean(listItems)
        multiplerHistogram[prefix]['median'] = np.median(listItems)
        multiplerHistogram[prefix]['LQ'] = np.quantile(listItems, 0.25)
        multiplerHistogram[prefix]['UQ'] = np.quantile(listItems, 0.75)
        multiplerHistogram[prefix]['5th'] = np.quantile(listItems, 0.05)
    else: 
        multiplerHistogram[prefix] = {  'list': [multiplierNumWeeksAverage[multiplier]],
                                        'listLength': 1, 
                                        'mean': multiplierNumWeeksAverage[multiplier],
                                        'median': [multiplierNumWeeksAverage[multiplier]],
                                        'LQ': [multiplierNumWeeksAverage[multiplier]],
                                        'UQ': [multiplierNumWeeksAverage[multiplier]],
                                        '5th': [multiplierNumWeeksAverage[multiplier]]}

MultiplierHistogramFiltered = {k:v for k,v in multiplerHistogram.items() if k > 0.6 and k < 2}
#%% PART 2: COMPARES LATEST SEARCHES TO THE HISTORICAL AVERAGE OVER A GREATER PERIOD OF TIME
print(list(MultiplierHistogramFiltered))
#%% PART 3: BASELINE CHECk FOR STOCK RECOVERY - CHECK VALUE AFTER 3 MONTHS


#%% PART 5: CREATE INTERCONNECTEDNESS HISTOGRAM - WEEKBINS NEEDS FIXING

#%% PART 6: FIND THE INTERCONNECTEDNESS AND ASSOCIATED RETURN

#%% PART 7: CHECK HOW INTERCONNECTEDNESS REDUCES CHANCE OF A LOSS AND HOW IT COMPARES TO S&P500 IF THEY BOTH DROP
#first check if dates do overlap significantly
#then compare it to the returns maybe by how much they overlap
#then compare it to a straight true or false overlap
#then check how much above and beyond the avg individual return was above s&p500
#also just baseline the s&p500 return after 3 months and compare to all stock averages
#also need to filter out mad values in all stock answers before averaging
#also check if snp500 experienced any of these drops above just the min of (30%)? of individual stocks for example
#%% PROCESS ANY UNWANTED DATA IE REMOVING LOW SIGNIFICANCE RETURNS

#%% ONE SAMPLE T-TESTING TO FIND P-VALUE
#need to do indepedance, Equality of variance and normality of data / residuals
#make a scatter plot for each return and the associated search value and a line of best fit
#could use an object or a dataframe

#%%GRAPPPHHHHIIIIINNNNNGGGGGGGGG
#GRAPHS TO PRODUCE
#- ALL interconnectedness vs returns both whole plot and mean, median, LQ, UQ, 5th, IQR, RANGE, 
#- INDIVIDUAL STOCKS REBOUND AND INDIVIDUAL STOCKS PERCENTAGE CHANCE OF TAKING A NEGATIVE
#- S&P 500 VALUE / DROPS AGAINST INTERCONNECTEDNESS BOTH ON WEEKBIN AXIS
#- DF S&P 500 PRICE RETURNS VS RETURNS OF STOCKS AFTER DROP (ESP THOSE WITH HIGH INTERCONNECTNESS)
#- SORTED HISTOGRAM OF STOCKS MOST GURANTEED TO PRODUCE A RESULT AND ANOTHER WITH THE HIGHEST RESULT
#-
#-
#-
#%%PLOTTING NUM OF SHARED DROPS WITH RETURN FROM
cheese = [[x,v['mean']] for x,v in MultiplierHistogramFiltered.items()]

cheese_df = pd.DataFrame(cheese, columns=['Return Multiplier', 'Interconnectedness'])
cheese_df.describe()
fig = px.scatter(cheese_df, x="Interconnectedness", y="Return Multiplier")
fig.show()


# %% PLOTTING S&P500 VALUE AGAINST 
ham = [[x, w] for x,w in {k:v for k,v in multiplierNumWeeksAverage.items() if k > 0.6 and k < 2}.items()]

ham_df = pd.DataFrame(ham, columns=['Return Multiplier', 'Interconnectedness'])
ham_df.describe()
fig = px.scatter(ham_df, x="Interconnectedness", y="Return Multiplier")
fig.show()

# %%
multiplierNumWeeksAverage
# %%
p = np.poly1d(np.polyfit(ham_df['Interconnectedness'], ham_df['Return Multiplier'], 1))

# %%
p
# %%
regr_results = sp.stats.linregress(ham_df['Interconnectedness'], ham_df['Return Multiplier'])
print(regr_results)
# %%
ham_df.describe()
# %%
