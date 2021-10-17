# %% IMPORT FROM POSTGRESQL AND RUN DATA ANALYSIS
import os
import numpy as np
import pandas as pd
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
def findNearestDate(searchesDates, dat): #finds the closest day in the search array
    delta = 3000
    d1 = date(int(dat.split('-')[0]), int(dat.split('-')[1]), int(dat.split('-')[2]))
    for day in searchesDates:
        d0 = date(int(day.split('-')[0]), int(day.split('-')[1]), int(day.split('-')[2]))
        diff = d1 - d0
        if abs(diff.days) < delta:
            closestDay = day
            delta = abs(diff.days)
    return closestDay

def initializeWeekBins(datesArray):
    hokeyPokey = {}
    for i in range(len(datesArray)):
        if i == 0:
            hokeyPokey[f'{i} {i+5}'] = 0
        elif i % 5 == 0:
            hokeyPokey[f'{i} {i+5}'] = 0
    return  hokeyPokey

def initializeStocksInWeeklyBins(weeklyBins):
    vanilla = {}
    for key in weeklyBins:
        vanilla[key] = {}
    return vanilla

def findReturnSearchAverages(rs):
    returnSearches = rs
    for i in range(len(list(returnSearches))):
        total = 0
        for val in returnSearches[list(returnSearches)[i]]:
            total += val
    return total / len(returnSearches[list(returnSearches)[i]])

def hasOverLap(wklyBinIndexes, maxIndex, minIndex):
    maxOverLap = 0
    bongo = [] #array of bins to increment
    for key in wklyBinIndexes:
        keyStart, keyEnd =  key.split(' ')
        keyStart, keyEnd, maxIndex, minIndex = [int(keyStart), int(keyEnd), int(maxIndex), int(minIndex)]
        overlap = max(maxIndex - keyStart, keyEnd - minIndex)
        #if find bigger overlap, reset overlap and create for new max overlap
        if (maxIndex >= keyStart and minIndex < keyEnd) or (maxIndex <= keyStart and minIndex > keyEnd):
            overlap = 5
            bongo.append(key)
        elif (maxIndex >= keyStart and maxIndex < keyEnd and overlap > maxOverLap) or (minIndex < keyEnd and minIndex >= keyStart and overlap > maxOverLap):
            maxOverlap = overlap
            bongo.append(key)
    return bongo

def findNormalizedSearchValue(stock, endDate, dataframe): #need to fix this so it doesn't return dud values!!!!!!!!!!!!! can't return nan, inf, Nonetype, only float
#finds the current search interest of a stock over
#the last two weeks as compared to the last # months
    num = 6 #weekly results is num / 4 is months
    if stock in dataframe:
        stockSearches = dataframe[stock]
        # print(stockSearches)
        indexOfEndDate = list(dataframe['date']).index(endDate)
        # indexOfEndDate = stockSearches.index(endDate)
        lastThreeMonths = stockSearches[indexOfEndDate - num : indexOfEndDate+1]
        #going to compare last two weeks to average of last three months exclusive of last two weeks
        average = np.mean(lastThreeMonths[:-2])
        return np.mean(lastThreeMonths[:-1]) / (average if average != 0 else 1)  #need to fix this cell
    return None

def datesOverlap(wklybins, stocksInWeeklyBins, dip, stock): #needs to take the stock to make sure im not double counting the stock but should still average dates
    # a nice idea but what i actually want to do is create week bins and just histogram them for every stock drop
    wklyBinsCopy = wklybins.copy()
    stocksInBinsCopy = stocksInWeeklyBins.copy()
    overlapIndexes = hasOverLap(wklybins, dip['maxIndex'], dip['minIndex'])
    
    for overlap in overlapIndexes:
        if stock not in stocksInBinsCopy[overlap].keys():
            wklyBinsCopy[overlap] = wklyBinsCopy[overlap] + 1
        stocksInBinsCopy[overlap][stock] = stocksInBinsCopy[overlap].get(stock, 0) + 1
    return wklyBinsCopy, stocksInBinsCopy

def findReturnVsInterconnectedness(weekbinss, dippy, how_many_days_later, stock, associatedReturnDict, associatedReturnDictInv, associatedReturnList):
    if dippy['minIndex']+how_many_days_later < len(dates):
        returnPerc = df_sp_prices[stock][dip['minIndex']+howManyDaysLater] / dippy['min']
    else:
        returnPerc = df_sp_prices[stock][len(dates)-1] / dippy['min']
    associatedReturnDict[returnPerc] = max([weekbinss[x] for x in hasOverLap(weekbinss, dippy['maxIndex'], dippy['minIndex'])])
    associatedReturnDictInv[max([weekbinss[x] for x in hasOverLap(weekbinss, dip['maxIndex'], dippy['minIndex'])])] = associatedReturnDictInv.get(max([weekbinss[x] for x in hasOverLap(weekbinss, dippy['maxIndex'], dippy['minIndex'])]), 0) + 1
    associatedReturnList.append([returnPerc, max([weekbinss[x] for x in hasOverLap(weekbinss, dippy['maxIndex'], dippy['minIndex'])])])
    return associatedReturnDict, associatedReturnDictInv, associatedReturnList

def findLocationsOfValueDrop(array, analysisRange, decrement): #FIND ALL STATS IN HERE AND MAKE IT A BUNCH OF SMALLER FUNCTIONS AND ALSO INCLUDE WHICHEVER DATE BINS IT'S IN
    gradients = {}
    length = len(array)

    for i in range(length):
        if i > (1-analysisRange)*length:
            break
        tempArray = list(array[i:i+math.ceil(analysisRange*length)])
        maximum = max(tempArray)
        maxIndex = tempArray.index(maximum) + i
        minimum = min(tempArray) if min(tempArray) != 0 else 0.001
        minIndex = tempArray.index(0) + i if minimum == 0.001 else tempArray.index(minimum) + i
        if maxIndex - minIndex != 0:
            deltaX = maxIndex - minIndex
        else: #can just set it to a positive integer cause I only want negative gradients and max and min are the same value
            deltaX = 15
        dateBins = hasOverLap(initializeWeekBins(dates), maxIndex, minIndex)
        grad = (maximum - minimum) / deltaX #possibly should be perc/deltaX
        gradients[i] = {"max": maximum, 
                        "min": minimum, 
                        "grad": grad, 
                        "perc": maximum/minimum,
                        'minIndex': minIndex,
                        'maxIndex': maxIndex,
                        'dateBins': dateBins}
    negGrads = {k:v for (k,v) in gradients.items() if v['perc'] > decrement and v['grad'] < 0}

    #clean return if refering to exact same drop
    prevMax = None
    prevMin = None
    for val in list(negGrads):
        # print(prevMin is not None)
        if prevMax is not None:
            if negGrads[val]['max'] == prevMax and negGrads[val]['min'] == prevMin:
                # print(len(negGrads), val)
                negGrads.pop(val, None)
                # print(len(negGrads), val)
        else: 
            prevMax = negGrads[val]['max']
            prevMin = negGrads[val]['min']
        # print('next stock')
    return negGrads

# %% PART 1: FIND LOCATIONS OF NEGATIVE GRADIENT
yearsToWeeks = 5 * 52
xRange = yearsToWeeks * analysisRange

negGradsForAllStocks = {}
for stock in df_sp_prices:
    if stock != 'date' and stock != 'Date':
        negGradsForAllStocks[stock] = findLocationsOfValueDrop(df_sp_prices[stock], analysisRange, dec)

print(f'The total number of negative gradient segments in the last 5 years\n of every stock in the S&P500 is: {sum} \nWith an analysis range of {xRange} weeks.')


#%% PART 2: COMPARES LATEST SEARCHES TO THE HISTORICAL AVERAGE OVER A GREATER PERIOD OF TIME
negGradSearches = dict()

for stock in negGradsForAllStocks:
    for dip in negGradsForAllStocks[stock]:
        particularDip = negGradsForAllStocks[stock][dip]
        closestSearchDate = findNearestDate(df_sp_searches['date'], df_sp_prices['Date'][particularDip['minIndex']])
        normalizedSeachValue = findNormalizedSearchValue(newDict[stock], closestSearchDate, df_sp_searches)
        if stock in negGradSearches:
            negGradSearches[stock][df_sp_prices['Date'][particularDip['minIndex']]] = normalizedSeachValue 
        else:
            negGradSearches[stock] = {}
            negGradSearches[stock][df_sp_prices['Date'][particularDip['minIndex']]] = normalizedSeachValue 

print(f'Number of negGradSearches associated to drops: {len(negGradSearches.keys())}.')


#%% PART 3: BASELINE CHECk FOR STOCK RECOVERY - CHECK VALUE AFTER 3 MONTHS
threeMonthReturn = dict()
monthsLaterPercentages = list()
maxReturns = None
minReturns = None
countDips = 0
countSearches = 0

for stock in negGradsForAllStocks:
    for dip in negGradsForAllStocks[stock].values():
        if stock not in threeMonthReturn: #For 3 month return
            threeMonthReturn[stock] = {}
        countDips += 1
        if dip['minIndex']+howManyDaysLater < len(dates):
            returnPerc = df_sp_prices[stock][dip['minIndex']+howManyDaysLater] / dip['min']
        else:
            returnPerc = df_sp_prices[stock][len(dates)-1] / dip['min']
        if maxReturns == None:
            maxReturns = {'stock': stock, 'returnVal': returnPerc, 'date': df_sp_prices['Date'][dip['minIndex']], '3month': df_sp_prices[stock][dip['minIndex']+howManyDaysLater], 'min': dip['min']}
            minReturns = {'stock': stock, 'returnVal': returnPerc, 'date': df_sp_prices['Date'][dip['minIndex']], '3month': df_sp_prices[stock][dip['minIndex']+howManyDaysLater], 'min': dip['min']}
        elif returnPerc > maxReturns['returnVal']:
            maxReturns = {'stock': stock, 'returnVal': returnPerc, 'date': df_sp_prices['Date'][dip['minIndex']], '3month': df_sp_prices[stock][dip['minIndex']+howManyDaysLater], 'min': dip['min']}
        elif returnPerc < minReturns['returnVal']:
            minReturns = {'stock': stock, 'returnVal': returnPerc, 'date': df_sp_prices['Date'][dip['minIndex']], '3month': df_sp_prices[stock][dip['minIndex']+howManyDaysLater], 'min': dip['min']}
        monthsLaterPercentages.append(returnPerc)
        threeMonthReturn[stock][dip['minIndex']] = returnPerc




#%%PART 4: CHECK SEARCHES AGAINST RETURN VALUES
returnPercentages = dict() #histogram of percentage returns
returnSearches = dict()
returnSearchesAverages = dict()
naanIndex = []
coolObj = []

weekBins = initializeWeekBins(dates)
stocksInWeekBins = initializeStocksInWeeklyBins(weekBins)

for stock in negGradsForAllStocks:
    print(stock)
    if stock in negGradSearches: #REQ FOR BELOW
        stockSearches = negGradSearches[stock]

    for dip in negGradsForAllStocks[stock].values():
        weekBins, stocksInWeekBins = datesOverlap(weekBins, stocksInWeekBins, dip, stock)
        if dip['minIndex']+howManyDaysLater < len(dates):
            returnPerc = df_sp_prices[stock][dip['minIndex']+howManyDaysLater] / dip['min']
            if math.isnan(returnPerc):
                naanIndex.append([stock, dip['minIndex']])
                continue
        else:
            returnPerc = df_sp_prices[stock][len(dates)-1] / dip['min']

        returnPercentages[f'{str(returnPerc)[:3]}'] = returnPercentages.get(f'{str(returnPerc)[:3]}', 0.00) + 1 
        if (stock in negGradSearches) and isinstance(stockSearches[dates[dip['minIndex']]], float) and (not math.isnan(stockSearches[dates[dip['minIndex']]])) and not (math.isinf(stockSearches[dates[dip['minIndex']]])) and (stockSearches[dates[dip['minIndex']]] != 0):
            if f'{str(returnPerc)[:3]}' not in returnSearches:
                returnSearches[f'{str(returnPerc)[:3]}'] = []
            returnSearches[f'{str(returnPerc)[:3]}'].append(stockSearches[dates[dip['minIndex']]])
            coolObj.append([returnPerc, stockSearches[dates[dip['minIndex']]]])

#order the objects
returnPercentages = {k: v for k, v in sorted(returnPercentages.items(), key=lambda item: item[0])}
returnSearches = {k: v for k, v in sorted(returnSearches.items(), key=lambda item: item[0])}
returnSearchesAverages = findReturnSearchAverages(returnSearches)

#%% PART 5: FIND THE INTERCONNECTEDNESS AND ASSOCIATED RETURN
associatedReturnDict = dict()
associatedReturnDictInv = dict()
associatedReturnList = []

for stock in negGradsForAllStocks:
    for dip in negGradsForAllStocks[stock].values():
        associatedReturnDict, associatedReturnDictInv, associatedReturnList = findReturnVsInterconnectedness(weekBins, dip, howManyDaysLater, stock, associatedReturnDict, associatedReturnDictInv, associatedReturnList)
     
associatedReturnDict = {k: v for k, v in sorted(associatedReturnDict.items(), key=lambda item: item[0])}
print('finishedddddddd')
#if dates have more than a 50% overlap with dates in average max-min histogram, increment count in histograms where key is the average date range(start and finish) which is stored in a separate  dictionary
#key keys showing all the start and end days that overlap with the average and then become part of the average

#%% PART 6: CHECK HOW INTERCONNECTEDNESS REDUCES CHANCE OF A LOSS AND HOW IT COMPARES TO S&P500 IF THEY BOTH DROP
#first check if dates do overlap significantly
#then compare it to the returns maybe by how much they overlap
#then compare it to a straight true or false overlap
#then check how much above and beyond the avg individual return was above s&p500
#also just baseline the s&p500 return after 3 months and compare to all stock averages
#also need to filter out mad values in all stock answers before averaging
#also check if snp500 experienced any of these drops above just the min of (30%)? of individual stocks for example
#%% PROCESS ANY UNWANTED DATA IE REMOVING LOW SIGNIFICANCE RETURNS
#Remove outlier search values > 5
#Remove any return multipliers > 3
coolObj = [[x[0], x[1]] for x in coolObj if x[1] < 5 and x[0] < 3 and x[1] > 0.5 and x[0] > 0.6]

print(f'datasize is {len(coolObj)} pairs longs')

#MAYBE MAKE A LOG TRANSFORMATION ON SEARCH INDEX OR A SQUAREROOT TRANSFORMATION PERHAPS
#%%Make scatter plot of multiplier vs searches dataframe
relationshipDF = pd.DataFrame(np.array(coolObj), columns=['multiplier', 'normSearch'])

#%% ONE SAMPLE T-TESTING TO FIND P-VALUE
#need to do indepedance, Equality of variance and normality of data / residuals
#make a scatter plot for each return and the associated search value and a line of best fit
#could use an object or a dataframe

relationshipDF.describe()
# relationshipDF.head(10)
#%%
len(associatedReturnDict)
#%%
min(relationshipDF['normSearch'])
#%%
len(df_sp_searches.columns)
#%%
negGradsForAllStocks['dal']
#%%
returnSearches
#%%
returnPercentages

#%%
returnSearchesAverages
#if the searches are below 

#%%
associatedReturnDictInv = {k: v for k, v in sorted(associatedReturnDictInv.items(), key=lambda item: item[0])}
associatedReturnDictInv

#%%
stocksInWeekBins['1050 1055']
#%%
returnStats = np.array(monthsLaterPercentages)
med = np.median(returnStats)
med

#%%
weekBins
#%%PLOTTING NUM OF SHARED DROPS WITH RETURN FROM
cheese = np.array(associatedReturnList)
cheese_df = pd.DataFrame(cheese, columns=['Return Multiplier', 'Interconnectedness'])
cheese_df.describe()
fig = px.scatter(cheese_df, x="Interconnectedness", y="Return Multiplier")
fig.show()

#%%PLOTTING
cheese_df.describe()

#%%


# %%
practiceArray = pd.Series([0, 1, 3, 5, 10, 7, 8, 7, 5, 11, 11, 12.5, 5, 9, 6, 15])

# %%
# %%
# %%
negGradsForAllStocks['amzn']
# %%
# %%
print(min(monthsLaterPercentages))
# %%
for name in df_sp_searches.columns:
    print(name)

# %%
xRange
# %%
james, henry = 'banana phone'.split(' ')
print(henry)
# %%

# %%
