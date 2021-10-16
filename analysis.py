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
load_dotenv('.env')
nest_asyncio.apply()
#START OF NON-OPTIONAL CODE
df_sp_prices = pd.read_csv('df_sp_prices.csv')
df_sp_searches = pd.read_csv('df_sp_searches.csv')
dictionary = pd.read_csv('dictionary.csv')
newDict = {}
for row in dictionary.values:
    newDict[row[0]] = row[1]
print('Successfully imported csv files :)')


# %%
# starts at zero and searches finds the max and min of the adjacent 20%
# for max and min and creates gradient, saves locations to object, and then moves onto
# next value and saves gradient etc until finish
# searches object and eliminates similar entries or upgrades a min or max 
# point if it creates a more significant gradient or a larger pecentage
practiceArray = pd.Series([0, 1, 3, 5, 10, 7, 8, 7, 5, 11, 11, 12.5, 5, 9, 6, 15])
#5 neg grads

def findLocationsOfValueDrop(array, analysisRange, decrement):
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
    negGrads = {k:v for (k,v) in gradients.items() if v['perc'] > decrement and v['grad'] < 0}

    #clean if refering to exact same drop
    prevMax = None
    prevMin = None
    for val in list(negGrads):
        # print(prevMin is not None)
        if prevMax is not None:
            if negGrads[val]['max'] == prevMax and negGrads[val]['min'] == prevMin:
                print(len(negGrads), val)
                negGrads.pop(val, None)
                print(len(negGrads), val)
        else: 
            prevMax = negGrads[val]['max']
            prevMin = negGrads[val]['min']
            # print(prevMin)
        print('next stock')


    return negGrads


yearsToWeeks = 5 * 52
analysisRange = 0.01 #range determines allowable drop gradient
xRange = yearsToWeeks * analysisRange
dec = 1.2 #signifies a 20 percent drop
# gradientLimit = 3 #for eg 20% in 2 months so 20%/60days

negGradsForAllStocks = {}
for stock in df_sp_prices:
    if stock != 'date' and stock != 'Date':
        # print(stock)
        negGradsForAllStocks[stock] = findLocationsOfValueDrop(df_sp_prices[stock], analysisRange, dec)
sum = 0
for stock in negGradsForAllStocks:
    sum += len(negGradsForAllStocks[stock])
print(f'The total number of negative gradient segments in the last 5 years\n of every stock in the S&P500 is: {sum}')
print(f'With an analysis range of {xRange} weeks')

#%%
negGradsForAllStocks['aapl']
#%% for searches compare current two weeks to avg of last 2 - 6 months
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
    
def findNormalizedSearchValue(stock, endDate, dataframe):
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
        print('successful search data')
        return np.mean(lastThreeMonths[:-1]) / average
    print('No search data for this time period')
    return None

negGradSearches = dict()
#negGradSearches = {amzn: {endDate as i: normalized value, endDate2 as i2: normalized value}}
# ivd = {v: k for k, v in dictionary.to_dict('series').items()}

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


#%% BASELINE CHECk FOR STOCK RECOVERY - CHECK VALUE AFTER 3 MONTHS
dates = df_sp_prices['Date']
amazonExample = negGradsForAllStocks['amzn']
monthsLaterPercentages = list()
threeMonthReturn = dict()
returnPercentages = dict() #histogram of percentage returns
returnSearches = dict()
returnSearchesAverages = dict()
howManyDaysLater = 60
countDips = 0
maxReturns = None
minReturns = None
naanIndex = []
for stock in negGradsForAllStocks:
    if stock not in threeMonthReturn:
        threeMonthReturn[stock] = {}
        if stock in negGradSearches:
            stockSearches = negGradSearches[stock]
    for dip in negGradsForAllStocks[stock].values():
        countDips += 1
        if dip['minIndex']+60 < len(dates):
            returnPerc = df_sp_prices[stock][dip['minIndex']+howManyDaysLater] / dip['min']
            if math.isnan(returnPerc):
                naanIndex.append([stock, dip['minIndex']])
                continue
            monthsLaterPercentages.append(returnPerc)
            threeMonthReturn[stock][dip['minIndex']] = returnPerc
            returnPercentages[f'{str(returnPerc)[:3]}'] = returnPercentages.get(f'{str(returnPerc)[:3]}', 0.00) + 1 

            if (stock in negGradSearches) and isinstance(stockSearches[dates[dip['minIndex']]], float) and (not math.isnan(stockSearches[dates[dip['minIndex']]])) and not (math.isinf(stockSearches[dates[dip['minIndex']]])) and (stockSearches[dates[dip['minIndex']]] != 0):

                if f'{str(returnPerc)[:3]}' not in returnSearches:
                    returnSearches[f'{str(returnPerc)[:3]}'] = []
                returnSearches[f'{str(returnPerc)[:3]}'].append(stockSearches[dates[dip['minIndex']]])
                
            if maxReturns == None:
                maxReturns = {'stock': stock, 'returnVal': returnPerc, 'date': df_sp_prices['Date'][dip['minIndex']], '3month': df_sp_prices[stock][dip['minIndex']+howManyDaysLater], 'min': dip['min']}
                minReturns = {'stock': stock, 'returnVal': returnPerc, 'date': df_sp_prices['Date'][dip['minIndex']], '3month': df_sp_prices[stock][dip['minIndex']+howManyDaysLater], 'min': dip['min']}
            elif returnPerc > maxReturns['returnVal']:
                maxReturns = {'stock': stock, 'returnVal': returnPerc, 'date': df_sp_prices['Date'][dip['minIndex']], '3month': df_sp_prices[stock][dip['minIndex']+howManyDaysLater], 'min': dip['min']}
            elif returnPerc < minReturns['returnVal']:
                minReturns = {'stock': stock, 'returnVal': returnPerc, 'date': df_sp_prices['Date'][dip['minIndex']], '3month': df_sp_prices[stock][dip['minIndex']+howManyDaysLater], 'min': dip['min']}
        else:

            returnPerc = df_sp_prices[stock][len(dates)-1] / dip['min']
            monthsLaterPercentages.append(returnPerc)
            threeMonthReturn[stock][dip['minIndex']] = returnPerc
            returnPercentages[f'{str(returnPerc)[:3]}'] = returnPercentages.get(f'{str(returnPerc)[:3]}', 0) + 1

            if (stock in negGradSearches) and isinstance(stockSearches[dates[dip['minIndex']]], float) and (not math.isnan(stockSearches[dates[dip['minIndex']]])) and not (math.isinf(stockSearches[dates[dip['minIndex']]])) and (stockSearches[dates[dip['minIndex']]] != 0):
                if f'{str(returnPerc)[:3]}' not in returnSearches:
                    returnSearches[f'{str(returnPerc)[:3]}'] = []
                returnSearches[f'{str(returnPerc)[:3]}'].append(stockSearches[dates[dip['minIndex']]])

returnPercentages = {k: v for k, v in sorted(returnPercentages.items(), key=lambda item: item[0])}
returnSearches = {k: v for k, v in sorted(returnSearches.items(), key=lambda item: item[0])}

#filter return searches for the same search period as it means outliers could affect data
# for val in returnSearches:
#     newList = list()
#     for chur in  returnSearches[val]:
#         if chur not in newList and not math.isinf(chur):
#             newList.append(chur)
#     returnSearches[val] = newList


for i in range(len(list(returnSearches))):
    total = 0
    for val in returnSearches[list(returnSearches)[i]]:
        total += val
    returnSearchesAverages[list(returnSearches)[i]] = total / len(returnSearches[list(returnSearches)[i]])
#%%
suma = 0.00
numOfLoses = 0
for percent in list(monthsLaterPercentages):
    if not math.isnan(percent):
        suma += percent
        if percent < 1:
            numOfLoses += 1
print(suma / len(monthsLaterPercentages))
print(suma / len(monthsLaterPercentages), numOfLoses*100/len(monthsLaterPercentages), countDips)
#%%
threeMonthReturn

#%%
maxReturns

#%%
minReturns
#%%
returnSearches
#%%
returnPercentages

#%%
returnSearchesAverages
#if the searches are below 
#%%
returnStats = np.array(monthsLaterPercentages)
med = np.median(returnStats)
med

#%%
negGradSearches
#%%PLOTTING
import matplotlib.pyplot as plt
labels = returnSearchesAverages.keys()
men_means = returnSearchesAverages.values()
women_means = returnPercentages.values()

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, men_means, width, label='Searches')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Searches')
ax.set_title('Searches')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3)

fig.tight_layout()

plt.show()

#%%PLOTTING
import matplotlib.pyplot as plt
labels = returnSearchesAverages.keys()
men_means = returnSearchesAverages.values()
women_means = returnPercentages.values()

x = np.arange(len(labels))  # the label locations
x = [2*i for i in x]

width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects2 = ax.bar(x, women_means, width, label='Prices')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Returns')
ax.set_title('Num of Returns')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects2, padding=3)
fig.tight_layout()
plt.figure(figsize=(10, 3))  # width:10, height:8

plt.show()
#%%
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
#POSTGRESQL DB USING GRAPHQL OR A FULL STACK TYPESCRIPT APPLICATION
#USING THAT VIDEO REFERENCE



#this is a key diversion in the analysis as I am intentionally choosing not to group
#encountered drops greater than the limit together. Two periods of declination
#could infact be part of the actual drop and I could create an amalgamation function
#that amalgamates drops into overall drops classified within some variation limit
#However the issue with this is that my analysis is jsut looking at drops that
#are larger than 20% for example. An investor can't predict whether or not
#the stock price has finished dropping and so an amalgamation technique
#would assume that you could know that the stock had finished dropping.
#This could be a good additional investigation area though as this amalgamation technique
#would allow me to create a sample of total drop percentages and then you could
#analysis stock value recoveries. I would classify complete drops knowing the start of the 
#drop and the statistally significant end of the drop.
def amalgamate(obj):
    length = len(obj.keys())
    finalObj = {}
    def checkLeft(): 
        #check for bigger values on left, considering not making the grad too much less
        pass
    def checkRight(): #check for smaller values on right, not making the grad too much less
        pass
    def diffInDays(start, end):
        return
    prevMax = None
    prevMin = None
    for i in range(length):
        key = list(obj.keys())[i]
        val = obj[key]
        print(val)
    return
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
