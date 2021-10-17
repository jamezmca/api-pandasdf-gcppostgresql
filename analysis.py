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
import matplotlib.pyplot as plt #change to plotly
load_dotenv('.env')
nest_asyncio.apply()
#START OF NON-OPTIONAL CODE
df_sp_prices = pd.read_csv('df_sp_prices.csv')
df_sp_searches = pd.read_csv('df_sp_searches.csv')
df_sp_price = pd.read_csv('df_sp_price.csv')
dictionary = pd.read_csv('dictionary.csv')
newDict = {}
for row in dictionary.values:
    newDict[row[0]] = row[1]
print('Successfully imported csv files :)')

#List all functions here i think


#CHANGE FROM NEWS TO ALL SEARCHES AND RERUN CURRENT ANALYSIS IN MAIN.PY
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
    
#need to fix this so it doesn't return dud values!!!!!!!!!!!!! can't return nan, inf, Nonetype, only float
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
        # print('successful search data')
        return np.mean(lastThreeMonths[:-1]) / (average if average != 0 else 1)  #need to fix this cell
    # print(stock)
    # print('No search data for this time period')
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

print(f'finito for negGradSearches len: {len(negGradSearches.keys())}')
#%% BASELINE CHECk FOR STOCK RECOVERY - CHECK VALUE AFTER 3 MONTHS
dates = df_sp_prices['Date']
monthsLaterPercentages = list()
threeMonthReturn = dict()
returnPercentages = dict() #histogram of percentage returns
returnSearches = dict()
returnSearchesAverages = dict()
dateHistogram = dict()
dateHistogramRanges = dict() #holds an array of the keys that it averages to 
#create a final date for eg {"15 28": {max: [index1, index2]}, min: [index1, index2]}
howManyDaysLater = 60
countDips = 0
countSearches = 0
maxReturns = None
minReturns = None
naanIndex = []
coolObj = []


#first check if dates do overlap significantly
#then compare it to the returns maybe by how much they overlap
#then compare it to a straight true or false overlap
#then check how much above and beyond the avg individual return was above s&p500
#also just baseline the s&p500 return after 3 months and compare to all stock averages
#also need to filter out mad values in all stock answers before averaging
#also check if snp500 experienced any of these drops above just the min of (30%)? of individual stocks for example
def datesOverlap(histo, datesHisto, dip, stock): #needs to take the stock to make sure im not double counting the stock but should still average dates
    #could just add a third varibale to ranges which pushes the stock 
    # print(histo)
    # a nice idea but what i actually want to do is create week bins and just histogram them for every stock drop
    def hasOverLap(maxIndex, minIndex, datesToCheckMax, datesToCheckMin):
        #datesIndex = [dates2Max, dates2Min] from key.split(' ')
        halfLength = datesToCheckMin - datesToCheckMax
        bongo = False
        #dates1 is maxIndex: 53 minIndex: 62
        #create halfLength2 = dates2Max - dates2Min
        #check if maxIndex > dates2Max + length2 & maxIndex < dates2min
        if maxIndex >= datesToCheckMax and maxIndex <= datesToCheckMax + halfLength:
            bongo = True
        #check if minIndex < dates2Min & dates2Max + length2 < minIndex
        if minIndex <= datesToCheckMin and minIndex >= datesToCheckMin - halfLength:
            bongo = True
        #check if dates1 has overlap with dates2
        return bongo
    newHisto = {}
    newDatesHisto = {}
    if len(list(histo)) > 0:
        print(len(list(histo)))
        for dateRange in histo:
            datesToCheckMax = int(dateRange.split(' ')[0])
            datesToCheckMin = int(dateRange.split(' ')[1])
            if hasOverLap(dip['maxIndex'], dip["minIndex"], datesToCheckMax, datesToCheckMin):
                # print(dateRange, f"{dip['maxIndex']} {dip['minIndex']}")
                datesHisto[f"{datesToCheckMax} {datesToCheckMin}"]['max'].append(dip['maxIndex']) 
                datesHisto[f"{datesToCheckMax} {datesToCheckMin}"]['min'].append(dip['minIndex']) 
                #create new and save with old
                    #make vars first
                newMaxIndex = math.floor(np.average(np.array(datesHisto[f"{datesToCheckMax} {datesToCheckMin}"]['max'])))
                newMinIndex = math.ceil(np.average(np.array(datesHisto[f"{datesToCheckMax} {datesToCheckMin}"]['min'])))
                newDatesHisto[f'{newMaxIndex} {newMinIndex}'] = {'max': datesHisto[f"{datesToCheckMax} {datesToCheckMin}"]['max'], 'min': datesHisto[f"{datesToCheckMax} {datesToCheckMin}"]['min']}
                #del old
                # del datesHisto[f"{datesToCheckMax} {datesToCheckMin}"]
                #create new histo
                newHisto[f'{newMaxIndex} {newMinIndex}'] = histo.get(f"{datesToCheckMax} {datesToCheckMin}", 0) + 1
                # del histo[f"{datesToCheckMax} {datesToCheckMin}"]

            else:
                newHisto = dict(histo)
                newDatesHisto = dict(datesHisto)
                newHisto[f"{dip['maxIndex']} {dip['minIndex']}"] = 1 #histo.get(oldkey, 0) + 1
                newDatesHisto[f"{dip['maxIndex']} {dip['minIndex']}"] = {'max': [dip['maxIndex']], 'min': [dip['minIndex']]} #histo.get(oldkey, 0) + 1
                

            #append and then init new histo key to old histo and return histo here
    else: 
        newHisto[f"{dip['maxIndex']} {dip['minIndex']}"] = 1 #histo.get(oldkey, 0) + 1
        newDatesHisto[f"{dip['maxIndex']} {dip['minIndex']}"] = {'max': [dip['maxIndex']], 'min': [dip['minIndex']]} #histo.get(oldkey, 0) + 1
    #checks current histogram for more than a 50% overlap in dates and if none
    #creates a new range, else
    #increments count for current range and changes the key name to be the new average
    #taken from the list histogram which also gets a new name and del old key


    #add in new index "maxIndex minIndex" key and set count to 1 and push to dateHistoRanges
    #returns the new histogram 
    return newHisto, newDatesHisto



for stock in negGradsForAllStocks:
    print(stock)
    if stock not in threeMonthReturn:
        threeMonthReturn[stock] = {}
        if stock in negGradSearches:
            stockSearches = negGradSearches[stock]
    for dip in negGradsForAllStocks[stock].values():
        # print(dateHistogram)
        dateHistogram, dateHistogramRanges = datesOverlap(dateHistogram, dateHistogramRanges, dip, stock)
        # print(dateHistogram)
        countDips += 1
        # print(dip)
        if dip['minIndex']+60 < len(dates):
            returnPerc = df_sp_prices[stock][dip['minIndex']+howManyDaysLater] / dip['min']
            if math.isnan(returnPerc):
                naanIndex.append([stock, dip['minIndex']])
                continue
            monthsLaterPercentages.append(returnPerc)
            threeMonthReturn[stock][dip['minIndex']] = returnPerc
            returnPercentages[f'{str(returnPerc)[:3]}'] = returnPercentages.get(f'{str(returnPerc)[:3]}', 0.00) + 1 
            

            if (stock in negGradSearches) and isinstance(stockSearches[dates[dip['minIndex']]], float) and (not math.isnan(stockSearches[dates[dip['minIndex']]])) and not (math.isinf(stockSearches[dates[dip['minIndex']]])) and (stockSearches[dates[dip['minIndex']]] != 0):
                coolObj.append([returnPerc, stockSearches[dates[dip['minIndex']]]])
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
                coolObj.append([returnPerc, stockSearches[dates[dip['minIndex']]]])

#order the objects
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

#if dates have more than a 50% overlap with dates in average max-min histogram, increment count in histograms where key is the average date range(start and finish) which is stored in a separate  dictionary
#key keys showing all the start and end days that overlap with the average and then become part of the average
#%% PROCESS ANY UNWANTED DATA IE REMOVING LOW SIGNIFICANCE RETURNS
#Remove outlier search values > 5
#Remove any return multipliers > 3
coolObj = [[x[0], x[1]] for x in coolObj if x[1] < 5 and x[0] < 3 and x[1] > 0.5 and x[0] > 0.6]

print(f'datasize is {len(coolObj)} pairs longs')

#MAYBE MAKE A LOG TRANSFORMATION ON SEARCH INDEX OR A SQUAREROOT TRANSFORMATION PERHAPS
#%%Make scatter plot of multiplier vs searches dataframe
relationshipDF = pd.DataFrame(np.array(coolObj), columns=['multiplier', 'normSearch'])

plt.xlabel('Normalized searches')
# plt.xlim(0, 4)
# plt.ylim(0, 3)
plt.ylabel('Return Mulitplier')
plt.scatter(relationshipDF.normSearch, relationshipDF.multiplier)
#%% ONE SAMPLE T-TESTING TO FIND P-VALUE
#need to do indepedance, Equality of variance and normality of data / residuals
#make a scatter plot for each return and the associated search value and a line of best fit
#could use an object or a dataframe

relationshipDF.describe()
# relationshipDF.head(10)

#%%
min(relationshipDF['normSearch'])
#%%
len(df_sp_searches.columns)
#%%
countSearches
#%%
returnSearches
#%%
returnPercentages

#%%
returnSearchesAverages
#if the searches are below 

#%%
dateHistogramRanges
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

# %%
