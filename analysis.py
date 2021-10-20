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
from plotly.subplots import make_subplots
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

def sp500AvgPrice(werkbins, dollares):
    james = dict()
    for werk in werkbins:
        avgPrice = np.mean(np.array(dollares[int(werk.split(' ')[0]): int(werk.split(' ')[1])]) )
        james[werk] = avgPrice
    return james

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

#%%
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
# print(f'The total number of negative gradient segments in the last 5 years\n of every stock in the S&P500 is: {sum} \nWith an analysis range of {xRange} weeks.')
allStocksWeekBinsCount = {x:y[0] for x,y in weekBins.items()}

spWkBins = initializeWeekBins(dates)
negGradsForSP500, spWkBins = findLocationsOfValueDrop(df_sp_price['price'], analysisRange, dec, 'S&P500', howManyDaysLater, spWkBins)
spAvgPricePerWkBin = sp500AvgPrice(spWkBins, df_sp_price['price'])

print(f'finished part 1 for analysis range: {xRange} weeks')
#%% PART 2: CREATE HISTOGRAM OF RETURN PERCENTAGE AND AVERAGE INTERCONNECTEDNESS

#do it using week bins
returnsHistogramLists = dict()

for welk in  weekBins.values():
    for k,v in welk[1].items():
        grade = np.around(v, 1)
        if grade in returnsHistogramLists:
            returnsHistogramLists[grade].append(welk[0])
        else:
            returnsHistogramLists[grade] = [welk[0]]
returnsHistogramLists = {k: v for k, v in sorted(returnsHistogramLists.items(), key=lambda item: item[0]) if k > 0.6 and k < 2}
returnsHistogramAverages = {k:np.mean(np.array(v)) for k,v in returnsHistogramLists.items()}
#%%
returnsHistogramAverages
#%% PART 3: CREATE THE INVERSE OF THE ABOVE SO PERCENTILES FOR AN INTERCONNECTEDNESS AND PERCENTILES OF THE RETURN VALUE LIST
interconnectednessHistogram = dict()
for wek in weekBins.values():
    interCon = wek[0]
    if interCon in interconnectednessHistogram:
        interconnectednessHistogram[interCon]['list'] += (list(wek[1].values())) 
        listItems = np.array(interconnectednessHistogram[interCon]['list'])
        interconnectednessHistogram[interCon]['listLength'] = len(listItems)
        interconnectednessHistogram[interCon]['mean'] = np.mean(np.array(listItems))
        interconnectednessHistogram[interCon]['median'] = np.median(np.array(listItems))
        interconnectednessHistogram[interCon]['LQ'] = np.quantile(np.array(listItems), 0.25)
        interconnectednessHistogram[interCon]['UQ'] = np.quantile(np.array(listItems), 0.75)
        interconnectednessHistogram[interCon]['5th'] = np.quantile(np.array(listItems), 0.05)
    else:
        listItems = list(wek[1].values())
        if len(listItems) > 0:
            npyarray = np.array(listItems)
            interconnectednessHistogram[interCon] = {
                'list': listItems,
                'listLength': len(listItems), 
                'mean': np.mean(npyarray),
                'median': np.median(npyarray),
                'LQ': np.quantile(npyarray, 0.25),
                'UQ': np.quantile(npyarray, 0.75),
                '5th': np.quantile(npyarray, 0.05)
            }
            
#MAYBE MAKE SEPARATE ARRAYS AND PROCESS USING DICTIONARY COMPREHENSION
interconnectednessHistogramSmooth = dict()
for key,val in interconnectednessHistogram.items():
    if key < 100:
        grade = int(str(key)[:1] + '5')
        if key < 10:
            if 5 not in interconnectednessHistogramSmooth:
                interconnectednessHistogramSmooth[5] = val['list']
            else: 
                interconnectednessHistogramSmooth[5] += val['list']
        else:
            if grade not in interconnectednessHistogramSmooth:
                interconnectednessHistogramSmooth[grade] = val['list']
            else:
                interconnectednessHistogramSmooth[grade] += val['list']
    else : 
        grade = int(str(key)[:1] + '05')
        if grade not in interconnectednessHistogramSmooth:
            interconnectednessHistogramSmooth[grade] = val['list']
        else:
            interconnectednessHistogramSmooth[grade] += val['list']
interconnectednessHistogramSmoothMetrics = {k:{'mean': np.mean(np.array(v)), 'median': np.median(np.array(v)), 'UQ': np.quantile(np.array(v), 0.75), 'LQ': np.quantile(np.array(v), 0.25), '5th': np.quantile(np.array(v), 0.05)} for k,v in interconnectednessHistogramSmooth.items()}


interconnectednessHistogram = {k: v for k, v in sorted(interconnectednessHistogram.items(), key=lambda item: item[0])}
interconnectednessHistogramSmoothMetrics = {k: v for k, v in sorted(interconnectednessHistogramSmoothMetrics.items(), key=lambda item: item[0])}

#%%
interconnectednessHistogramSmoothMetrics
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

#%% PLOT - RETURN BINS AGAINST AVERAGE INTERCONNECTEDNESS FOR THAT BIN 
pineapple = [[x, w] for x,w in returnsHistogramAverages.items()]

pineapple_df = pd.DataFrame(pineapple, columns=['Return Multiplier', 'Interconnectedness'])
pineapple_df.describe()
fig = px.scatter(pineapple_df, x="Interconnectedness", y="Return Multiplier", color="Return Multiplier")
fig.show()

# p = np.poly1d(np.polyfit(eggies_df['Interconnectedness'], eggies_df['Return Multiplier'], 1))
# regr_results = sp.stats.linregress(eggies_df['Interconnectedness'], eggies_df['Return Multiplier'])
#%% PLOT - TOTAL DATA PLOT WITH STATISTICAL ANALYSIS 
eggies = [[x, w] for x,w in {k:v for k,v in multiplierNumWeeksAverage.items() if k > 0.6 and k < 2}.items()]

eggies_df = pd.DataFrame(eggies, columns=['Return Multiplier', 'Interconnectedness'])
eggies_df.describe()
fig = px.scatter(eggies_df, x="Interconnectedness", y="Return Multiplier", color="Return Multiplier")
fig.show()

p = np.poly1d(np.polyfit(eggies_df['Interconnectedness'], eggies_df['Return Multiplier'], 1))
regr_results = sp.stats.linregress(eggies_df['Interconnectedness'], eggies_df['Return Multiplier'])

#%%PLOT - RETURN MEAN MEDIAN QUARTILES ETC VS WEEKBINS averaged into 2 week periods 
#MAYBE PLOT BOX AND WHISKER TOO OVEERTOP OR EVEN HAVE ALL VALUES PLOTTED
cheese = [[x,v['mean']] for x,v in interconnectednessHistogramSmoothMetrics.items()]
gouda = [[x,v['LQ']] for x,v in interconnectednessHistogramSmoothMetrics.items()]
swiss = [[x,v['UQ']] for x,v in interconnectednessHistogramSmoothMetrics.items()]
cheddar = [[x,v['5th']] for x,v in interconnectednessHistogramSmoothMetrics.items()]
edam = [[x,v['median']] for x,v in interconnectednessHistogramSmoothMetrics.items()]
ham = [[x, w] for x,w in {k:v for k,v in multiplierNumWeeksAverage.items() if k > 0.6 and k < 2}.items()]

ham_df = pd.DataFrame(ham, columns=['Return Multiplier', 'Interconnectedness'])
cheese_df = pd.DataFrame(cheese, columns=['Interconnectedness', 'Return Multiplier'])
gouda_df = pd.DataFrame(gouda, columns=['Interconnectedness', 'Return Multiplier'])
swiss_df = pd.DataFrame(swiss, columns=['Interconnectedness', 'Return Multiplier'])
cheddar_df = pd.DataFrame(cheddar, columns=['Interconnectedness', 'Return Multiplier'])
edam_df = pd.DataFrame(edam, columns=['Interconnectedness', 'Return Multiplier'])

f1 = px.line(cheese_df, x="Interconnectedness", y="Return Multiplier")
f2 = px.line(gouda_df, x="Interconnectedness", y="Return Multiplier")
f3 = px.line(swiss_df, x="Interconnectedness", y="Return Multiplier")
f4 = px.line(cheddar_df, x="Interconnectedness", y="Return Multiplier")
f5 = px.line(edam_df, x="Interconnectedness", y="Return Multiplier")
f6 = px.scatter(ham_df, x="Interconnectedness", y="Return Multiplier", color="Return Multiplier")

# fig.show()

ching = make_subplots()
ching.add_traces(f1.data + f2.data + f3.data + f4.data + f5.data+ f6.data)

ching.show()


#%% PLOT - HISTOGRAM OF EVENTS FOR WEEKS AGAINST S&P500 STOCK PRICE FOR FIVE YEARS
bolognese = [[int(x.split(' ')[0]), w] for x,w in allStocksWeekBinsCount.items()]
lettuce = [[int(x.split(' ')[0]), y] for x,y in spAvgPricePerWkBin.items()]
bolognese_df = pd.DataFrame(bolognese, columns=['Week Bin', 'Interconnectedness'])
lettuce_df = pd.DataFrame(lettuce, columns=['Week Bin', 'Price'])
subfig = make_subplots(specs=[[{"secondary_y": True}]])
fig = px.line(lettuce_df, y='Price', x='Week Bin')
fig2 = px.bar(bolognese_df, y='Interconnectedness', x='Week Bin', color="Interconnectedness")
# fig.show()

fig2.update_traces(yaxis="y2")

subfig.add_traces(fig.data + fig2.data)
subfig.layout.xaxis.title="Weeks in the last 5yrs"
subfig.layout.yaxis.title="S&P500 Price $$$"
subfig.layout.yaxis2.title="Interconnectedness"
# recoloring is necessary otherwise lines from fig und fig2 would share each color
# e.g. Linear-, Log- = blue; Linear+, Log+ = red... we don't want this

subfig.show()

# %%
weekBins
# %%
