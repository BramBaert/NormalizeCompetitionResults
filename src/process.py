import pandas as pd
import plotly.express as px
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from distfit import distfit

C_USE_SHOT_COUNT_NORMALIZED = 1
C_NORMALIZED_SHOT_COUNT = 30
C_USE_UNISEX = 1

def calculate_result_reversed(row):
    ''' This function reverses the results starting from the maximum result as 0
        Input:
            row: a dataFrame row where the 'Result' holds a list of all results
        '''
    max_value = max(row[resultKey])

    return [abs(value - max_value) for value in row[resultKey]]

def normalize_shots(row,targetShots=60):
    ''' This function normalizes the result to 60 shots based on the normal amount of shots
        for the respective category
        Input:
            row:            a dataFrame row where the 'Result' holds the result of a single 
                            shooter/competition
            targetShots:    The number of shots to normalize to (default=60)
        Returns: 
            The 'Results' column normalized to <targetShots>
    '''
    match row['Category']:
        case "DUV":
            return row['Result']*targetShots/30
        case "BEN":
            return row['Result']*targetShots/30
        case "CAD":
            return row['Result']*targetShots/40
        case _:
            return row['Result']*targetShots/60

def fit_distribution_old(categoryToFit):
    ''' This function uses distfit to try and find the distribution that best maps to the data
        Input:
            categoryToFit: A string describing the category from dfStats to use as data input
    '''
    X = np.array(df.loc[df['Category'] == categoryToFit][resultKey],dtype='float32')
    #X=np.array(dfStats.loc[dfStats['Category'] == categoryToFit][resultKey].values[0],dtype='float32')
    #print(X)

    # Initialize distfit
    dist = distfit()

    # Determine best-fitting probability distribution for data
    dist.fit_transform(X)
    #print("Type of dist.summary = {}".format(type(dist.summary)))
    print(dist.summary)
    print(dist.histdata)
    dfHist = pd.DataFrame(np.column_stack(dist.histdata),columns=['DistFitFrequency','DistFitBin'])
    fig = px.line(dfHist, x='DistFitBin',y='DistFitFrequency')
    fig.add_trace(px.histogram(df.loc[df['Category'] == categoryToFit],x=resultKey, histnorm='probability density', nbins=len(dfHist),opacity=0.3,color_discrete_sequence=['green']).data[0])
    fig.show()

    fig = px.line(dist.summary, x='name', y='score')
    fig.show()

    dist.plot(n_top=5)

    fig, ax = dist.plot(chart='pdf', n_top=7)
    fig, ax = dist.plot(chart='cdf', n_top=7, ax=ax)

    #for S1 beta(a=3.12755,b=0.931507,loc=436.994,scale=174.306)
    #print(dist.summary.columns.values)
    # -> ['name' 'score' 'loc' 'scale' 'arg' 'params' 'model' 'bootstrap_score' 'bootstrap_pass' 'color']

    print(dist.summary.loc[dist.summary['name'] == "loggamma"]['arg'].values[0])
    print(dist.summary.loc[dist.summary['name'] == "loggamma"]['params'].values[0])
    print(scipy.stats.loggamma.fit(df.loc[df['Category'] == categoryToFit][resultKey]))
    print(type(scipy.stats.loggamma.fit(df.loc[df['Category'] == categoryToFit][resultKey])))
    print(len(df.loc[df['Category'] == categoryToFit][resultKey]))

def fit_distribution(categoryToFit,dfFit:pd.DataFrame,dfHist:pd.DataFrame):
    ''' This function uses distfit to try and find the distribution that best maps to the data
        Input:
            categoryToFit   : A string describing the category from dfStats to use as data input
            dfFit           : DataFrame (empty or the result of a previous fit_distribution)
            dfHist          : DataFrame (empty or the result of a previous fit_distribution)
        Return:
            dfFit           : The input DataFrame appended with the distfit.summary dataFrame
                              for the respective category with an additional column 'Category' 
            dfHist          : The input DataFrame appended with the histogram data for the 
                              respective category with an additional column 'Category'
    '''
    X = np.array(df.loc[df['Category'] == categoryToFit][resultKey],dtype='float32')
    #X=np.array(dfStats.loc[dfStats['Category'] == categoryToFit][resultKey].values[0],dtype='float32')
    #print(X)

    # Initialize distfit
    dist = distfit()

    # Determine best-fitting probability distribution for data
    dist.fit_transform(X,verbose='warning')

    dfLocalFit= dist.summary.copy(deep=True)
    dfLocalFit['Category'] = categoryToFit

    dfLocalHist = pd.DataFrame(np.column_stack(dist.histdata),columns=['DistFitFrequency','DistFitBin'])
    dfLocalHist['Category'] = categoryToFit

    dfFit =  pd.concat([dfFit, dfLocalFit], ignore_index=True)
    dfHist =  pd.concat([dfHist, dfLocalHist], ignore_index=True)

    return dfFit,dfHist

def fitBetaDist(row):
    ''' This function returns the parameters for a beta distribution
        Input:
            row:    A row from a dataFrame consisting of at least two columns 'Result'
                    and 'Result60Shots'. The global variable resultKey will be used to
                    determine which of the two to select
        Return:
            A tuple consisting of the parameter for a beta distribution (alpha, beta loc, scale)
    '''
    return(scipy.stats.beta.fit(row[resultKey]))

def fitLogGammaDist(row):
    ''' This function returns the parameters for a logGamma distribution
        Input:
            row:    A row from a dataFrame consisting of at least two columns 'Result'
                    and 'Result60Shots'. The global variable resultKey will be used to
                    determine which of the two to select
        Return:
            A tuple consisting of the parameter for a beta distribution (c, loc, scale)
    '''
    return(scipy.stats.loggamma.fit(row[resultKey]))

def fitGenExtremeDist(row):
    ''' This function returns the parameters for a Generalized extreme value distribution
        Input:
            row:    A row from a dataFrame consisting of at least two columns 'Result'
                    and 'Result60Shots'. The global variable resultKey will be used to
                    determine which of the two to select
        Return:
            A tuple consisting of the parameter for a beta distribution (c, loc, scale)
    '''
    return(scipy.stats.genextreme.fit(row[resultKey]))

def determineCorrection(row,refCat):
    ''' This function returns the Correction factor between categories
        Input:
            row:    A row from a dataFrame consisting of at least two columns 'Result'
                    and 'Result60Shots'. The global variable resultKey will be used to
                    determine which of the two to select
            refCat: The reference category to use
        Return:
            A list of the correction factors per point for the respective category
    '''
    return((dfStats.loc[dfStats['Category'] == refCat]['LogGammaLinSpace'].values[0])/row['LogGammaLinSpace'])

def unisex(val):
    ''' This function converts the category label to a uni-sex label
    Input:
        value: The label to convert
    Return:
        The uni-sex label
    '''

    match val:
        case "D1":
            return "U1"
        case "S1":
            return "U1"
        case "D2":
            return "U2"
        case "S2":
            return "U2"
        case "D3":
            return "U3"
        case "S3":
            return "U3"
        case "JD":
            return "JUN"
        case "JH":
            return "JUN"
        case _:
            return val
    
def appendStatsFunction(dataFrame:pd.DataFrame,function,name:str="algoName") -> pd.DataFrame:
    localDf=dataFrame.copy(deep=True)
    localDf['linSpace'] = localDf[resultKey]
    localDf['algoVal']  = localDf[resultKey].apply(function)
    localDf['algoName'] = name
    if(np.array == type(localDf['algoVal'])):
        localDf = localDf.explode(['linSpace','algoVal']).reset_index(drop=True)
    else:
        localDf = localDf.explode('linSpace').reset_index(drop=True)
    return localDf

def appendDistFunction(dataFrame:pd.DataFrame,linFunct,AlgoFunc,name:str="algoName") -> pd.DataFrame:
    localDf=dataFrame.copy(deep=True)
    localDf['linSpace'] = dfStats.apply(lambda row: linFunct,axis=1)
    localDf['algoVal']  = dfStats.apply(lambda row: AlgoFunc,axis=1)
    localDf['algoName'] = name
    print("Creating an algoVal df with type:{}".format(type(localDf['algoVal'])))
    if(np.array == type(localDf['algoVal'])):
        localDf = localDf.explode(['linSpace','algoVal']).reset_index(drop=True)
    else:
        localDf = localDf.explode('linSpace').reset_index(drop=True)
    return localDf

def printLineHistogram(algoData:pd.DataFrame,
                       lines:list,
                       title:str="",
                       yaxis_title:str="",
                       histnorm:str='probability density'):
    # Strip the df to the lines we want to print
    localDf = algoData.loc[algoData["algoName"].isin(lines)]
    # Create subplots using Plotly Express
    fig = px.line(localDf, x='linSpace', y='algoVal', color="algoName", facet_col='Category', facet_col_wrap=3,color_discrete_sequence=['red','green','blue'])

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Points',
        yaxis_title=yaxis_title
    )

    #fig.print_grid()

    dfStats_exploded_result = dfStats.explode([resultKey]).reset_index(drop=True)
    categoryCount = len(dfStats['Category'].unique())
    for count, cat in enumerate(dfStats['Category'].unique()):
        #rowVal = int((categoryCount-count+3)/3)
        rowVal = int(categoryCount/3) - int(count/3)
        if categoryCount%3:
            rowVal += 1
        colVal = int(count%3)+1
        #print("Count {} = Cat {}: row = {} , col = {}".format(count,cat,rowVal,colVal))
        fig.add_trace(px.histogram(dfStats_exploded_result.loc[dfStats_exploded_result['Category'] == cat],x=resultKey, histnorm=histnorm, nbins=50,opacity=0.5).data[0],row=rowVal,col=colVal)
    # Show the plot
    fig.data = fig.data[::-1]
    fig.show()

# Import the data
df = pd.read_csv('data.csv', encoding='iso-8859-1')

# Make the categories uni-sex
if C_USE_UNISEX:
    df['Category'] = df['Category'].apply(unisex)

# Normalize all shots to the target number of competition shots TODO make the targetShots a commandline argument
df['ResultNormalizedShots'] = df.apply(lambda row: normalize_shots(row,C_NORMALIZED_SHOT_COUNT), axis=1)

# TODO make this depend on a commandline argument 
if C_USE_SHOT_COUNT_NORMALIZED: 
    resultKey = 'ResultNormalizedShots'
else:
    resultKey = 'Result'

#fit_distribution_old("JH")
#plt.show()

# Run distribution fitting over all categories
dfFit = pd.DataFrame()
dfHist = pd.DataFrame()
for cat in df['Category'].unique():
    dfFit,dfHist = fit_distribution(cat,dfFit,dfHist)

# Calculate the overall best algorithm
sum_scores = dfFit.groupby('name')['score'].sum().reset_index()
sum_scores['Category'] = 'Total'
sum_scores.sort_values(['score'],inplace=True)
algoSorterList = sum_scores['name'].tolist()

print("The best distribution is {}".format(algoSorterList[0]))
if "loggamma" != algoSorterList[0]:
    print("Warning: the used distibution algorithm for normalizing the results, isn't the most optimal one.")

# Concatenate sum_scores to the original DataFrame
dfFit = pd.concat([dfFit, sum_scores], ignore_index=True)

# Make a plot showing the score of the different distirbution algorithms
dfFit.sort_values(['name'],key=lambda column:column.map(lambda e: algoSorterList.index(e)), inplace=True)
fig = px.line(dfFit,x='name',y='score',color='Category')
fig.update_traces(showlegend = True)
fig.show()

# Start creation of the per Category statistics DataFrame
dfStats = df.groupby(['Category', 'Discipline']).agg({'Result': list, 'ResultNormalizedShots': list}).reset_index()
# Add the different statistics info
dfStats['Count']                = dfStats[resultKey].apply(np.count_nonzero)
dfStats['BetaParams']           = dfStats.apply(fitBetaDist, axis=1)
dfStats['LogGammaParams']       = dfStats.apply(fitLogGammaDist, axis=1)
dfStats['GenExtremeParams']     = dfStats.apply(fitGenExtremeDist, axis=1)

# Add the different statistical parameters
dfAlgo = appendStatsFunction(dfStats,np.mean,"Mean")
dfAlgo = pd.concat([dfAlgo,appendStatsFunction(dfStats,np.median,"Median")], ignore_index=True)
dfAlgo = pd.concat([dfAlgo,appendStatsFunction(dfStats,np.max,"Max")], ignore_index=True)
dfAlgo = pd.concat([dfAlgo,appendStatsFunction(dfStats,np.min,"Min")], ignore_index=True)

# Add the Beta function
dfSingleDist=dfStats.copy(deep=True)
dfSingleDist['linSpace'] = dfStats.apply(lambda row: np.linspace(scipy.stats.beta.ppf(0.01,
                                                                                a=row['BetaParams'][0],
                                                                                b=row['BetaParams'][1],
                                                                                loc=row['BetaParams'][2],
                                                                                scale=row['BetaParams'][3]), 
                                                            scipy.stats.beta.ppf(0.99,
                                                                                a=row['BetaParams'][0],
                                                                                b=row['BetaParams'][1],
                                                                                loc=row['BetaParams'][2],
                                                                                scale=row['BetaParams'][3]),
                                                            1000),axis=1)
dfStats['BetaLinSpace'] = dfSingleDist['linSpace']
dfSingleDist['algoVal'] = dfSingleDist.apply(lambda row: [min(0.035,x) for x in scipy.stats.beta.pdf(row['linSpace'],
                                                                                            a=row['BetaParams'][0],
                                                                                            b=row['BetaParams'][1],
                                                                                            loc=row['BetaParams'][2],
                                                                                            scale=row['BetaParams'][3])],
                                    axis=1)
dfSingleDist['algoName'] = "betaPdf"
betaPdfDf           = dfSingleDist.explode(['linSpace','algoVal']).reset_index(drop=True)
dfAlgo              = pd.concat([dfAlgo,betaPdfDf], ignore_index=True)

dfSingleDist['algoVal']  = dfSingleDist.apply(lambda row: scipy.stats.beta.cdf(row['linSpace'],
                                                                     a=row['BetaParams'][0],
                                                                     b=row['BetaParams'][1],
                                                                     loc=row['BetaParams'][2],
                                                                     scale=row['BetaParams'][3]),
                                    axis=1)
dfSingleDist['algoName'] = "betaCdf"
betaCdfDf = dfSingleDist.explode(['linSpace','algoVal']).reset_index(drop=True)
dfAlgo = pd.concat([dfAlgo,betaCdfDf], ignore_index=True)

# Add the log gamma function
dfSingleDist['linSpace']    = dfStats.apply(lambda row: np.linspace(scipy.stats.loggamma.ppf(0.01,
                                                                                    c=row['LogGammaParams'][0],
                                                                                    loc=row['LogGammaParams'][1],
                                                                                    scale=row['LogGammaParams'][2]), 
                                                            scipy.stats.loggamma.ppf(0.99,
                                                                                    c=row['LogGammaParams'][0],
                                                                                    loc=row['LogGammaParams'][1],
                                                                                    scale=row['LogGammaParams'][2]),
                                                            1000),
                                    axis=1)
dfStats['LogGammaLinSpace'] = dfSingleDist['linSpace']
dfSingleDist['algoVal']     = dfSingleDist.apply(lambda row: [min(0.035,x) for x in scipy.stats.loggamma.pdf(row['linSpace'],
                                                                                                c=row['LogGammaParams'][0],
                                                                                                loc=row['LogGammaParams'][1],
                                                                                                scale=row['LogGammaParams'][2])],
                                    axis=1)
dfSingleDist['algoName']    = "logGammaPdf"
logGammaPdfDf               = dfSingleDist.explode(['linSpace','algoVal']).reset_index(drop=True)
dfAlgo                      = pd.concat([dfAlgo,logGammaPdfDf], ignore_index=True)

dfSingleDist['algoVal']     = dfSingleDist.apply(lambda row: scipy.stats.loggamma.cdf(row['linSpace'],
                                                                            c=row['LogGammaParams'][0],
                                                                            loc=row['LogGammaParams'][1],
                                                                            scale=row['LogGammaParams'][2]), axis=1)
dfSingleDist['algoName']    = "logGammaCdf"
logGammaCdfDf               = dfSingleDist.explode(['linSpace','algoVal']).reset_index(drop=True)
dfAlgo                      = pd.concat([dfAlgo,logGammaCdfDf], ignore_index=True)

# Add the Generalized extreme distibution function
dfSingleDist['linSpace']        = dfStats.apply(lambda row: np.linspace(scipy.stats.genextreme.ppf(0.01,
                                                                                    c=row['GenExtremeParams'][0],
                                                                                    loc=row['GenExtremeParams'][1],
                                                                                    scale=row['GenExtremeParams'][2]), 
                                                            scipy.stats.genextreme.ppf(0.99,
                                                                                    c=row['GenExtremeParams'][0],
                                                                                    loc=row['GenExtremeParams'][1],
                                                                                    scale=row['GenExtremeParams'][2]),
                                                            1000),
                                    axis=1)
dfStats['GenExtremeLinSpace']   = dfSingleDist['linSpace']
dfSingleDist['algoVal']         = dfSingleDist.apply(lambda row: [min(0.035,x) for x in scipy.stats.genextreme.pdf(row['linSpace'],
                                                                                                c=row['GenExtremeParams'][0],
                                                                                                loc=row['GenExtremeParams'][1],
                                                                                                scale=row['GenExtremeParams'][2])],
                                    axis=1)
dfSingleDist['algoName']        = "GenExtremePdf"
genExtremePdfDf                 = dfSingleDist.explode(['linSpace','algoVal']).reset_index(drop=True)
dfAlgo                          = pd.concat([dfAlgo,genExtremePdfDf], ignore_index=True)

dfSingleDist['algoVal']         = dfSingleDist.apply(lambda row: scipy.stats.genextreme.cdf(row['linSpace'],
                                                                            c=row['GenExtremeParams'][0],
                                                                            loc=row['GenExtremeParams'][1],
                                                                            scale=row['GenExtremeParams'][2]), axis=1)
dfSingleDist['algoName']        = "GenExtremeCdf"
genExtremeCdfDf                 = dfSingleDist.explode(['linSpace','algoVal']).reset_index(drop=True)
dfAlgo                          = pd.concat([dfAlgo,genExtremeCdfDf], ignore_index=True)

printLineHistogram(dfAlgo,['betaCdf','logGammaCdf','GenExtremeCdf'],title="Cumulative Density Function",histnorm='probability')
printLineHistogram(dfAlgo,['betaPdf','logGammaPdf','GenExtremePdf'],title="Probability Density Function")

if(1 == C_USE_UNISEX):
    refCat = "U1"
else:
    refCat = "S1"
dfCorrection['Correction']           = dfStats.apply(lambda row: determineCorrection(row,refCat,),axis=1)

exit()

#dfStats['Sigma']                = dfStats[resultKey].apply(np.std)
#dfStats['Skew']                 = dfStats[resultKey].apply(scipy.stats.skew)
#dfStats['SkewedNormLinspace']   = dfStats['Skew'].apply(lambda skew: np.linspace(scipy.stats.skewnorm.ppf(0.01, skew), scipy.stats.skewnorm.ppf(0.99, skew), 1000))
#dfStats['SkewedLinSpace']       = (dfStats['SkewedNormLinspace']*dfStats['Sigma']) + dfStats['Mean']
#dfStats['SkewedPdf']            = dfStats.apply(lambda row: scipy.stats.skewnorm.pdf(row['SkewedNormLinspace'], row['Skew']), axis=1)
#dfStats['SkewedCdf']            = dfStats.apply(lambda row: scipy.stats.skewnorm.cdf(row['SkewedNormLinspace'], row['Skew']), axis=1)
#dfStats['BetaParams']           = dfStats.apply(fitBetaDist, axis=1)
'''
dfStats['BetaLinSpace']         = dfStats.apply(lambda row: np.linspace(scipy.stats.beta.ppf(0.01,
                                                                                a=row['BetaParams'][0],
                                                                                b=row['BetaParams'][1],
                                                                                loc=row['BetaParams'][2],
                                                                                scale=row['BetaParams'][3]), 
                                                                        scipy.stats.beta.ppf(0.99,
                                                                                a=row['BetaParams'][0],
                                                                                b=row['BetaParams'][1],
                                                                                loc=row['BetaParams'][2],
                                                                                scale=row['BetaParams'][3]),
                                                                        1000),axis=1)
dfStats['BetaPdf']              = dfStats.apply(lambda row: [min(0.025,x) for x in scipy.stats.beta.pdf(row['BetaLinSpace'],
                                                                                a=row['BetaParams'][0],
                                                                                b=row['BetaParams'][1],
                                                                                loc=row['BetaParams'][2],
                                                                                scale=row['BetaParams'][3])], axis=1)
dfStats['BetaCdf']              = dfStats.apply(lambda row: scipy.stats.beta.cdf(row['BetaLinSpace'],
                                                                                a=row['BetaParams'][0],
                                                                                b=row['BetaParams'][1],
                                                                                loc=row['BetaParams'][2],
                                                                                scale=row['BetaParams'][3]), axis=1)
                                                                                '''
#dfStats['LogGammaParams']       = dfStats.apply(fitLogGammaDist, axis=1)
'''dfStats['LogGammaLinSpace']     = dfStats.apply(lambda row: np.linspace(scipy.stats.loggamma.ppf(0.01,
                                                                                c=row['LogGammaParams'][0],
                                                                                loc=row['LogGammaParams'][1],
                                                                                scale=row['LogGammaParams'][2]), 
                                                                        scipy.stats.loggamma.ppf(0.99,
                                                                                c=row['LogGammaParams'][0],
                                                                                loc=row['LogGammaParams'][1],
                                                                                scale=row['LogGammaParams'][2]),
                                                                                1000),axis=1)
dfStats['LogGammaPdf']          = dfStats.apply(lambda row: [min(0.025,x) for x in scipy.stats.loggamma.pdf(row['LogGammaLinSpace'],
                                                                                c=row['LogGammaParams'][0],
                                                                                loc=row['LogGammaParams'][1],
                                                                                scale=row['LogGammaParams'][2])], axis=1)
dfStats['LogGammaCdf']          = dfStats.apply(lambda row: scipy.stats.loggamma.cdf(row['LogGammaLinSpace'],
                                                                                c=row['LogGammaParams'][0],
                                                                                loc=row['LogGammaParams'][1],
                                                                                scale=row['LogGammaParams'][2]), axis=1)'''
dfStats['Correction']           = dfStats.apply(lambda row: determineCorrection(row,"S1"),axis=1)

'''
for cat in dfStats['Category'].unique():
    dfStatsCat = dfStats.loc[dfStats['Category'] == cat]
    print("{:3s}: Mean {:.3f}, Max {:.1f}, Count {:.3f}".format(cat,
                                                                dfStatsCat['Mean'].values[0],
                                                                dfStatsCat['Max'].values[0],
                                                                dfStatsCat['Count'].values[0]))
'''

#print("S1 has {} datapoints.".format(len(dfStats[dfStats['Category'] == "S1"][resultKey].values[0])))


#dfStats_exploded = dfStats.explode(['ResultReversed']).reset_index(drop=True)
#fig = px.line(dfStats_exploded, y='ResultReversed', facet_col='Category', facet_col_wrap=3,color_discrete_sequence=['red'])
#fig.show()

# Explode the lists in 'SkewedLinSpace' and 'SkewedPdf' into separate rows
dfStats_exploded = dfStats.explode(['LogGammaLinSpace','BetaLinSpace', 'BetaCdf','LogGammaCdf']).reset_index(drop=True)

# Create subplots using Plotly Express
fig = px.line(dfStats_exploded, x='BetaLinSpace', y='BetaCdf', facet_col='Category', facet_col_wrap=3,color_discrete_sequence=['red','green','blue'])

# Update layout
fig.update_layout(
    title='CDFs for Different Categories',
    xaxis_title='Points',
    yaxis_title='CDF'
)

dfStats_exploded_result = dfStats.explode([resultKey]).reset_index(drop=True)
categoryCount = len(dfStats['Category'].unique())
for count, cat in enumerate(dfStats['Category'].unique()):
    rowVal = int((categoryCount-count+4)/3)
    colVal = int(count%3)+1
    fig.add_trace(px.histogram(dfStats_exploded_result.loc[dfStats_exploded_result['Category'] == cat],x=resultKey, histnorm='probability', nbins=50,opacity=0.5).data[0],row=rowVal,col=colVal)
    fig.add_trace(px.line(dfStats_exploded.loc[dfStats_exploded['Category'] == cat], x='LogGammaLinSpace', y='LogGammaCdf', facet_col='Category', facet_col_wrap=3,color_discrete_sequence=['blue','red','green']).data[0],row=rowVal,col=colVal)

# Show the plot
fig.data = fig.data[::-1]
fig.show()

# Explode the lists in 'SkewedLinSpace' and 'SkewedPdf' into separate rows
dfStats_exploded = dfStats.explode(['BetaLinSpace','LogGammaLinSpace', 'BetaPdf','LogGammaPdf']).reset_index(drop=True)

# Create subplots using Plotly Express
fig = px.line(dfStats_exploded, x='LogGammaLinSpace', y='LogGammaPdf', facet_col='Category', facet_col_wrap=3,color_discrete_sequence=['red','green','blue'])

# Update layout
fig.update_layout(
    title='PDFs for Different Categories',
    xaxis_title='Points',
    yaxis_title='PDF'
)

dfStats_exploded_result = dfStats.explode([resultKey]).reset_index(drop=True)
categoryCount = len(dfStats['Category'].unique())
for count, cat in enumerate(dfStats['Category'].unique()):
    rowVal = int((categoryCount-count+4)/3)
    colVal = int(count%3)+1
    #fig.add_trace(px.histogram(dfStats_exploded_result.loc[dfStats_exploded_result['Category'] == cat],x=resultKey, histnorm='probability', nbins=50,opacity=0.5).data[0],row=rowVal,col=colVal)
    fig.add_trace(px.histogram(df.loc[df['Category'] == cat],x=resultKey, histnorm='probability density', nbins=50,opacity=0.5).data[0],row=rowVal,col=colVal)
    fig.add_trace(px.line(dfStats_exploded.loc[dfStats_exploded['Category'] == cat], x='BetaLinSpace', y='BetaPdf', facet_col='Category', facet_col_wrap=3,color_discrete_sequence=['green','blue','red']).data[0],row=rowVal,col=colVal)
# Show the plot
fig.data = fig.data[::-1]
fig.show()


# Explode the lists in 'LogGammaLinSpace' and 'Correction' into separate rows
dfStats_exploded = dfStats.explode(['LogGammaLinSpace', 'Correction']).reset_index(drop=True)

# Create subplots using Plotly Express
fig = px.line(dfStats_exploded, x='LogGammaLinSpace', y='Correction', facet_col='Category', facet_col_wrap=3,color_discrete_sequence=['blue','red','green'],log_y=True)

# Update layout
fig.update_layout(
    title='Correction factor',
    xaxis_title='Points',
    yaxis_title='Correction factor'
)
fig.show()


plt.show()
