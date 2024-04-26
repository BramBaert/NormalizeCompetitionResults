import pandas as pd
import plotly.express as px
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from distfit import distfit

C_USE_SHOT_COUNT_NORMALIZED = 0
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

def determineCorrection(row, refCat, algo):
    ''' This function returns the Correction factor between categories
        Input:
            row:    A row from a dataFrame consisting of at least two columns 'Result'
                    and 'Result60Shots'. The global variable resultKey will be used to
                    determine which of the two to select
            refCat: The reference category to use
        Return:
            A list of the correction factors per point for the respective category
    '''
    return((dfStats.loc[dfStats['Category'] == refCat][f'{algo}LinSpace'].values[0])/row[f'{algo}LinSpace'])

def unisex(val):
    ''' This function converts the category label to a uni-sex label
    Input:
        value: The label to convert
    Return:
        The uni-sex label
    '''

    match val:
        case "D1":
            return "D1 & S1"
        case "S1":
            return "D1 & S1"
        case "D2":
            return "D2 & S2"
        case "S2":
            return "D2 & S2"
        case "D3":
            return "D3 & S3"
        case "S3":
            return "D3 & S3"
        case "JD":
            return "JUN"
        case "JH":
            return "JUN"
        case _:
            return val
    
def explodeStatsFunction(dataFrame:pd.DataFrame, linSpace:str, algo:str, name:str) -> pd.DataFrame:
    localDf=dataFrame[['Category','Discipline']].copy(deep=True)
    localDf['linSpace'] = dataFrame[linSpace]
    localDf['algoVal']  = dataFrame[algo]
    localDf['algoName'] = name
    if ((np.array == type(localDf['algoVal'][0])) or
        (np.ndarray == type(localDf['algoVal'][0])) or
        (list == type(localDf['algoVal'][0]))):
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
    fig = px.line(localDf, x='linSpace', y='algoVal', color="algoName", facet_col='Category', facet_col_wrap=3,color_discrete_sequence=['red','green','blue'],
                    labels={
                     "linSpace": "Points",
                     "algoVal": "Probability",
                     "algoName": "Distribution"
                 })

    # Update layout
    fig.update_layout(
        title=title
    )
    if(7 == len(fig.layout['annotations'])):
        fig.update_layout(
        xaxis5_title="Points",
        xaxis6_title="Points",
        xaxis5_showticklabels=True,
        xaxis6_showticklabels=True)
    elif(11 == len(fig.layout['annotations'])):
        fig.update_layout(
        xaxis6_title="Points",
        xaxis6_showticklabels=True)
    else:
        print(f"Unknown grid structure {len(fig.layout['annotations'])}")
    

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
    #fig.data = fig.data[::-1]
    fig.show()

def normalizeResult(row, resultKey:str, correctionTable:pd.DataFrame, refCat:str = "S1"):
    """ This function will return the normalized result of the value in resultKey column for the respective Discipline and Category
    Input:
        row:              A row from a dataframe containing at least the column 'Category', 'Discipline' and the provided resultKey
        resultKey:        The column to pick, the result to normalize, from
        correctionTable:  A table consisting of a column per Category as well as Discipline column. The rows are how to go from \
                          a result in one category to the 'normalized' result of another category
        refCat:           The Category string to be used to find the reference category where to normalize towards 
    Return:
        The Normalized result

    An example layout of the correctionTable
                    BEN         CAD         DUV         JUN     D1 & S1     D2 & S2     D3 & S3 Discipline
        0     146.46111  155.806244   99.546756  221.895903   232.66529  224.432236  200.215601         LK
        1    146.605823  155.951624   99.749401  221.984311  232.744738  224.509675  200.311329         LK
        2    146.750536  156.097005   99.952046   222.07272  232.824185  224.587114  200.407057         LK
        3     146.89525  156.242386  100.154691  222.161128  232.903632  224.664553  200.502785         LK
    """
    disciplineCorrectionDf = correctionTable.loc[correctionTable['Discipline'] == row['Discipline']]
    rowCorrection = disciplineCorrectionDf.iloc[(disciplineCorrectionDf[row['Category']]-row[resultKey]).abs().argsort()[:1]]
    return row[resultKey]*(rowCorrection[refCat]/rowCorrection[row['Category']]).values[0]

def normalizeResultCDF(row, resultKey:str, correctionTable:pd.DataFrame, refCat:str = "S1"):
    """ This function will return the normalized result of the value in resultKey column for the respective Discipline and Category
    Input:
        row:              A row from a DataFrame containing at least the column 'Category', 'Discipline' and the provided resultKey
        resultKey:        The column to pick, the result to normalize, from
        correctionTable:  A table consisting of a column per Category as well as Discipline column. Each row holds a tuple per \
                          Category containing the result in the first element and the CDF value for that result in the second \
                          element.
        refCat:           The Category string to be used to find the reference category where to normalize towards 
    Return:
        The Normalized result

    An example layout of the correctionTable
                    BEN   BEN_CDF         CAD   CAD_CDF     D1 & S1 D1 & S1_CDF     D2 & S2 D2 & S2_CDF     D3 & S3 D3 & S3_CDF         DUV   DUV_CDF         JUN   JUN_CDF Discipline
        0     146.46111      0.01  155.806244      0.01   232.66529        0.01  224.432236        0.01  200.215601        0.01   99.546756      0.01  221.895903      0.01         LK
        1    146.605823  0.010058  155.951624  0.010049  232.744738    0.010053  224.509675    0.010054  200.311329    0.010056   99.749401  0.010052  221.984311  0.010052         LK
        2    146.750536  0.010115  156.097005  0.010099  232.824185    0.010107  224.587114    0.010109  200.407057    0.010112   99.952046  0.010104   222.07272  0.010105         LK
        3     146.89525  0.010174  156.242386  0.010149  232.903632    0.010161  224.664553    0.010164  200.502785    0.010169  100.154691  0.010157  222.161128  0.010158         LK
        ...
    """

    disciplineCorrectionDf = correctionTable.loc[correctionTable['Discipline'] == row['Discipline']]
    fromRow = disciplineCorrectionDf.iloc[(disciplineCorrectionDf[row['Category']]-row[resultKey]).abs().argsort()[:1]]
    toRow   = disciplineCorrectionDf.iloc[(disciplineCorrectionDf[f"{refCat}_CDF"]-fromRow[f"{row['Category']}_CDF"].values[0]).abs().argsort()[:1]]
    return round(row[resultKey]*(fromRow[refCat].values[0]/toRow[row['Category']].values[0]),2)

# Import the data
df = pd.read_csv('input_data/BE_national.csv', encoding='iso-8859-1')

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
fig = px.line(dfFit,x='name',y='score',color='Category',log_y=True)
fig.update_traces(showlegend = True)
fig.update_layout(title="Fitting result for different types of distributions",
                    xaxis_title="Distribution",
                    yaxis_title="Fitting score (lower is better)",
                    legend_title="Age category")
fig.show()

# Start creation of the per Category statistics DataFrame
dfStats = df.groupby(['Category', 'Discipline']).agg({'Result': list, 'ResultNormalizedShots': list}).reset_index()

# Add the different statistics info
dfStats['Count']                = dfStats[resultKey].apply(np.count_nonzero)
dfStats['Min']                  = dfStats[resultKey].apply(np.min)
dfStats['Max']                  = dfStats[resultKey].apply(np.max)
dfStats['Mean']                 = dfStats[resultKey].apply(np.mean)
dfStats['Median']               = dfStats[resultKey].apply(np.median)

# Add the beta distibution function
dfStats['BetaParams']           = dfStats.apply(fitBetaDist, axis=1)
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
dfStats['BetaPdf']              = dfStats.apply(lambda row: [min(0.035,x) for x in scipy.stats.beta.pdf(row['BetaLinSpace'],
                                                                                a=row['BetaParams'][0],
                                                                                b=row['BetaParams'][1],
                                                                                loc=row['BetaParams'][2],
                                                                                scale=row['BetaParams'][3])], axis=1)
dfStats['BetaCdf']              = dfStats.apply(lambda row: scipy.stats.beta.cdf(row['BetaLinSpace'],
                                                                                a=row['BetaParams'][0],
                                                                                b=row['BetaParams'][1],
                                                                                loc=row['BetaParams'][2],
                                                                                scale=row['BetaParams'][3]), axis=1)

# Add the LogGamma distibution function                                                                                
dfStats['LogGammaParams']       = dfStats.apply(fitLogGammaDist, axis=1)
dfStats['LogGammaLinSpace']     = dfStats.apply(lambda row: np.linspace(scipy.stats.loggamma.ppf(0.01,
                                                                                c=row['LogGammaParams'][0],
                                                                                loc=row['LogGammaParams'][1],
                                                                                scale=row['LogGammaParams'][2]), 
                                                                        scipy.stats.loggamma.ppf(0.99,
                                                                                c=row['LogGammaParams'][0],
                                                                                loc=row['LogGammaParams'][1],
                                                                                scale=row['LogGammaParams'][2]),
                                                                                1000),axis=1)
dfStats['LogGammaPdf']          = dfStats.apply(lambda row: [min(0.035,x) for x in scipy.stats.loggamma.pdf(row['LogGammaLinSpace'],
                                                                                c=row['LogGammaParams'][0],
                                                                                loc=row['LogGammaParams'][1],
                                                                                scale=row['LogGammaParams'][2])], axis=1)
dfStats['LogGammaCdf']          = dfStats.apply(lambda row: scipy.stats.loggamma.cdf(row['LogGammaLinSpace'],
                                                                                c=row['LogGammaParams'][0],
                                                                                loc=row['LogGammaParams'][1],
                                                                                scale=row['LogGammaParams'][2]), axis=1)

# Add the Generalized extreme distibution function
dfStats['GenExtremeParams']     = dfStats.apply(fitGenExtremeDist, axis=1)
dfStats['GenExtremeLinSpace']   = dfStats.apply(lambda row: np.linspace(scipy.stats.genextreme.ppf(0.01,
                                                                                    c=row['GenExtremeParams'][0],
                                                                                    loc=row['GenExtremeParams'][1],
                                                                                    scale=row['GenExtremeParams'][2]), 
                                                            scipy.stats.genextreme.ppf(0.99,
                                                                                    c=row['GenExtremeParams'][0],
                                                                                    loc=row['GenExtremeParams'][1],
                                                                                    scale=row['GenExtremeParams'][2]),
                                                            1000),
                                    axis=1)
dfStats['GenExtremePdf']        = dfStats.apply(lambda row: [min(0.035,x) for x in scipy.stats.genextreme.pdf(row['GenExtremeLinSpace'],
                                                                                                                c=row['GenExtremeParams'][0],
                                                                                                                loc=row['GenExtremeParams'][1],
                                                                                                                scale=row['GenExtremeParams'][2])], axis=1)
dfStats['GenExtremeCdf']        = dfStats.apply(lambda row: scipy.stats.genextreme.cdf(row['GenExtremeLinSpace'],
                                                                                        c=row['GenExtremeParams'][0],
                                                                                        loc=row['GenExtremeParams'][1],
                                                                                        scale=row['GenExtremeParams'][2]), axis=1)

# Add the different statistical parameters
dfAlgo = explodeStatsFunction(dfStats,resultKey,'Mean',"Mean")
dfAlgo = pd.concat([dfAlgo,explodeStatsFunction(dfStats,resultKey,'Median',"Median")], ignore_index=True)
dfAlgo = pd.concat([dfAlgo,explodeStatsFunction(dfStats,resultKey,'Min',"Minimum")], ignore_index=True)
dfAlgo = pd.concat([dfAlgo,explodeStatsFunction(dfStats,resultKey,'Max',"Maximum")], ignore_index=True)

dfAlgo = pd.concat([dfAlgo,explodeStatsFunction(dfStats,'BetaLinSpace','BetaPdf',"betaPdf")], ignore_index=True)
dfAlgo = pd.concat([dfAlgo,explodeStatsFunction(dfStats,'BetaLinSpace','BetaCdf',"betaCdf")], ignore_index=True)
dfAlgo = pd.concat([dfAlgo,explodeStatsFunction(dfStats,'LogGammaLinSpace','LogGammaPdf',"logGammaPdf")], ignore_index=True)
dfAlgo = pd.concat([dfAlgo,explodeStatsFunction(dfStats,'LogGammaLinSpace','LogGammaCdf',"logGammaCdf")], ignore_index=True)
dfAlgo = pd.concat([dfAlgo,explodeStatsFunction(dfStats,'GenExtremeLinSpace','GenExtremePdf',"GenExtremePdf")], ignore_index=True)
dfAlgo = pd.concat([dfAlgo,explodeStatsFunction(dfStats,'GenExtremeLinSpace','GenExtremeCdf',"GenExtremeCdf")], ignore_index=True)

printLineHistogram(dfAlgo,['betaCdf','logGammaCdf','GenExtremeCdf'],title="Cumulative Density Function",histnorm='probability')
printLineHistogram(dfAlgo,['betaPdf','logGammaPdf','GenExtremePdf'],title="Probability Density Function")

if(1 == C_USE_UNISEX):
    refCat = "D1 & S1"
else:
    refCat = "S1"
dfStats['LogGammaCorrection']   = dfStats.apply(lambda row: determineCorrection(row,refCat,"LogGamma"),axis=1)
dfStats['BetaCorrection']       = dfStats.apply(lambda row: determineCorrection(row,refCat,"Beta"),axis=1)
dfStats['GenExtremeCorrection'] = dfStats.apply(lambda row: determineCorrection(row,refCat,"GenExtreme"),axis=1)

dfCorrection = explodeStatsFunction(dfStats,'LogGammaLinSpace','LogGammaCorrection',"LogGamma")
dfCorrection = pd.concat([dfCorrection,explodeStatsFunction(dfStats,'BetaLinSpace','BetaCorrection',"Beta")], ignore_index=True)
dfCorrection = pd.concat([dfCorrection,explodeStatsFunction(dfStats,'GenExtremeLinSpace','GenExtremeCorrection',"GenExtreme")], ignore_index=True)
fig = px.line(dfCorrection, x='linSpace', y='algoVal', color="algoName", facet_col='Category', facet_col_wrap=3,color_discrete_sequence=['red','green','blue'])
fig.update_layout(title="Correction factors - Linear Mapping")
fig.show()

'''
# Creating a new DataFrame with required columns
CorrectionTable = dfStats[['Discipline', 'Category', 'LogGammaLinSpace']].copy()
CorrectionTable = CorrectionTable.explode(['LogGammaLinSpace'])
print(type(CorrectionTable))
#pivot_table = pd.pivot(CorrectionTable, index='Discipline', columns='Category', values='LogGammaLinSpace')
pivot_table = pd.pivot(CorrectionTable, columns=['Discipline','Category'], values='LogGammaLinSpace')
print(type(pivot_table))
print(pivot_table)
'''

CorrectionTable = pd.DataFrame()
for dis in dfStats['Discipline'].unique():
    disDf = pd.DataFrame()
    for cat in dfStats.loc[dfStats['Discipline'] == dis]['Category'].unique():
        disDf[cat] = dfStats.loc[dfStats['Category'] == cat]['LogGammaLinSpace'].explode('LogGammaLinSpace')
    disDf['Discipline'] = dis
    CorrectionTable = pd.concat([CorrectionTable,disDf],ignore_index=True)

dfNormalized = df.copy(deep=True)
df['State'] = "Original"
dfNormalized[resultKey] = dfNormalized.apply(lambda row: normalizeResult(row, resultKey, CorrectionTable, refCat),axis=1)
dfNormalized['State'] = "Liniarly Normalized"
dfCompare = pd.concat([df,dfNormalized],ignore_index=True)

CorrectionTable = pd.DataFrame()
for dis in dfStats['Discipline'].unique():
    disDf = pd.DataFrame()
    for cat in dfStats.loc[dfStats['Discipline'] == dis]['Category'].unique():
        # Create a tuple (linSpace,CDF) per category
        #disDf[cat] = list(zip(dfStats.loc[dfStats['Category'] == cat]['LogGammaLinSpace'].explode('LogGammaLinSpace'),dfStats.loc[dfStats['Category'] == cat]["LogGammaCdf"].explode('LogGammaCdf')))

        # Ad the linspace and CDF in separate columns
        disDf[cat]          = dfStats.loc[dfStats['Category'] == cat]['LogGammaLinSpace'].explode('LogGammaLinSpace')
        disDf[f'{cat}_CDF']  = dfStats.loc[dfStats['Category'] == cat]["LogGammaCdf"].explode('LogGammaCdf')
    disDf['Discipline'] = dis
    CorrectionTable = pd.concat([CorrectionTable,disDf],ignore_index=True)

dfNormalized = df.copy(deep=True)
df['State'] = "Original"
dfNormalized[resultKey] = dfNormalized.apply(lambda row: normalizeResultCDF(row, resultKey, CorrectionTable, refCat),axis=1)
dfNormalized['State'] = "CDF Normalized"
dfCompare = pd.concat([dfCompare,dfNormalized],ignore_index=True)

fig = px.box(dfCompare,y=resultKey,x='Category',color='State')
fig.update_layout(title="Comparison between original results and Normalized results")
fig.show()

print(dfCompare[(dfCompare['State'] == "CDF Normalized") & (dfCompare['Year'] == 2023)].sort_values('Result',ascending=False).head(n=25))
dfCompare[(dfCompare['State'] == "CDF Normalized") & (dfCompare['Year'] == 2023)].sort_values('Result',ascending=False).to_csv('BOA 2023 Normalized.csv',index=False)
#fig = px.box(dfCompare,y=resultKey,x='Category',facet_col='State')
#fig.show()
