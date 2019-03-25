# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 09:59:57 2019

@author: GaneshNethi
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

data = pd.read_csv('D://ML-Class//Time Series//datasets//MonthWiseMarketArrivals_Clean.csv')

#changing the date column to a Tie Interval Column
data.date = pd.DatetimeIndex(data.date)

#change the index to the date column
data.index = pd.PeriodIndex(data.date,freq='M')

#sort the data frame by date
data = data.sort_values(by='date')

data_Bglr = data.loc[data.city=='BANGALORE'].copy()

#Drop redundant columns
data_Bglr = data_Bglr.drop(['market','month','year','state','city','priceMin','priceMax'],axis=1) 

data_Bglr.priceMod.plot()
data_Bglr.quantity.plot()
data_Bglr.priceMod.plot(kind='hist',bins=30)


data_Bglr['priceModLog'] = np.log(data_Bglr.priceMod)
data_Bglr['priceModLog']
data_Bglr.priceModLog.plot()

#MeanModel
model_mean_pred = data_Bglr.priceModLog.mean()

#Let us store this as our Mean Prediction Value
data_Bglr['priceMean'] = np.exp(model_mean_pred)
data_Bglr.plot(kind='line',x='date',y=['priceMod','priceMean'])

def RMSE(predicted, actual):
    mse = (predicted - actual)**2
    rmse = np.sqrt(mse.sum()/mse.count())
    return round(rmse,3)


Rmse = RMSE(data_Bglr['priceMean'],data_Bglr['priceMod'])
    
data_Bglr_results = pd.DataFrame(columns=['Model','Forecast','RMSE'])
data_Bglr_results.loc[0,'Model'] = 'Mean'
data_Bglr_results.loc[0,'Forecast']  = np.exp(model_mean_pred)
data_Bglr_results.loc[0,'RMSE'] = Rmse



#--------------------------------------------------------------------
#what is the starting month of our data
data_Bglr.date.min()

#convert date in datetimedelta figure starting from zero
data_Bglr['timeIndex'] = data_Bglr.date - data_Bglr.date.min()

#convert to months using the timedelta function
data_Bglr['timeIndex'] = data_Bglr['timeIndex']/np.timedelta64(1,'M')

#Round the number to 0
data_Bglr['timeIndex'] = data_Bglr['timeIndex'].round(0).astype(int)

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller

#Now plot linear regression between priceMod and timeIndex
model_linear = smf.ols('priceModLog ~ timeIndex',data=data_Bglr).fit()
model_linear.summary()

#Parameters for y=mx+c equation
model_linear.params
model_linear_pred = model_linear.predict()

#plot the prediction line
data_Bglr.plot(kind='line',x='timeIndex',y='priceModLog')
plt.plot(data_Bglr.timeIndex,model_linear_pred,'-')
data_Bglr['priceLinear'] = np.exp(model_linear_pred)

#RMSE
model_linear_rmse = RMSE(data_Bglr.priceLinear,data_Bglr.priceMod)

#Manula Calculation
model_linear_forecast_manual = 0.009283*146+6.112108


data_Bglr_results.loc[1,'Model'] = 'Linear'
data_Bglr_results.loc[1,'Forecast'] = np.exp(model_linear_forecast_manual)
data_Bglr_results.loc[1,'RMSE'] = model_linear_rmse
#-----------------------------------------------------------------
#Random Walk
data_Bglr["priceModLogShift1"] = data_Bglr.priceModLog.shift()

# Lets plot the one-month difference curve
data_Bglr["priceModLogDiff"] = data_Bglr.priceModLog - data_Bglr.priceModLogShift1

data_Bglr.priceModLogDiff.plot()
data_Bglr["priceRandom"] = np.exp(data_Bglr.priceModLogShift1)
data_Bglr.plot(kind="line", x="timeIndex", y = ["priceMod","priceRandom"])

# Root Mean Squared Error (RMSE)
model_random_RMSE = RMSE(data_Bglr.priceRandom, data_Bglr.priceMod)
model_random_RMSE

data_Bglr_results.loc[2,"Model"] = "Random"
data_Bglr_results.loc[2,"Forecast"] = np.exp(data_Bglr.priceModLogShift1[-1])
data_Bglr_results.loc[2,"RMSE"] = model_random_RMSE
data_Bglr_results.head()
#-------------------------------------------------------------------
def adf(ts):
    
    # Determing rolling statistics
    rolmean = pd.rolling_mean(ts, window=12)
    rolstd = pd.rolling_std(ts, window=12)

    #Plot rolling statistics:
    orig = plt.plot(ts, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    # Calculate ADF factors
    adftest = adfuller(ts, autolag='AIC')
    adfoutput = pd.Series(adftest[0:4], index=['Test Statistic','p-value','# of Lags Used',
                                              'Number of Observations Used'])
    for key,value in adftest[4].items():
        adfoutput['Critical Value (%s)'%key] = value
    return adfoutput


data_Bglr['priceModLogMA12'] = pd.rolling_mean(data_Bglr.priceModLog, window = 12)
data_Bglr.plot(kind ="line", y=["priceModLogMA12", "priceModLog"])

data_Bglr['priceModLogMA4'] = pd.rolling_mean(data_Bglr.priceModLog, window = 4)
data_Bglr.plot(kind ="line", y=["priceModLogMA4", "priceModLog"])

data_Bglr["priceMA12"] = np.exp(data_Bglr.priceModLogMA12)
data_Bglr.tail()
model_MA12_forecast = data_Bglr.priceModLog.tail(12).mean()
model_MA12_RMSE = RMSE(data_Bglr.priceMA12, data_Bglr.priceMod)
model_MA12_RMSE

data_Bglr["priceMA4"] = np.exp(data_Bglr.priceModLogMA4)
data_Bglr.tail()
model_MA4_forecast = data_Bglr.priceModLog.tail(4).mean()
model_MA4_RMSE = RMSE(data_Bglr.priceMA4, data_Bglr.priceMod)
model_MA4_RMSE

data_Bglr_results.loc[3,"Model"] = "Moving Average 12"
data_Bglr_results.loc[3,"Forecast"] = np.exp(model_MA12_forecast)
data_Bglr_results.loc[3,"RMSE"] = model_MA12_RMSE
data_Bglr_results.head()

data_Bglr_results.loc[4,"Model"] = "Moving Average 4"
data_Bglr_results.loc[4,"Forecast"] = np.exp(model_MA4_forecast)
data_Bglr_results.loc[4,"RMSE"] = model_MA4_RMSE
data_Bglr_results.head()

data_Bglr['priceModLogExp12'] = pd.ewma(data_Bglr.priceModLog, halflife=12)
data_Bglr.plot(kind ="line", y=["priceModLogExp12", "priceModLog"])
data_Bglr["priceExp12"] = np.exp(data_Bglr.priceModLogExp12)
data_Bglr.tail()
# Root Mean Squared Error (RMSE)
model_Exp12_RMSE = RMSE(data_Bglr.priceExp12, data_Bglr.priceMod)
model_Exp12_RMSE

y_exp = data_Bglr.priceModLog[-1]
y_exp
y_for = data_Bglr.priceModLogExp12[-1]
y_for
halflife = 12
alpha = 1 - np.exp(np.log(0.5)/halflife)
alpha
model_Exp12_forecast = alpha * y_exp + (1 - alpha) * y_for

data_Bglr_results.loc[5,"Model"] = "Exp Smoothing 12"
data_Bglr_results.loc[5,"Forecast"] = np.exp(model_Exp12_forecast)
data_Bglr_results.loc[5,"RMSE"] = model_Exp12_RMSE
data_Bglr_results.head()
#--------------------------------------------------------------
from statsmodels.tsa.seasonal import seasonal_decompose

data_Bglr.index = data_Bglr.index.to_datetime()
decomposition = seasonal_decompose(data_Bglr.priceModLog, model = "additive")
decomposition.plot()

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

data_Bglr["priceDecomp"] = np.exp(trend + seasonal)

# Root Mean Squared Error (RMSE)
model_Decomp_RMSE = RMSE(data_Bglr.priceDecomp, data_Bglr.priceMod)
model_Decomp_RMSE

data_Bglr.plot(kind="line", x="timeIndex", y = ["priceMod",
                                              "priceDecomp"])

data_Bglr.plot(kind="line", x="timeIndex", y = ["priceMod", "priceMean", "priceLinear", "priceRandom",
                                             "priceMA12", "priceExp12", "priceDecomp"])

#ARIMA
ts = data_Bglr.priceModLog
ts_diff = data_Bglr.priceModLogDiff
ts_diff.dropna(inplace = True)

#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(ts_diff, nlags=20)
lag_acf

ACF = pd.Series(lag_acf)
ACF.plot(kind = "bar")

lag_pacf = pacf(ts_diff, nlags=20, method='ols')
PACF = pd.Series(lag_pacf)
PACF.plot(kind = "bar")

from statsmodels.tsa.arima_model import ARIMA
# Running the ARIMA Model(1,0,1)
model_AR1MA = ARIMA(ts_diff, order=(1,0,1))
results_ARIMA = model_AR1MA.fit(disp = -1)
results_ARIMA.fittedvalues.head()
ts_diff.plot()
results_ARIMA.fittedvalues.plot()
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
predictions_ARIMA_diff.tail()
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_log = pd.Series(ts.ix[0], index=ts.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
data_Bglr['priceARIMA'] = np.exp(predictions_ARIMA_log)
data_Bglr.plot(kind="line", x="timeIndex", y = ["priceMod", "priceARIMA"])

data_Bglr.plot(kind="line", x="timeIndex", y = ["priceMod", "priceMean", "priceLinear", "priceRandom",
                                             "priceMA12", "priceExp12", "priceDecomp", "priceARIMA"])

