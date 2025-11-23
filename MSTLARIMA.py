#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 11:23:38 2024

@author: arda
"""

import matplotlib.pyplot as plt
from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoARIMA
from statsforecast.utils import ConformalIntervals
import pandas as pd
from pylab import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose
from LOF import LOFAlgorithm

LOF = LOFAlgorithm()


plt.style.use('grayscale')
plt.rcParams['lines.linewidth'] = 1.5

rcParams['figure.figsize'] = (18,7)

df = pd.read_csv("DataSet.csv")

df['time'] = pd.to_datetime(df['time'])

df.set_index('time', inplace=True)

df = df[['src','dst']]

df = df[df['dst'] == '192.168.1.20']

df = df.groupby('time').size().reset_index()

df.columns = ['time','Request']

row = df

df["unique_id"]="1"

df.columns=["ds", "y", "unique_id"]

StatsForecast.plot(df)

fig, axs = plt.subplots(nrows=1, ncols=1)

print('##')
print(df)
print('##')

LOFDF = df.reset_index()
LOFDF = LOFDF[['index','y']]
print(LOFDF)

LOFDF = LOF.LofProcess(LOFDF)

LOFDF.insert(1,'ds',df['ds'])

plt.scatter(LOFDF["ds"],LOFDF['y'], color="k", s=3.0, label="Data points")
scatter = plt.scatter(
    LOFDF["ds"],
    LOFDF['y'],
    s=-100 * LOFDF['outlier'],
    edgecolors="r",
    facecolors="none",
    label="Outlier scores",
)

test = df[df.ds>'2024-06-09 22:30:00']

plt.title("Number of requests sent to 192.168.1.20");
plt.xlabel("Hours",color='black')
plt.tick_params(axis='y',colors='black')
plt.tick_params(axis='x',colors='black')
plt.show()

result = seasonal_decompose(
            df['y'], model='additive', filt=None, period=24,
            two_sided=True, extrapolate_trend=0)

data = pd.DataFrame(result.observed)
data.columns = ['data']
data['unique_id'] = 1

horizon = len(test) 
models = [MSTL(season_length=[24, 168],
trend_forecaster=AutoARIMA(prediction_intervals=ConformalIntervals(n_windows=3, h=horizon)))]

sf = StatsForecast(df=df,
                    models=models,
                    freq='S',
                    n_jobs=-1)

sf.fit()
StatsForecast(models=[MSTL],freq=[24,168])

result=sf.fitted_[0,0].model_
result

sf.fitted_[0, 0].model_.tail(24 * 28).plot(subplots=True, grid=True,color='black')

plt.tick_params(axis='y',colors='black')
plt.tick_params(axis='x',colors='black')
plt.tight_layout()
plt.show()

values = sf.forecast(horizon, fitted=True)

values=sf.forecast_fitted_values()

StatsForecast.plot(values)
values['unique_id'] = 1
FORECAST = sf.forecast(h=horizon).reset_index()

Y_hat1 = pd.concat([values,FORECAST],  keys=['unique_id', 'ds'])
Y_hat1


fig, ax = plt.subplots(1, 1)
plot_df = pd.concat([df, Y_hat1]).set_index('ds')

Forecaseted_df = plot_df[plot_df.index > '2024-06-09 22:40:47']
plot_df = plot_df[plot_df.index <= '2024-06-09 22:40:46']

plot_df['y'].plot(ax=ax, linewidth=2)
plot_df["MSTL"].plot(ax=ax,linewidth=2, color="black")

Forecaseted_df["MSTL"].plot(ax=ax, label='Forecasted',linewidth=2, color="Green")
plt.xlim('2024-06-09 22:17:00', '2024-06-09 22:55:00')

ax.set_title(' Forecast', fontsize=22)
ax.set_ylabel('Number of requests sent to 192.168.1.20', fontsize=15)
ax.set_xlabel('Daily', fontsize=15)
ax.legend(prop={'size': 15})
ax.grid(True)
plt.show()

print("ROW DATA:")
print(row)
print("Forecasted Data:")
print(Forecaseted_df)

