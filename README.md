# Overview
This is the 2nd solution of [US Stock Price Prediction](https://comp.probspace.com/competitions/us_stock_price) competition hosted by ProbSpace in Japan.

# About this competition
* Develop an algorithm to predict the stock price one week later using the stock price transition data of the US Stock Exchange.
* Predict the closing price of 2019/11/24 week based on the US stock data (closing price) for a total of 419 weeks from 2011/11/13 to 2019/11/17 week.
* External data can not be used.
* The prediction performance of the model is evaluated by the evaluation function RMSLE (Root Mean Squared Logarithmic Error).

# Solution
## Prediction target
This model predicts the difference between the logarithmic stock price and the previous week instead of the stock price (closing price) itself.

## Models
It has the following 9 models and ensembles the results. The price movements between Dow and NASDAQ is different, so they are divided by market. For example, LightGBM_NYSE learns based on data from all markets, but only NYSE's prediction results are used.

![StockPricePrediction drawio](https://user-images.githubusercontent.com/40084422/145959316-4240c709-c446-4bbf-af8e-01b1759e09f1.png)

|No.|Model|Training Dataset|Predicition|
|:----|:----|:----|:----|
|1|LightGBM_ALL_1|NYSE, NASDAQ, AMEX|NYSE, NASDAQ, AMEX|
|2|LightGBM_ALL_2[^1]|NYSE, NASDAQ, AMEX|NYSE, NASDAQ, AMEX|
|3|LightGBM_NYSE|NYSE, NASDAQ, AMEX|NYSE|
|4|LightGBM_NASDAQ|NASDAQ|NASDAQ|
|5|LightGBM_AMEX|NASDAQ|AMEX|
|6|MLP_NYSE|NYSE, NASDAQ, AMEX|NYSE|
|7|MLP_NASDAQ&AMEX|NASDAQ, AMEX|NASDAQ, AMEX|
|8|ExtraTrees_NYSE&AMEX|NYSE|NYSE, AMEX|
|9|ExtraTrees_NASDAQ|NASDAQ|NASDAQ|

[^1]: The difference from LightGBM_ALL_1 is only random seed and hyper parameters.

## Features
The base features are the same as [LightGBM Base line (LB = 0.03781)](https://comp.probspace.com/competitions/us_stock_price/discussions/DT-SN-Posta3d47ae1bcea01c64bd5). Sector, Industry and Market were target encoded.

* Lag feature amount (for 4 weeks)
* 52-week simple moving average and its standard deviation
* Listed year
* Sector
* Industry
* Market
* Year
* Month
* Day
* How many weeks from the beginning of the year

|No.|Model|Features|
|:----|:----|:----|
|1|LightGBM_ALL_1|base features|
|2|LightGBM_ALL_2|base features|
|3|LightGBM_NYSE|base features|
|4|LightGBM_NASDAQ|base features|
|5|LightGBM_AMEX|base features|
|6|MLP_NYSE|base features, Skew|
|7|MLP_NASDAQ&AMEX|base features, EMA_GAP, Historical Volatility, Kurt, KST Oscillator, PCA, K-Means|
|8|ExtraTrees_NYSE&AMEX|base features|
|9|ExtraTrees_NASDAQ|base features|

## Score
0.03692 in [Private LB](https://comp.probspace.com/competitions/us_stock_price/ranking).

# How to open it on Google Colaboratory
Please take a look at [How-to-open-ipynb-on-Google-Colaboratory](https://github.com/shoji9x9/How-to-open-ipynb-on-Google-Colaboratory).
