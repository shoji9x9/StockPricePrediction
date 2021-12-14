# Overview
This is a 2nd solution of [US Stock Price Prediction](https://comp.probspace.com/competitions/us_stock_price) competition hosted by ProbSpace in Japan.

# About this competition
* Develop an algorithm to predict the stock price one week later using the stock price transition data of the US Stock Exchange.
* Predict the closing price of 2019/11/24 week based on the US stock data (closing price) for a total of 419 weeks from 2011/11/13 to 2019/11/17 week.
* External data can not be used.
* The prediction performance of the model is evaluated by the evaluation function RMSLE (Root Mean Squared Logarithmic Error).

# Solution
## Prediction target
This model predicts the difference between the logarithmic stock price and the previous week instead of the stock price (closing price) itself.

## Models
It has the following 9 models and ensembles the results. The price movements between Dow and NASDAQ is different, so they are divided by market.

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

## Score
0.03692 in Private LB.

# How to open it on Google Colaboratory
Please take a look at [How-to-open-ipynb-on-Google-Colaboratory](https://github.com/shoji9x9/How-to-open-ipynb-on-Google-Colaboratory).
