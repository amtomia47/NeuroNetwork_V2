import yfinance as yf
import pandas as pd

## get stock price of Microsoft in last ten years
file = open('Stock_data/stock_name','r')
name = file.read()
data = yf.Ticker(name).history(period='10y')

data.to_csv('Stock_data/original/'+name+'10Y')