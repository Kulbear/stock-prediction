#!/usr/bin/env python
# coding: utf-8

# # Stock Data Preparation Demo
# 
# Today we will do a simple experiment on a stock prediction case. 
# 
# In order to save time on preparing the data, I'd like to introduce **TuShare**. You can use any other data source if you don't want to deal with Chinese stock market or you are not familiar with Chinese (TuShare's doc is all Chinese). 
# 
# In addition, you can have as many features as you want unless you keep the close price (the one we want to predict!) as the last column in your Pandas DataFrame.
# 
# **This is just a demo about how to use TuShare and deal with its data. You can find a separate (and simple) script with instructions about fetching more detailed data.**

# In[1]:


import tushare as ts # TuShare is a utility for crawling historical data of China stocks
import pandas as pd
import os

def get_data(traingdata = True):
    stocks = {}
    with open("stocklist.txt","r") as ff:
        lines = ff.readlines()
        for line in lines:
            items = line.split(",")
            if len(items[0]) > 0:
                stocks[items[0]] = items[1]

    if traingdata:
        start_date = '1995-01-01'
        end_date = None  # We will use today as the end date here, you can specify one if you want
    else:
        start_date = '2018-06-01'
        end_date = None  # We will use today as the end date here, you can specify one if you want

    for (stock_index, stock_name) in stocks.items():
        print(stock_index)
        if traingdata:
            csv_name = './trainingdata/%s'%(stock_index)
        else:
            csv_name = './inferencedata/%s'%(stock_index)
        # the data saved to csv order by date, the latest date is at the top.
        df = ts.get_h_data(stock_index, start=start_date, autype=None, retry_count=5, pause=5)
        df = df.sort_index(ascending=True)
        #df = df.reset_index(drop=True)
        df = df.reset_index()
        col_list = df.columns.tolist()
        col_list.remove('close')
        col_list.remove('amount') # Just for simplicity, should not be removed
        col_list.append('close')
        df = df[col_list]
        df['volume'] = df['volume'] / 1000000
        df.to_csv(csv_name, index=False)
        validate_df = pd.read_csv(csv_name)
        validate_df.head()



if __name__ == "__main__":
    get_data(False)


