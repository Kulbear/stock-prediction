#!/usr/bin/python
# -*- coding: utf-8 -*-
import tushare as ts
import pandas as pd


def wash(df, target='close'):
    """Process the entered DataFrame object.
    
    The last column of the output DataFrame is our prediction target.
    
    # Arguments
        df: input Pandas DataFrame object.
    # Returns
        Postprocessed DataFrame object.
    """
    df = df.reset_index(drop=True)
    col_list = df.columns.tolist()
    col_list.remove(target)
    col_list.append(target)
    return df[col_list]

def get_3_years_history(stock_index, ktype='D'):
    """Get 3 years history for a specified stock.
    
    History with detailed information (candlestick chart data) then saved to csv format.
    
    # Arguments
        stock_index: stock index code.
        ktype: candlestick data type.
    """
    df = ts.get_hist_data(stock_index, ktype=ktype)
    df = wash(df)
    print('\nSaving DataFrame: \n', df.head(5))
    df.to_csv('{}-3-year.csv'.format(stock_index), index=False)

def get_all_history(stock_index, start, autype=None):
    """Get history for a specified stock during a specified period.
    
    Saved to csv format.
    
    # Arguments
        stock_index: stock index code.
        start: start date of the interested period.
        autype: rehabilitation type.
    """
    df = ts.get_h_data(stock_index, start=start, autype=autype)
    df = wash(df)
    print('\nSaving DataFrame: \n', df.head(5))
    df.to_csv('{}-from-{}.csv'.format(stock_index, start), index=False)
    
get_all_history('000002', start='1995-01-01')
get_3_years_history('000002')
