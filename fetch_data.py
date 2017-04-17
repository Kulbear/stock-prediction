import tushare as ts
import pandas as pd

def wash(df):
    df = df.reset_index(drop=True)
    col_list = df.columns.tolist()
    col_list.remove('close')
    col_list.append('close')
    return df[col_list]

def get_3_years_history(stock_index, ktype='D'):
    df = ts.get_hist_data(stock_index, ktype=ktype)
    df = wash(df)
    df.to_csv('{}-3-year.csv'.format(stock_index), index=False)

def get_all_history(stock_index, start, ktype='D', autype=None):
    df = ts.get_h_data(stock_index, start=start, autype=autype)
    df = wash(df)
    df.to_csv('{}-from-{}.csv'.format(stock_index, start), index=False)

get_all_history('000002', start='2016-01-01')
get_3_years_history('000002')
