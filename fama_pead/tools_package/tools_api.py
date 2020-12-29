# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 13:37:46 2020

@author: WickhamLee
"""
import numpy as np
# import time
import pandas as pd
#-----------------------
#dataframe_date_standard0
#-----------------------
#df_dictionary的index中有的日期，但df_input没有的话，在df_input以np.nan补充一行
def dataframe_date_standard0(df_input, df_dictionary):
    out_df = df_input.copy()
    out_df = out_df.T
    df_dictionary.index = pd.DatetimeIndex(df_dictionary.index)
    date_dis = []
    for i in df_dictionary.index:
        if i not in list(df_input.index):
            date_dis.append(i.strftime("%Y-%m-%d"))
            out_df[i] = [np.nan for _ in range(out_df.shape[0])]
    return out_df.T.sort_index(), date_dis

#-----------------------
#dataframe_date_standard1
#-----------------------
#df_input的index里的日期会比df_dictionary多，删掉df_input里多出来的日期
def dataframe_date_standard1(df_input, df_dictionary):
    df_input.index = pd.DatetimeIndex(df_input.index)
    out_df = df_input.copy().T
    
    df_dictionary.index = pd.DatetimeIndex(df_dictionary.index)
    for i in df_input.index:
        if i not in list(df_dictionary.index):
            del out_df[i]
    
    for i in df_dictionary.index:
        if i not in list(df_input.index):
            out_df[i] = [np.nan for _ in range(out_df.shape[0])]
    return out_df.T.sort_index().ffill()