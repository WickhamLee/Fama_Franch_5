# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 11:25:54 2020

@author: whli
"""
import numpy as np
import pandas as pd
import os
import datetime as dt
import time
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
plt.style.use('classic')
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False
from tqdm import tqdm
from openpyxl import load_workbook
import warnings
warnings.filterwarnings("ignore") 

from tools_api import dataframe_date_standard1 as dds1
from load_portfolio import load_portfolio_data as lpd
# from basic_code.load_portfolio import load_portfolio_data as lpd
class load_basic_data:
    def __init__(self, deal_name):
        self.deal_name = deal_name
#        上级路径
        self.here_path = os.getcwd()
        #上级路径
        self.back1_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        #上上级路径
        self.back2_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
        
#        self.close_price, self.market_cap, self.pb_ratio, self.ROE, self.reldiff_total_assets = self.load()
        self.portfolio = lpd(self.deal_name).portfolio_raw
        # self.dfs_adj_raw = 
        self.dfs_adj = self.length_standard()
    def load(self):
#        读盘路径
        data_path = os.path.join(self.back1_path, "basic_data")
        
        close_price = pd.read_hdf(os.path.join(data_path, 'close.hd5'), 'table')
        print("收盘价数据已载入")
        market_cap = pd.read_hdf(os.path.join(data_path, 'market_cap.hd5'), 'table')
        print("市值数据已载入")
        pb_ratio = pd.read_hdf(os.path.join(data_path, 'pb_ratio.hd5'), 'table')
        print("市净率数据已载入")
        ROE = pd.read_hdf(os.path.join(data_path, 'ROE.hd5'), 'table')
        print("ROE数据已载入")
        reldiff_total_assets = pd.read_hdf(os.path.join(data_path, 'reldiff_total_assets.hd5'), 'table')
        print("总资产数据已载入")
        risk_free_rate = pd.read_excel(os.path.join(data_path, 'risk_free_rate.xlsx'), index_col=[0])
        print("无风险利率数据已载入")
        # trade_date = pd.read_excel(os.path.join(data_path, 'trade_calander.xlsx'), index_col=[0])
        # print("交易日期已载入")
        # pead_pub = pd.read_excel(os.path.join(data_path, 'pead_pub_.xlsx'), index_col=[0])
        # print("pead的fdhwelfhoewhfnl已载入")
        
        if (close_price.shape[0] == market_cap.shape[0] == pb_ratio.shape[0] == ROE.shape[0] == reldiff_total_assets.shape[0]) &(
                close_price.shape[1] == market_cap.shape[1] == pb_ratio.shape[1] == ROE.shape[1] == reldiff_total_assets.shape[1]):
            print("载入数据长度一致")
        else:
            raise Exception("载入数据长度不一致："+
                            "收盘价:"+str(close_price.shape[0])+"x"+str(close_price.shape[1])+
                            "市值："+str(market_cap.shape[0])+"x"+str(market_cap.shape[1])+
                            "市净率:"+str(pb_ratio.shape[0])+"x"+str(pb_ratio.shape[1])+
                            "ROE:"+str(ROE.shape[0])+"x"+str(ROE.shape[1])+
                            "总资产:"+str(reldiff_total_assets.shape[0])+"x"+str(reldiff_total_assets.shape[1]))
        return close_price.T.sort_index().T, market_cap.T.sort_index().T, pb_ratio.T.sort_index().T, ROE.T.sort_index().T, reldiff_total_assets.T.sort_index().T,risk_free_rate#, trade_date

    def length_standard(self):
        out_ = []
        print('开始调整底层数据尺寸')
        for i in tqdm(self.load()):
            # print(dds1(i, self.portfolio))
            out_.append(dds1(i, self.portfolio))
            
        return out_