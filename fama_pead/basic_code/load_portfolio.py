# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 13:10:29 2020

@author: WickhamLee
"""

import os
import sys
# here_path = os.getcwd()
back1_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
# back2_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
# if here_path not in sys.path: sys.path.append(here_path)
if os.path.join(back1_path, 'tools_package') not in sys.path: sys.path.append(os.path.join(back1_path, 'tools_package'))
import numpy as np
import pandas as pd
import datetime as dt
import time
import statsmodels.api as sm
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
plt.style.use("classic") 
import seaborn as sns
plt.style.use('classic')
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False
from openpyxl import load_workbook
import warnings
warnings.filterwarnings("ignore") 

import fund_mgm_utilities as fmu
from signals_general import *
# from tools_api import dataframe_date_standard as dds
# from load_data import load_basic_data
# from basic_code.load_data import load_basic_data
# from unit_portfolio_construct import unit_portfolio_construct as upc



class load_portfolio_data():
    def __init__(self,deal_name):
        self.deal_name = deal_name
        #底层数据载入
        # self.basic_data = upc()
        
#        上级路径
        self.here_path = os.getcwd()
        #上级路径
        self.back1_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        #上上级路径
        self.back2_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
        
        
        #原始文件
        self.portfolio_raw = self.origin_data_load()
        #行数对其后的portfolio
        # self.portfolio_adj = self.length_standard()
    def origin_data_load(self):
        # deal_name = '财丰一号'
        # a= pd.read_excel(r"D:\fama_pead\portfolio_data\财丰1号\财丰1号.xlsx")
        dependent_variable = reldiff(pd.read_excel(os.path.join(self.back1_path, 'portfolio_data', 
                                                                self.deal_name, self.deal_name+'.xlsx'),
                                           index_col=[0]))
        return dependent_variable
    
    # def length_standard(self):
    #     re = dds(self.portfolio_raw, self.basic_data.close_price)
    #     if re[1] == []:
    #         print(self.deal_name + '时间序列已完整，无需补齐')
    #     else:
    #         print(self.deal_name + '时间序列已补齐, 缺失日期'+str(re[1])+ '已用np.nan补齐')
    #     return dds(self.portfolio_raw, self.basic_data.close_price)[0]
if __name__ == '__main__':
    a = load_portfolio_data('财丰一号')