import os
import sys
here_path = os.getcwd()
back1_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
back2_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
if here_path not in sys.path: sys.path.append(here_path)
import numpy as np
import pandas as pd

import datetime as dt
import time
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
plt.style.use('classic')
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False
from openpyxl import load_workbook
import warnings
warnings.filterwarnings("ignore") 

back1_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if os.path.join(back1_path, 'tools_package') not in sys.path: sys.path.append(os.path.join(back1_path, 'tools_package'))
import fund_mgm_utilities as fmu
from signals_general import *
from load_data import load_basic_data
# from basic_code.load_data import load_basic_data
class unit_portfolio_construct():
    def __init__(self, deal_name):
        self.deal_name = deal_name
        self.LBD = load_basic_data(self.deal_name)
        self.here_path = self.LBD.here_path
        #上级路径
        self.back1_path = self.LBD.back1_path
        #上上级路径
        self.back2_path = self.LBD.back2_path
#        载入读好的底层数据

        # self.close_price, self.market_cap, self.pb_ratio, self.ROE, self.reldiff_total_assets, self.risk_free_rate, self.trade_date = LBD.load()
        self.close_price = self.LBD.dfs_adj[0]
        self.market_cap = self.LBD.dfs_adj[1]
        self.pb_ratio = self.LBD.dfs_adj[2]
        self.ROE = self.LBD.dfs_adj[3]
        self.reldiff_total_assets = self.LBD.dfs_adj[4]
        self.risk_free_rate = self.LBD.dfs_adj[5]
        
        
        
        
#        将各个个股按因子标记好属于哪个分类
        self.market_cap_cut = self.market_cap_cut()
        self.pb_ratio_cut = self.pb_ratio_cut()
        self.ROE_cut = self.ROE_cut()
        self.reldiff_total_assets_cut = self.reldiff_total_assets_cut()
#        收益率矩阵
        self.ret = reldiff(self.close_price.values)
#        市值加平均的市场组合收益率
        self.Rm = self.cap_weighted_ret()

        # self.fac_pead = self.fac_pead()
#==============================================================================        
        self.BH, self.SH = self.factor_cal(self.pb_ratio_cut, 1)
        self.BM, self.SM = self.factor_cal(self.pb_ratio_cut, 0)
        self.BL, self.SL = self.factor_cal(self.pb_ratio_cut,-1)

        self.BR, self.SR = self.factor_cal(self.ROE_cut, 1)
        self.BN_ROE, self.SN_ROE = self.factor_cal(self.ROE_cut, 0)
        self.BW, self.SW = self.factor_cal(self.ROE_cut,-1)
        
        self.BA, self.SA = self.factor_cal(self.reldiff_total_assets_cut, 1)
        self.BN_reldiff_total_assets, self.SN_reldiff_total_assets = self.factor_cal(self.reldiff_total_assets_cut, 0)
        self.BC, self.SC = self.factor_cal(self.reldiff_total_assets_cut,-1)
        
        self.BN = (self.BN_ROE+self.BN_reldiff_total_assets)/2
        self.SN = (self.SN_ROE+self.SN_reldiff_total_assets)/2
#==============================================================================  

#==============================================================================  
        self.SMB_bm = (self.SH+self.SN+self.SL)/3 - (self.BH+self.BN+self.BL)/3    
        self.SMB_op = (self.SR+self.SN+self.SW)/3 - (self.BR+self.BN+self.BW)/3 
        self.SMB_inv= (self.SC+self.SN+self.SA)/3 - (self.BC+self.BN+self.BA)/3 
        self.SMB = (self.SMB_bm+self.SMB_op+self.SMB_inv)/3
        
        self.HML = (self.SH+self.BH)/2 - (self.SL+self.BL)/2
        
        self.RMW = (self.SR+self.BR)/2 - (self.SW+self.BW)/2
        
        self.CMA = (self.SC+self.BC)/2 - (self.SA+self.BA)/2
        # self.test = self.factor_cal(self.market_cap_cut, -1)
#所有股票按市值分类，在每一个横截面上，比中位数大的记1，比中位数小的记-1
    def market_cap_cut(self):
#        market_cap_cut_df = pd.DataFrame(columns = self.market_cap.columns, index = self.market_cap.index)
        market_cap_cut_mtx = np.zeros((self.market_cap.shape[0], self.market_cap.shape[1]))
        
        for i in range(market_cap_cut_mtx.shape[0]):
            tool=self.market_cap.values[i][np.logical_not(np.isnan(self.market_cap.values[i]))]
            i_median = np.median(tool)
            market_cap_cut_mtx[i][self.market_cap.values[i]>=i_median] = 1
            market_cap_cut_mtx[i][self.market_cap.values[i]<i_median] = -1
        market_cap_cut_mtx[market_cap_cut_mtx==0] = np.nan
#        print(market_cap_cut_mtx)
        market_cap_cut_df = pd.DataFrame(market_cap_cut_mtx, columns = self.market_cap.columns, index = self.market_cap.index)
        
        return market_cap_cut_df

#所有股票按市净率分类，在每一个横截面上，比7分位数大的记1，比3分位数小的记-1，之间的记0
    def pb_ratio_cut(self):
        pb_ratio_cut_mtx = np.zeros((self.pb_ratio.shape[0], self.pb_ratio.shape[1]))
        
        for i in range(pb_ratio_cut_mtx.shape[0]):
            tool=self.pb_ratio.values[i][np.logical_not(np.isnan(self.pb_ratio.values[i]))] 
            i_3_7_percentile = np.percentile(tool, [30, 70])
#            print(i_3_7_percentile)
            pb_ratio_cut_mtx[i][self.pb_ratio.values[i]>=i_3_7_percentile[1]] = 1
            pb_ratio_cut_mtx[i][self.pb_ratio.values[i]<=i_3_7_percentile[0]] = -1
        
        pb_ratio_cut_mtx[np.isnan(self.pb_ratio.values)] = np.nan
        
        pb_ratio_cut_df = pd.DataFrame(pb_ratio_cut_mtx, columns = self.pb_ratio.columns, index = self.pb_ratio.index)
#        print(pb_ratio_cut_mtx )
        return pb_ratio_cut_df
    
#所有股票按ROE分类，在每一个横截面上，比7分位数大的记1，比3分位数小的记-1，之间的记0
    def ROE_cut(self):
        ROE_cut_mtx = np.zeros((self.ROE.shape[0], self.ROE.shape[1]))
        
        for i in range(ROE_cut_mtx.shape[0]):
            tool=self.ROE.values[i][np.logical_not(np.isnan(self.ROE.values[i]))] 
            i_3_7_percentile = np.percentile(tool, [30, 70])
            ROE_cut_mtx[i][self.ROE.values[i]>=i_3_7_percentile[1]] = 1
            ROE_cut_mtx[i][self.ROE.values[i]<=i_3_7_percentile[0]] = -1
#        ROE_cut_mtx[ROE_cut_mtx==0] = np.nan
        ROE_cut_mtx[np.isnan(self.ROE.values)] = np.nan

        ROE_cut_df = pd.DataFrame(ROE_cut_mtx, columns = self.ROE.columns, index = self.ROE.index)
        
        return ROE_cut_df
    
#所有股票按总资产分类，在每一个横截面上，比7分位数大的记1，比3分位数小的记-1，之间的记0
    def reldiff_total_assets_cut(self):
        reldiff_total_assets_cut_mtx = np.zeros((self.reldiff_total_assets.shape[0], self.reldiff_total_assets.shape[1]))
        
        for i in range(reldiff_total_assets_cut_mtx.shape[0]):
            tool=self.reldiff_total_assets.values[i][np.logical_not(np.isnan(self.reldiff_total_assets.values[i]))]
            i_3_7_percentile = np.percentile(tool, [30, 70])
            reldiff_total_assets_cut_mtx[i][self.reldiff_total_assets.values[i]>=i_3_7_percentile[1]] = 1
            reldiff_total_assets_cut_mtx[i][self.reldiff_total_assets.values[i]<=i_3_7_percentile[0]] = -1
            
        reldiff_total_assets_cut_mtx[np.isnan(self.reldiff_total_assets.values)] = np.nan    
            
        reldiff_total_assets_cut_df = pd.DataFrame(reldiff_total_assets_cut_mtx, columns = self.reldiff_total_assets.columns, index = self.reldiff_total_assets.index)
        
        return reldiff_total_assets_cut_df

#pead因子
    def fac_pead(self):
        long_ret, short_ret = [], []
        for i in range(self.ret.shape[0]):
            long_ret.append(np.nanmean(self.ret[i][self.pead_pub.values[i]==1]))
            short_ret.append(np.nanmean(self.ret[i][self.pead_pub.values[i]==-1]))
        
        total_ret = list(map(lambda x:x[0]-x[1], zip(long_ret, short_ret)))
        total_value=[1]
        for i in total_ret:
            if np.isnan(i): total_value.append(total_value[-1])
            else: total_value.append(total_value[-1]*(1+i))
        return total_value

    
#计算各个因子
    def factor_cal(self, fac, direcrion):
        fac_ts_small_cap, fac_ts_large_cap = [], []
        for i in range(fac.shape[0]):
            large_cap_selected = np.nansum(self.market_cap.values[i][(fac.values[i]==direcrion)&(self.market_cap_cut.values[i]== 1)])
            small_cap_selected = np.nansum(self.market_cap.values[i][(fac.values[i]==direcrion)&(self.market_cap_cut.values[i]==-1)])
            
            large_cap_pos = self.market_cap.values[i][(fac.values[i]==direcrion)&(self.market_cap_cut.values[i]== 1)]/large_cap_selected
            small_cap_pos = self.market_cap.values[i][(fac.values[i]==direcrion)&(self.market_cap_cut.values[i]==-1)]/small_cap_selected
            # print(self.market_cap.values[i])
            fac_ts_large_cap.append(np.nansum(self.ret[i][(fac.values[i]==direcrion)&(self.market_cap_cut.values[i]== 1)] * large_cap_pos))
            fac_ts_small_cap.append(np.nansum(self.ret[i][(fac.values[i]==direcrion)&(self.market_cap_cut.values[i]==-1)] * small_cap_pos))
            
        return np.array(fac_ts_large_cap), np.array(fac_ts_small_cap)
    
#计算市值加平均的市场组合收益率
    def cap_weighted_ret(self):
        ret_ts = []
        for i in range(self.market_cap.shape[0]):
            total_cap_selected = np.nansum(self.market_cap.values[i])
            pos = self.market_cap.values[i]/total_cap_selected
            ret_ts.append(np.nansum(self.ret[i] * pos))
        
        return np.array(ret_ts)
if __name__ == '__main__':
    a = unit_portfolio_construct()
    # b = a.test