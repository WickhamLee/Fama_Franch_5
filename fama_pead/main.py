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

# from basic_code.load_data import load_basic_data
from basic_code.unit_portfolio_construct import unit_portfolio_construct as upc
from basic_code.load_data import load_basic_data as lbd
# from tools_package import fund_mgm_utilities as fmu
from tools_package.signals_general import *
from basic_code.analysis_main import main 




#--------输入要分析的产品名称----------------
# deal_name = '盛泉恒元进取1号'
# deal_name = '财丰一号'
upc = upc(deal_name)



dependent_variable = upc.LBD.portfolio
dependent_variable['date'] = dependent_variable.index
fac_list = ['const', 'Rm-Rf', 'SMB', 'HML', 'RMW', 'CMA']    

# 按季度分析的准备工作
dependent_variable["quarter"] = pd.DatetimeIndex(dependent_variable["date"]).quarter
gb = dependent_variable.groupby([dependent_variable['date'].dt.year, 'quarter'])
frequency = gb.apply(lambda _df: _df.shape[0])


# 按月分析的准备工作
# dependent_variable["year"] = pd.DatetimeIndex(dependent_variable["date"]).quarter
# gb = dependent_variable.groupby([dependent_variable['date'].dt.year, dependent_variable['date'].dt.month])
# frequency = gb.apply(lambda _df: _df.shape[0])


# 基本信息、β、显著性、因子均值      
info_, corr, p_value, dic_acc_ret, dic_SMB, dic_HML, dic_RMW, dic_CMA = main(upc, frequency, dependent_variable)   
        
    
print(info_['2020-10-31'])


df_p_value = pd.DataFrame(p_value,index = fac_list)
fig = plt.figure(1, figsize=(40,20))
# fig.set_facecolor('#17212e')
ax = plt.subplot(111)#, facecolor='#17212e')
sns.heatmap(df_p_value.T, annot=True, fmt='.4f')
ax.set_title('各个因子在时间序列上的β')
plt.show()


df_corr = pd.DataFrame(corr, index = fac_list).T
df_corr.to_excel(os.path.join(here_path, 'portfolio_data', deal_name, '分析结果', 'β_'+deal_name+'.xlsx'))

df_p_value = pd.DataFrame(p_value, index = fac_list).T
df_p_value.to_excel(os.path.join(here_path, 'portfolio_data', deal_name, '分析结果', '显著性_'+deal_name+'.xlsx'))

df_fac = pd.concat([pd.Series(dic_acc_ret),
                    pd.Series(dic_SMB),
                    pd.Series(dic_HML),
                    pd.Series(dic_RMW),
                    pd.Series(dic_CMA)], axis=1)
df_fac.columns = fac_list[1:]
df_fac.to_excel(os.path.join(here_path, 'portfolio_data', deal_name, '分析结果', '因子值_'+deal_name+'.xlsx'))




