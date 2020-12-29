# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 16:02:52 2020

@author: WickhamLee
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import datetime as dt
def main(u, frequency, dependent_variable):

    info_, corr, p_value, dic_acc_ret, dic_SMB, dic_HML, dic_RMW, dic_CMA= {}, {}, {}, {}, {}, {}, {}, {}
    count_ = 0
    for i in range(frequency.shape[0]):
        # print(count_, count_+frequency[i])
        # print(u.Rm)
        # print(u.risk_free_rate)
        acc_ret = u.Rm[count_:count_+frequency[i]] - u.risk_free_rate.values.reshape(1,len(u.Rm))[0][count_:count_+frequency[i]]
        SMB = u.SMB[count_:count_+frequency[i]]
        HML = u.HML[count_:count_+frequency[i]]
        RMW = u.RMW[count_:count_+frequency[i]]
        CMA = u.CMA[count_:count_+frequency[i]]
        
        dependent_variable_i = dependent_variable['close'].values.reshape(1,len(u.Rm))[0][count_:count_+frequency[i]]
        # count_ += frequency[i]
        
        # variables_mtx = np.mat([SMB, HML, RMW, CMA]).T
        variables_mtx = np.mat([acc_ret, SMB, HML, RMW, CMA]).T
        # print(variables_mtx.shape)
        # year_month = str(dependent_variable.index[count_].year)+'年'+ str(dependent_variable.index[count_].month)+'月'
        EOM = 31
        
        while True:
            try:
                year_month = dt.datetime(dependent_variable.index[count_].year, dependent_variable.index[count_].month, EOM).strftime("%Y-%m-%d")
                break
            except:
                EOM -= 1
                
        
        
        
        # print(variables_mtx , dependent_variable_i)
        # weight = np.dot(np.dot(np.linalg.inv(np.dot(variables_mtx.T,variables_mtx)),variables_mtx.T),dependent_variable_i)
        # weights_[year_month] = np.array(weight)[0]
    
        variables_mtx = sm.add_constant(variables_mtx) # 若模型中有截距，必须有这一步
        # print(variables_mtx.shape , dependent_variable_i.shape)
        try:
            # print(1) 
            # print(dependent_variable_i[np.logical_not(np.isnan(dependent_variable_i))])
            # print(variables_mtx[np.logical_not(np.isnan(dependent_variable_i))])
            model = sm.OLS(dependent_variable_i[np.logical_not(np.isnan(dependent_variable_i))], variables_mtx[np.logical_not(np.isnan(dependent_variable_i))]).fit() # 构建最小二乘模型并拟合
        except:
            # print(2)
            model = sm.OLS(np.array([1,1,1]), np.zeros((3,6))).fit()
        
        # print(model.summary())
        info_[year_month] = model.summary()
        
        sumy = model.summary().as_csv().split('\n')
        # print(variables_mtx.shape)
        corr[year_month], p_value[year_month] =[], []
        for j in range(variables_mtx.shape[1]):
            corr[year_month].append(float(sumy[11+j].split(",")[1].strip()))
            p_value[year_month].append(float(sumy[11+j].split(",")[4].strip()))
        # corr[year_month] = correalition
            
        # dic_acc_ret[year_month] = np.mean(acc_ret)
        dic_SMB[year_month] = np.mean(SMB)
        dic_HML[year_month] = np.mean(HML)
        dic_RMW[year_month] = np.mean(RMW)
        dic_CMA[year_month] = np.mean(CMA)
        
        count_ += frequency[i]
    return info_, corr, p_value, dic_acc_ret, dic_SMB, dic_HML, dic_RMW, dic_CMA