import sys
from numba import jit
import pandas as pd
import numpy as np
from numba import guvectorize
import fund_mgm_utilities as fmu
import re
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression 
#from strat_signals.signals_wqgt import *


np.seterr(all='ignore')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                       Common backtesting signals
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# -----------
#   ma
# -----------
def ma(input_np, ma_win, min_win = None):
    if min_win is None: min_win= ma_win
    return pd.DataFrame(input_np).rolling(window = int(ma_win), min_periods = int(min_win)).mean().values


# -----------
#   min_roll
# -----------
#@jit(nopython = True)
def min_roll(input_np, win, min_wind = 1):
    return pd.DataFrame(input_np).rolling(window = int(win), min_periods = min_wind).min().values

# The following is faster but causes bugs when dealing with nans.
# Needs to be fixed first before being used.
#    out = np.empty_like(input_np) + np.nan
#    for j in range(input_np.shape[1]):
#        out[:, j] = fmu.mv_min(input_np[:, j], win, min_wind)
#    return out

# -----------
#   max_roll
# -----------
#@jit(nopython = True)
def max_roll(input_np, win, min_wind = 1):
    return pd.DataFrame(input_np).rolling(window = int(win), min_periods = min_wind).max().values

#    out = np.empty_like(input_np) + np.nan
#    for j in range(input_np.shape[1]):
#        out[:, j] = fmu.mv_max(input_np[:, j], win, min_wind)
#    return out


# -----------
#   all_roll
# -----------
def all_roll(input_np, all_wid, min_wid = None):
    if min_wid is None: min_wid= all_wid
    return pd.DataFrame(input_np).rolling(window = int(all_wid), min_periods = int(min_wid)).sum().values == all_wid


# --------------
#   sum_roll
# --------------
# Rolling sum of values from the last win data points
def sum_roll(input_np, win, min_win = None):
    if min_win is None: min_win = win
    return pd.DataFrame(input_np).rolling(window = int(min_win), min_periods = int(min_win)).sum().values



# --------
#   ref
# --------
# Data and idx_shift are arrays of the same size
# For each element in data, find an element that is n rows above it, where n is the corresponding element in idx_shift
def ref(data, idx_shift, fill_missing = np.nan):
    if type(idx_shift) == np.ndarray:
        return ref_dynamic(data, idx_shift, fill_missing)
    else:
        return ref_static(data, idx_shift, fill_missing)

# ---------
#   reldiff
# ---------
# Current row / last row - 1    
def reldiff(data):
    ret = data/ref(data,1, fill_missing = np.nan) - 1
    ret[np.isnan(ret)] = 0
    return ret


# ------
# Cross
# ------
# Marks the time when ts1 changse from being less than ts2 to ts1
def cross(ts1, ts2):
    return ref(ts1 < ts2, 1, fill_missing = False) & (ts1 >= ts2)

# ---------
#  isnan
# ---------
# Check if an element in numpy array is a number
def isnan(data):
    return np.isnan(data)


# ---------
#  lnot
# ---------
# Check if an element in numpy array is a number
def lnot(data):
    return np.logical_not(data)

# ---------
#  barslast
# ---------
# Return an array that is the same size bool_mtx
# Each element represents the number of rows since the last true value
def barslast(bool_mtx):
    barslast_np = barslast_numb(bool_mtx)
    return barslast_np.astype(int)
    
@jit(nopython = True)
def barslast_numb(bool_mtx):
    
    time_ct = bool_mtx.shape[0]
    asset_ct = bool_mtx.shape[1]
    
    barslast_np = np.zeros((time_ct,asset_ct)) 
    
    for i in range (1, time_ct):
        for j in range (asset_ct):
            if bool_mtx[i-1,j] == 0:
                barslast_np[i,j] = barslast_np[i-1,j] + 1
                
    return barslast_np





@jit(nopython = True, nogil = True, fastmath = True)
def ref_dynamic(data, idx_shift, fill_missing = np.nan):
    
    time_ct = data.shape[0]
    asset_ct = data.shape[1]
    
    out_mtx = np.full_like(data, fill_missing)

    for j in range(asset_ct):
        for i in range(time_ct):
            row_no = i - idx_shift[i, j]        
            if row_no >= 0:
                out_mtx[i,j] = data[row_no, j]
    
    return out_mtx


def ref_static(data, idx_shift, fill_missing = np.nan):

    out_mtx = np.full_like(data, fill_missing)
    if idx_shift > 0:
        out_mtx[idx_shift:] = data[:-idx_shift]
    elif idx_shift == 0:
        out_mtx = data.copy()
    else:
        # Forward shifting
        out_mtx[:idx_shift] = data[-idx_shift:] 
    
    return out_mtx




@jit(nopython = True)
def dynamic_window_sum(value4sum, win_mtx_np):

    out_mtx = np.full_like(value4sum *1.0, np.nan)

    for j in range(out_mtx.shape[1]):
        prv_sum_lost = True
        for i in range(win_mtx_np.shape[0]):
            if win_mtx_np[i, j] > 0 and i - win_mtx_np[i, j] >= -1:   
                if prv_sum_lost:       

                    out_mtx[i, j] = np.sum(value4sum[i - win_mtx_np[i, j]+1 : i+1, j])
                    prv_sum = out_mtx[i, j]
                    
                else:
                    # Fast method
                    crn_sum = prv_sum + value4sum[i, j]
                    
                    last_wid_start = i-win_mtx_np[i-1, j]
                    wid_adj_len = win_mtx_np[i, j] - (win_mtx_np[i - 1, j] + 1) # If zero then win len increased by 1, no adjustment necessary
                    
                    if wid_adj_len > 0:
                        crn_sum = crn_sum + np.sum(value4sum[last_wid_start - wid_adj_len : last_wid_start, j])
                    elif wid_adj_len < 0:
                        crn_sum = crn_sum + np.sum(value4sum[last_wid_start : last_wid_start - wid_adj_len, j])
                    
                    out_mtx[i, j] = crn_sum
                    prv_sum = crn_sum
                prv_sum_lost = False
            else:
                if win_mtx_np[i, j] == 0:                                     # This is to conform with inherited special definition
                    out_mtx[i, j] = np.sum(value4sum[0 : i+1, j])  
                prv_sum_lost = True
                        
    return out_mtx
    




# -----------
#   std_roll
# -----------
def std_roll(input_np, std_win, cap_value = None, min_win = None):
    if min_win is None: min_win = std_win
    std = pd.DataFrame(input_np).rolling(window = int(std_win), min_periods = int(min_win)).std().values
    if not(cap_value is None):
        std[std < cap_value] = cap_value
    return std



# -----------
#   skew_roll
# -----------
def skew_roll(input_np, skew_wid, min_wid = None):
    if min_wid is None: min_wid = skew_wid
    return pd.DataFrame(input_np).rolling(window = int(skew_wid), min_periods = int(min_wid)).skew().values


# -----------
#  kurt_roll
# -----------
def kurt_roll(input_np, kurt_wid, min_wid = None):
    if min_wid is None: min_wid = kurt_wid
    return pd.DataFrame(input_np).rolling(window = int(kurt_wid), min_periods = int(min_wid)).kurt().values


# --------------
#   count_since
# --------------
# Since x have occured, how many times have y occured
def count_since(x, y):
    return count(x, barslast(y))

# --------------
#  occur number
# --------------
# Since x have occured, if y occured today, how many times is this y
def occur_number(x, y):
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            a = 1
    return 1


# -------------
#   barpos
# -------------
# Time since last invalid data point
@jit(nopython = True)
def barpos(data):
    
    age = np.zeros_like(data)
    row_ct = data.shape[0]
    for j in range(data.shape[1]):
        for i in range(row_ct):
            if not(np.isnan(data[i,j])):
                age[i:, j] = np.linspace(1, row_ct - i, row_ct - i)
                break
    return age

# -------------
#  rowwise_pct
# -------------
# Find the rowwise percentile of data_np
def rowwise_pct(data_np):
    
    rank_loc = np.argsort(data_np, 1)     
    rank = rank_loc2rank(rank_loc)    
    valid_data_ct = np.sum(~np.isnan(data_np), 1)           
    pct = rank / np.expand_dims(valid_data_ct - 1, 1)
    pct[np.isnan(data_np)] = np.nan
    
    return pct



# -------------
#  rank_loc2rank
# -------------
# rank_loc stores the location of value of each rank
# This function converts it to the rank of each location
    
def rank_loc2rank(rank_loc):
    rank = rank_loc.copy()
    ordered_array = np.arange(rank_loc.shape[1])
    for row_no in range(rank_loc.shape[0]):
        rank[row_no, rank_loc[row_no, :]] = ordered_array
    return rank


# -------------
#  reduce_freq
# -------------
# Reduce the frequency which data_np changes
#@jit(nopython = True)
# !!! 想办法恢复这个函数使用git的机会
def reduce_freq(data_np, adj_freq = None, start_cycle_no = 1, trade_loc = []):
    
    # 计算交易日期
    if trade_loc == []:
        trade_loc= gen_even_trade_loc(data_np.shape[0], adj_freq, start_cycle_no)

    if adj_freq == 1:
        return data_np.copy()
    else:
        data_out_np = data_np.copy()
        for row_no in range(1, data_out_np.shape[0]):
            if not(trade_loc[row_no, 0]):
                data_out_np[row_no,:] = data_out_np[row_no - 1,:]

        return data_out_np
    
# --------------------
#  reduce_freq_frac
# --------------------
# Reduce the frequency which data_np changes  
# Do it on a rolling basis. Only change a frac each time.
def reduce_freq_frac(data_np, adj_freq = None, frac_ct = 1):
    
    frac_ct = min(adj_freq, frac_ct)
    
    incre = (adj_freq - 1)/ frac_ct
    for cycle_no in range(frac_ct):
        start_cycle_no = int(1 + round(cycle_no * incre, 0))
        
        sampled_data = reduce_freq(data_np, adj_freq = adj_freq, start_cycle_no = start_cycle_no)
        if cycle_no == 0:
            data_out = sampled_data
        else:
            data_out = data_out + sampled_data
            
    data_out = data_out / frac_ct
    
    return data_out
    

# ----------------------
#    is_recent_report
# ----------------------
# 判断表中数据是否最新
# ！！只接受excel中的数值格式日期：起点1900/1/1(值为1)，之后每天+1
# 目前支持的规则如下：
#   |   rule    |           功能
# 1 | 'annual'  |  判断数据是否为去年的年报

def is_recent_report(report_date_as_nums, date_list, rule = 'annual'):
    dt = pd.to_datetime(date_list)            # 将日期列表转为DatetimeIndex格式    
    dt_nums = np.zeros(len(date_list))

    if rule == 'annual':
        # 判断要计算的年份范围
        year_min = dt[0].year
        year_max = dt[-1].year
        
        # 判断每个日期所对应的上一年年末，并将日期转为数字格式
        for year in range(year_min, year_max + 1):
            dt_nums[dt.year == year] = pd.to_numeric(pd.to_datetime([str(year - 1) + '/12/31']))
        dt_nums = dt_nums / 864 / 10 ** 11 + 25569
        return (report_date_as_nums.T == dt_nums).T
        
    else:
        raise Exception('rule not recognized: ' + rule)


# ---------------------
#    gen_loc_by_rule
# ---------------------
# 读取日期列表date_list，按规则找到列表中特定的日期
# 目前支持的规则如下：
#   |  TAG   |            EXAMPLE            |              功能
# 1 |  'yy'  |    rule = 'yy/5/15', n = 1    |   找到每年特定日期后的第n个交易日
# 2 |  'mm'  |     rule = 'mm/1', n = 1      |   找到每月特定日期后的第n个交易日
# 3 |  'ww'  |     rule = 'ww4-3', n = 1     |  找到每月某周第某天后的第n个交易日
# 4 |        |  rule = [rule1, ... , ruleN]  |            多条规则
#   |        |  rule = rule1 + ... + ruleN   |

# 说明：
# 1 | 可以混合输入多条规则了，输入格式支持string或list
# 2 | 某日后的第n个交易日是从当天开始数的，如果当天是交易日，则n = 1就代表当天
# 3 | logic_and = False 或逻辑; logic_and = True 与逻辑

def gen_loc_by_rule(date_list, rule = 'yy/7/1', n = 1, logic_and = False):
    rule = ''.join(rule)
    loc_list = np.zeros(len(date_list))
    opr_ct = 0
    dt = pd.to_datetime(date_list)            # 将日期列表转为DatetimeIndex格式
    
    # 判断首条规则的种类
    letters = re.findall(re.compile(r'[a-z]', re.S), rule)
    tag = letters[0] + letters[1]
    tag_idx = rule.index(tag) + len(tag)
    rest_letters = letters[len(tag): ]
        
    # 循环按规则寻找指定的日期
    while len(rest_letters):
        target_date = ' ' + rule[tag_idx: tag_idx + rule[tag_idx: ].index(rest_letters[0])]
        loc_list = loc_list + gen_loc_after_date(dt, target_date, n = n, tag = tag)
        opr_ct += 1
        tag = rest_letters[0] + rest_letters[1]
        tag_idx = tag_idx + rule[tag_idx: ].index(tag) + len(tag)
        rest_letters = rest_letters[len(tag): ]
    
    # 读取字符串的最后一段
    target_date = ' ' + rule[tag_idx: ]
    loc_list = loc_list + gen_loc_after_date(dt, target_date, n = n, tag = tag)
    opr_ct += 1

    if logic_and: return (loc_list == opr_ct) * 1
    else: return (loc_list > 0) * 1
    

# ------------------------
#    gen_loc_after_date
# ------------------------
# 工具函数，取每个周期内特定日期后(含本身)的第n个交易日
# 参数的含义见gen_loc_by_rule

def gen_loc_after_date(dt, target_date, n, tag):
    loc_list = np.zeros(len(dt))
    
    if tag == 'yy':
        # 判断要计算的年份范围
        year_min = dt[0].year
        year_max = dt[-1].year
        
        # 找到每年不早于指定日期的第n个交易日
        for year in range(year_min + 1, year_max):
            loc_list[np.argmin(dt < pd.to_datetime(str(year) + target_date)) + n - 1] = 1
          
        # 第一年的数据单独处理 
        if dt[0] <= pd.to_datetime(str(year_min) + target_date):
            loc_list[np.argmin(dt < pd.to_datetime(str(year_min) + target_date)) + n - 1] = 1
        
        # 最后一年的数据单独处理 
        last_loc = np.argmin(dt < pd.to_datetime(str(year_max) + target_date))
        if last_loc + n - 1 < len(dt):
            if dt[last_loc] >= pd.to_datetime(str(year_max) + target_date):
                loc_list[last_loc + n - 1] = 1
                
    elif tag == 'mm':
        # 判断要计算的年份范围
        year_min = dt[0].year
        year_max = dt[-1].year
        
        # 找到每年不早于指定日期的第n个交易日
        for year in range(year_min + 1, year_max):
            for month in range(1, 13):
                loc_list[np.argmin(dt < pd.to_datetime(str(year) + ' ' + 
                                                       str(month) + target_date)) + n - 1] = 1
          
        # 第一年的数据单独处理    
        for month in range(dt[0].month + 1, 13):
            loc_list[np.argmin(dt < pd.to_datetime(str(year_min) + ' ' + 
                                                   str(month) + target_date)) + n - 1] = 1
        if dt[0] <= pd.to_datetime(str(year_min) + ' ' + 
                                                   str(dt[0].month) + target_date):
            loc_list[np.argmin(dt < pd.to_datetime(str(year_min) + ' ' + 
                                                   str(dt[0].month) + target_date))] = 1
        
        # 最后一年的数据单独处理     
        for month in range(1, dt[-1].month):
            loc_list[np.argmin(dt < pd.to_datetime(str(year_max) + ' ' 
                                                   + str(month) + target_date)) + n - 1] = 1
        last_loc = np.argmin(dt < pd.to_datetime(str(year_max) + ' ' + str(dt[-1].month) + target_date))
        if last_loc + n - 1 < len(dt):
            if dt[last_loc] >= pd.to_datetime(str(year_max) + ' ' + str(dt[-1].month) + target_date):
                loc_list[last_loc + n - 1] = 1 
    
    elif tag == 'ww':
        # 判断要计算第几周的周几
        week_no = int(target_date[: target_date.index('-')])
        day_no = int(target_date[target_date.index('-') + 1: target_date.index('-') + 2])
        week_of_month = get_week_of_month(dt)
        loc_list_1 = (week_of_month >= week_no) & (dt.dayofweek >= day_no)
        
        # 判断要计算的年份范围
        year_min = dt[0].year
        year_max = dt[-1].year
        
        # 找到每年不早于指定日期的第n个交易日
        for year in range(year_min + 1, year_max):
            for month in range(1, 13):
                loc_list[np.argmax((dt.year >= year) & (dt.month >= month) & loc_list_1) + n - 1] = 1
        
        # 第一年的数据单独处理    
        for month in range(dt[0].month + 1, 13):
            loc_list[np.argmax((dt.year >= year_min) & (dt.month >= month) & loc_list_1) + n - 1] = 1 
        if not loc_list_1[0]:
            loc_list[np.argmax((dt.year >= year_min) & (dt.month >= dt[0].month) & loc_list_1) + n - 1] = 1
            
        # 最后一年的数据单独处理   
        for month in range(1, dt[-1].month):
            loc_list[np.argmax((dt.year >= year_max) & (dt.month >= month) & loc_list_1) + n - 1] = 1 
        last_loc = np.argmax((dt.year >= year_max) & (dt.month >= dt[-1].month) & loc_list_1)
        if last_loc + n - 1 < len(dt):
            if (dt[last_loc].year >= year_max) & (dt.month[last_loc] >= month) & loc_list_1[last_loc]:
                loc_list[last_loc + n - 1] = 1 
       
    else:
        raise Exception('gen_loc_after_date: tag not recognized: ' + tag)   
       
    return loc_list

# -----------
#   ma
# -----------
def ma(input_np, ma_win, min_win = None):
    if min_win is None: min_win= ma_win
    return pd.DataFrame(input_np).rolling(window = int(ma_win), min_periods = int(min_win)).mean().values


# ------------------------
#    get_week_of_month
# ------------------------
# 通用的工具函数，判断一个Datetime列表中的每天属于当月的周几
# 可能需要一种更快的算法

def get_week_of_month(dt):
    week_of_month = np.zeros(len(dt))
    for i in range(len(dt)):
        first_day_of_month = pd.to_datetime(str(dt[i].year) + '/' + str(dt[i].month) + '/1')
        week_of_month[i] = int(dt[i].strftime("%W")) - int(first_day_of_month.strftime("%W")) + 1
    return week_of_month


# ------------------------
#     SimpleRegression
# ------------------------
class SimpleRegression():
    """
    Provides some regression methods.
    
    Parameters
    ----------
    method : string
             The regression method.
    
    n_seg : int
            Number of segments. Default = 5.
    
    poly_degree : int
                  Degree of the polynomial features. Default = 3.
    """
    def __init__(self, method = 'seg', n_seg = 5, poly_degree = 3):

        self.method = method
        if method == 'seg': self.n_seg = n_seg
        elif method == 'poly': self.poly_degree = poly_degree
        else: raise Exception('Method not recognized: ' + method)    
    
    
    def fit(self, x, y):
        """
        Fit model.

        Parameters
        ----------
        x : 1D array 
            Training data.
        y : 1D array 
            Target values.

        Returns
        -------
        self: An instance of self.
        """
        self.check_input([x, y])
        
        if self.method == 'seg':
            # Divide the curve into segments of equal length.
            x_min = np.nanmin(x)
            x_max = np.nanmax(x)
            self.seg_list = [x_min - sys.float_info.epsilon]
            for i in range(self.n_seg):
                self.seg_list.append(x_min / self.n_seg * (self.n_seg - i - 1) + 
                                     x_max / self.n_seg * (i + 1))
        
            # For each segment, caculate the mean of target values.
            self.coeff_list = []
            for i in range(len(self.seg_list) - 1):
                seg_loc = (x > self.seg_list[i]) & (x <= self.seg_list[i + 1])
                self.coeff_list.append(np.nanmean(y[seg_loc]))
         
        elif self.method == 'poly':
            # Polynomial fitting.
            bad = (x != x) | (y != y)
            x = x[~ bad].reshape(-1, 1)
            y = y[~ bad]
            x_poly = PolynomialFeatures(degree = self.poly_degree).fit_transform(x)
            self.coeff_list = LinearRegression().fit(x_poly, y)
       
        return self
        
    
    def predict(self, x): 
        """
        Predict using the model.
        
        Parameters
        ----------
        x : 1D array
            Samples.

        Returns
        -------
        y_hat: 1D array
               Predicted values
        """
        self.check_is_fitted()
        self.check_input([x])
        
        # Get the fitted model, and make prediction.
        if self.method == 'seg':
            y_hat = x.copy() + np.nan
            for i in range(len(self.seg_list) - 1):
                seg_loc = (x > self.seg_list[i]) & (x <= self.seg_list[i + 1])
                y_hat[seg_loc] = self.coeff_list[i]
        
        elif self.method == 'poly':
            y_hat = x.copy() + np.nan
            bad = x != x
            x = x[~ bad].reshape(-1, 1)
            x_poly = PolynomialFeatures(degree = self.poly_degree).fit_transform(x)
            y_hat[~ bad] = self.coeff_list.predict(x_poly)
        
        return y_hat
            
    
    # The model must be fitted before predicting.
    def check_is_fitted(self):
        if not hasattr(self, 'coeff_list'):
            raise Exception("This SimpleRegression instance is not fitted yet. Call 'fit' "
                            "with appropriate arguments before using this estimator.")
        
        
    # Check the inputted objects.
    def check_input(self, input_list):
        # Inputted objects must be 1D arrays.
        len_list = []
        for x in input_list:            
            if not type(x) is np.ndarray:
                raise ValueError("Expected array, got " + str(type(x)) + " instesd: \n" 
                                 + str(x) + ".")
            if (len(x.shape) > 1) & (min(x.shape) > 1):
                raise ValueError("Expected 1D array, got 2D array instesd: \n" 
                                 + str(x) + ".\nTry array.reshape(-1, 1) or array.reshape(1, -1).")           
            len_list.append(len(x))
            
        # Check whether all objects have the same length.
        if not len(set(len_list)) == 1:
            raise ValueError("Found input variables with inconsistent numbers of "
                             "samples: " + str(len_list) + ".")
            

# --------------------
#  gen_even_trade_loc
# --------------------
# 如果需要每隔  adj_freq 才交易一次，标记交易时间
# 输入参数：
# time_len： 总共有多少个交易日
# start_cycle_no： 是否推迟第一次交易的开始。1=不推迟
# 输出：
# 一个 time_len * 1 的矩阵。 True表示当日有交易
@jit(nopython = True)      
def gen_even_trade_loc(time_len, adj_freq, start_cycle_no = 1):
    start_cycle_no = min(start_cycle_no, adj_freq)
    out_arr = np.zeros((time_len, 1)) > 0
    out_arr[start_cycle_no - 1::adj_freq] = True

    return out_arr
    


# ---------
#  avg_pos
# ---------
# Create an average position history that takes into account time varying data
# validity

def avg_pos(valid_data_hist, normalize = True):
    avg_pos = np.zeros_like(valid_data_hist)
    avg_pos[np.logical_not(np.isnan(valid_data_hist))] = 1
    if normalize:
        avg_pos, stock_at_avg = normalize_pos(avg_pos)
    return avg_pos

# ---------------
#    replace
# ---------------    
# Replace values in a matrix
    
def replace(data_np, old_value, new_value, return_copy = True):
    if return_copy:
        return_data = data_np.copy()
    else:
        return_data = data_np
    
    if type(old_value) is np.ndarray:
        return_data[old_value] = new_value
    else:
        if fmu.isnan(old_value):
            return_data[np.isnan(return_data)] = new_value
        else:
            return_data[return_data == old_value] = new_value
        
    return return_data

# ---------------
#   replace_loc
# ---------------
# Replace values in a matrix at given locations    
def replace_loc(data_np, loc, new_value):
    return_data = data_np.copy() * 1.0
    return_data[loc] = new_value
    return return_data

# ------------
# pause_signal
# ------------
# On days marked by pause the position does not change
def pause_signal(signal_pos, pause):
    pause = (pause == 1)
    signal_pos_copy = signal_pos.copy()
    signal_pos_copy[0, pause[0, :]] = 0
    for i in range(1, pause.shape[0]):
        signal_pos_copy[i, pause[i, :]] = signal_pos_copy[i-1, pause[i, :]] 
    return signal_pos_copy





# size = (1000, 5000)

# a = np.random.random(size)
# bad = (np.random.random(size) < 0.5) 
# a[bad] = np.nan

# no_trade  = (np.random.random(size) < 0.1) | (barpos(a) <= 1)
# no_buy  = (np.random.random(size) < 0.1) | ((barpos(a) <= 1) & (a > 0))
# no_sell  = (np.random.random(size) < 0.1) | ((barpos(a) <= 1) & (a < 0))


# b = pause_signal_dir_norm(a, no_buy, no_sell, no_trade)

# print(np.nanmin(np.nanmin(b, 1)), np.nanmax(np.nanmax(b, 1)))
# print(np.nanmin(np.nansum(b,1)), np.nanmax(np.nansum(b,1)))

# trade = b.copy()
# trade[np.isnan(b)] = 0
# trade[1:,:] = b[1:,:] - b[:-1,:]
 
# print(np.nanmin(trade[no_trade]), np.nanmax(trade[no_trade]))
# print(np.nanmin(trade[no_buy]), np.nanmax(trade[no_buy]))
# print(np.nanmin(trade[no_sell]), np.nanmax(trade[no_sell]))


# ------------------
# apply_across_time
# ------------------
# Repeat a horizontal vector vertically
     
def apply_across_time(data, value):
    row_ct = data.shape[0]
    return np.repeat(np.expand_dims(value, axis = 0), row_ct, axis = 0)
    

# --------------
#    top_n
# --------------
# Return the top n scores
# Whose marker on subset is True
    
def top_n(score, subset, n):
    
    score_subset = score.copy() * 1.0
    score_subset[lnot(subset)] = np.nan
    
    rank_loc = np.argsort(score_subset, 1) 
    
    rank = (rank_loc2rank(rank_loc) + 1) * 1.0
    
    rank[lnot(subset)] = np.nan
    
    return rank <= n

# --------------------------
#   apply_across_asset
# --------------------------    
# Repeat an verical asset horizontally
def apply_across_asset(data, value):
    col_ct = data.shape[1]
    return np.repeat(value, col_ct, axis = 1)
    
# ---------
#   diff
# ---------
# Current row minues last row
def diff(data):
    ret = data - ref(data, 1, fill_missing = 0)
    ret[np.isnan(ret)] = 0
    return ret

# ---------
#   reldiff
# ---------
# Current row / last row - 1    
def reldiff(data):
    ret = data/ref(data,1, fill_missing = np.nan) - 1
    ret[np.isnan(ret)] = 0
    return ret
    

# -------------
#   norm_fac
# -------------
# Map the each row of the matrix to between 0 and 1, 0 being the original min 
# value and 1 being the original max value

def norm_fac(fac):

    min_rows = np.nanmin(fac, axis = 1)
    max_rows = np.nanmax(fac, axis = 1)
    range_rows = max_rows - min_rows
    
    fac_norm = fac.copy()
    for i in range(fac.shape[0]):
        fac_norm[i,:] = (fac[i, :] - min_rows[i]) / range_rows[i]
        
    return fac_norm


# -------------
#    res_fac
# -------------
# Make a new factor orthorgonal to existing factors
# Input: new_fac: a matrix of factor
#        basis_list: a list of existing factors

def res_fac(basis_list, new_fac):

    def res_fac_one_row(basis_list, new_fac, row):
            
        x_list = []
        for basis in basis_list:
            x_list.append(basis[row])
               
        x = np.array(x_list)
        y = new_fac[row, :]
        
        x = np.transpose(x)
        y = y.reshape(-1, 1)
        
        # Valid data
        valid = np.logical_not(np.isnan(np.sum(x, axis = 1) + np.squeeze(y)))
    
        xv = x[valid]
        yv = y[valid]
        res = y.copy() + np.nan
        
        res[valid] = get_res(yv, xv)
        
        return res

    res = new_fac.copy() + np.nan  
    
    for i in range(new_fac.shape[0]):
        res[i, :] = np.squeeze(res_fac_one_row(basis_list, new_fac, i))
        
    return res



# -------------
#    get_res
# -------------
# Get the residuals of y regression x
def get_res(y, x):
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    return y - np.matmul(x, coef)
   
    
# -------------
#    get_beta
# -------------
# Get the residuals of y regression x
def get_beta(y, x):
    try:
        a = np.linalg.lstsq(x, y, rcond=None)[0]
        return np.linalg.lstsq(x, y, rcond=None)[0][0]
    except:
        return np.array([np.nan for _ in range(x.shape[1])])


# --------------
#   fac2pos
# --------------
# Calculates position from a factor
def fac2pos(factor, not_selected, lb, ub, trade_freq, short = True):

    factor_valid = replace_loc(factor, not_selected, np.nan)                   #无效区间的因子为nan, 无排名
    pct = rowwise_pct(factor_valid)                                            # 对于每一个Bar, 计算出当时的横向排名百分位
    long_pos = ((pct <=ub) & (pct >=lb)) *1.0                           
    
    if short:
        lb_opp = 1 - ub
        ub_opp = 1 - lb
        
        short_pos = ((pct <= ub_opp) & (pct >= lb_opp)) * 1.0     
    else:
        short_pos = 0

    # 降低调仓频率
    raw_pos = reduce_freq((long_pos - short_pos), trade_freq)                                  # 每隔250个Bar才调整一次仓位
    
    signal_pos = replace(raw_pos, not_selected, 0)
                  
    return signal_pos

# ------------------
#   corr_key_values
# ------------------
# Returns 3 intermdiate variabels that will be used to calculate 
# Rollingregression and correlation
def corr_key_values(data_mtx,index):
    xy=np.multiply(data_mtx,index)
    x_2=np.multiply(data_mtx,data_mtx)
    y_2=np.multiply(index,index)
    return xy,x_2,y_2




# --------------
#    corr
# --------------
# Pairwise correlation of x1 aganist x2, using window length win
def corr(x1,x2,win):
    return corr_roll(x1,x2,win)


# --------------
#   corr_roll
# --------------
# Rolling correlation of each column of data_mtx aganist the vector named index
# Using a window length measured by win
    
def corr_roll(data_mtx,index,win, min_wid = None):
    xy,x_2,y_2=corr_key_values(data_mtx,index)
    x= ma(data_mtx, win)
    y= ma(index, win)
    t1= ma(xy, win)-np.multiply(x,y)
    t2= ma(x_2, win)-np.multiply(x,x)
    return t1/t2


# --------------
#   regress_roll
# --------------
# Rolling regression of each column of data_mtx aganist the vector named index
# Using a window length measured by win
# Returns the beta of regression results    
def regress_roll(data_mtx,index,win):
    xy,x_2,y_2=corr_key_values(data_mtx,index)
    x=ma(data_mtx, win, min_wid = None)
    y=ma(index, win, min_wid = None)
    t1=ma(xy, win, min_wid = None)-np.multiply(x,y)
    t2=ma(y_2, win, min_wid = None)-np.multiply(y,y)
    return t1/t2

# --------------
#   ffill
# --------------
# Replace nans in data with last known value
def ffill(data):
    return pd.DataFrame(data).ffill().values


# --------------
#   rank_select
# --------------
# For each row in the matrix factor, mark values whose row-wise percentile is between lb and ub 
# Ingore values that is marked false in the matrix valid region
def rank_select(factor, valid_region, lb, ub, ret_val = "long_pos", valid_region2 = None):
    
    if type(valid_region) is list:
    
        # A list of valid region is provided( for example in the case of multiple sectors)
        # In this case we neet to iterate through all filters
        for i, valid_region_i in enumerate(valid_region):
            pos_i = rank_select(factor, valid_region_i, lb, ub, ret_val = ret_val, valid_region2 = valid_region2)
            if i == 0:
                pos = pos_i
            else:
                pos = pos + pos_i
        
        return pos
    
    else:
        
        if not(valid_region2 is None):
            valid_region = (valid_region == 1) & (valid_region2 == 1)

        not_selected = lnot(valid_region)     
        factor_valid = replace_loc(factor, not_selected, np.nan)                   # 无效区间的因子为nan, 无排名
        ma_ret_pct = rowwise_pct(factor_valid)                                     # 对于每一个Bar, 计算出当时的横向排名百分位
        long_pos = ((ma_ret_pct <=ub) & (ma_ret_pct >=lb)) *1.0                           
        if ret_val == "long_pos":        
            return long_pos                           
        else:
            raise Exception("The return value you requested " + str(ret_val) + " is not supported")




# -------------
#  calc_age
# -------------
# is_member is a t by n matrix that uses 1 to mark whether an asset belongs to a 
# certain group at any given time.
# Returns another matrix that marks how long has an asset been a member
# Negative numbers shows how long agao has the asset been kicked out.
# Each entry-exit action resets the age.

def calc_age(is_member):
    
    out_mtx = np.zeros(is_member.shape)
    
    for row_no in range(1, out_mtx.shape[0]):
            
        prv_row = is_member[row_no - 1, :]
        crn_row = is_member[row_no, :]
        
        
        left = prv_row & ~(crn_row)
        join = ~(prv_row) & crn_row
        
        reset = left | join
        
        inc = prv_row & crn_row
        des = ~(prv_row) & ~(crn_row)
            
        out_mtx[row_no, reset] = 0
        out_mtx[row_no, inc] = out_mtx[row_no - 1, inc] + 1
        out_mtx[row_no, des] = out_mtx[row_no - 1, des] - 1
        
    return out_mtx  


# ----------------------
#    calc_group_avg
# ----------------------
# to replace the score of factor for each stock with the score of factor for 
# the group in which the stock belong.
    
def calc_group_avg(data_np, group_hist):
    
    group_list_raw = np.unique(group_hist)
    group_list = group_list_raw[np.logical_not(np.isnan(group_list_raw))]
    group_list = group_list[group_list != 0]
    
    time_ct = data_np.shape[0]
    group_ct = len(group_list)
    asset_ct = data_np.shape[1]

    group_avg = np.zeros((time_ct, group_ct))  # Pre-allocate for stroage

    for group_no, group_name in enumerate(group_list):
        
        loc = group_hist == group_name
        
        data_np_this_group = np.zeros((time_ct, asset_ct)) + np.nan
        data_np_this_group[loc] = data_np[loc]
        
        group_avg[:, group_no] = np.nanmean(data_np_this_group, axis = 1) 

            
    return group_avg


# ---------------
#    pos2nav
# ----------------
# Calculates a portofolio nav history using position history and price history 
# Of the traded asset.
# Used for cases where we need to quickly evaluate the signal
    
def pos2nav(pos, price):
    port_ret = np.nansum(reldiff(price) * ref(pos, 1), axis = 1)
    return np.expand_dims(np.cumprod(port_ret + 1), 1)

# ---------------
#    ma_z
# ----------------
# Rolling z score
    
def z_roll(input_np, wind):
    avg = ma(input_np, wind)
    std = std_roll(input_np, wind)
    z = (input_np - avg)/std
    return z



# -----------------------
#    add_alpha_non_neg
# -----------------------
# Add alpha to a set of base weights such that the final weight remains non-negative
    

def add_alpha_non_neg(base, alpha):
    
    alpha_neg = alpha.copy()
    alpha_neg[alpha >=0 ] = np.nan
    
    max_alpha_weight = -np.expand_dims(np.nanmax(base / alpha_neg, 1), 1)
    
    new_pos = base + max_alpha_weight * alpha
    
    return new_pos
    
    