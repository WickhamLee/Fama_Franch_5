import pandas as pd
import numpy as np
import math


# 待开发项目：
# !!! 写注释
# !!! 改的方便使用

def get_max_dd_rec_time(nav):
    crn_max_lv = nav[0]
    max_dd = 0

    crn_rec_time = 0
    max_rec_time = 0
    for i in range(nav.shape[0]):

        if nav[i] < crn_max_lv:
            crn_rec_time = crn_rec_time + 1
        else:
            crn_rec_time = 0  # 重设回撤修复周期
            crn_max_lv = nav[i]

        max_rec_time = max(max_rec_time, crn_rec_time)
        crn_dd = crn_max_lv/nav[i] - 1
        max_dd = max(crn_dd, max_dd)

    max_dd = max_dd

    return max_dd, max_rec_time

def get_max_dd(nav):
    return np.max(np.maximum.accumulate(nav)/nav-1)


def get_ann_fac(dt):
    
    if type(dt) is pd.core.indexes.datetimes.DatetimeIndex:
        dt = dt.values
    
    days_range = (dt[len(dt) - 1] - dt[0]).astype('timedelta64[D]').astype(int)
    ann_fac = len(dt) / (days_range / 365.25)
    
    return ann_fac

def get_win_rate(ret_ts):
    win_rate = sum(ret_ts >= 0) / (len(ret_ts) + 1)
    return win_rate


def get_avg_ret(ret_ts, ann_fac = 250):
    return np.mean(ret_ts) * ann_fac

def get_vol(ret_ts, ann_fac = 250):
    return np.std(ret_ts) * math.sqrt(ann_fac)

def get_sharpe(ann_ret, ann_vol):
    sharpe = ann_ret / ann_vol
    return sharpe

def get_ret_mdd_ratio(ann_ret, max_dd):
    return ann_ret / max_dd

def get_ts_range(dt, ret_or_nav_ts = []):
    if dt == []:

        starttime = "未知"
        endtime = "未知"
        days_covered = "未知"
        data_count = len(ret_or_nav_ts)

    else:

        starttime = min(dt)
        endtime = max(dt)
        days_covered = (dt[len(dt) - 1] - dt[0]).astype('timedelta64[D]').astype(int)
        data_count = len(dt) + 1

    return days_covered, starttime, endtime, data_count


def get_asset_perm(ret_or_nav_ts, dt = [], ann_fac = [], print_result = False, input_is_return = False):

    if type(dt) is pd.core.indexes.datetimes.DatetimeIndex:
        dt = dt.values
        
    if dt == [] and ann_fac == []:
        ann_fac = 250
    elif dt != [] and ann_fac == []:
        ann_fac = get_ann_fac(dt)

    if input_is_return:
        nav_ts = np.cumprod(ret_or_nav_ts + 1)
        ret_ts = ret_or_nav_ts
    else:
        ret_ts = ret_or_nav_ts[1:] / ret_or_nav_ts[0:-1] - 1
        nav_ts = ret_or_nav_ts


    ann_ret = get_avg_ret(ret_ts, ann_fac)
    ann_vol = get_vol(ret_ts, ann_fac)
    sharpe = get_sharpe(ann_ret, ann_vol)
    max_dd, max_rec_time = get_max_dd_rec_time(nav_ts)
    ret_mdd_ratio = get_ret_mdd_ratio(ann_ret, max_dd)
    win_rate = get_win_rate(ret_ts)
    days_covered, starttime, endtime, data_count = get_ts_range(dt, ret_or_nav_ts=ret_or_nav_ts)


    perm = {"年化收益": ann_ret,
        "波动率": ann_vol,
        "夏普": sharpe,
        "最大回撤" : max_dd,
        "收益回撤比": ret_mdd_ratio,
        "最长回复修复时间": max_rec_time,
        "胜率": win_rate ,
        "数据密度（数据点/年)": ann_fac,
        "数据数量": data_count,
        "开始": starttime,
        "结束": endtime,
        "时长(天）": days_covered}

    if print_result:
        for item in perm:
            print(item + ':  ' + str(perm[item]))

    return perm

# 计算移动的最大回撤
# ！！！ 没写完！
def calc_rolling_dd(lv, window, min_periods = 'None'):
    for i in range(len(lv.shpae[0])):
        rolling_dd = 1
    return rolling_dd




# -----------------------------
# 将净值时间序列转换成收益时间序列
# -----------------------------

def nav2ret(nav_ts, df_nav_col=[], first_row_value = 0):

    if type(nav_ts) is np.ndarray:

        nav_np = nav_ts
        ret_np = np.zeros(nav_np.shape) + first_row_value

        if len(nav_np.shape) > 1:
            ret_np[1:, :] = nav_np[1:, :] / nav_np[0:-1, :] - 1
        else:
            ret_np[1:] = nav_np[1:] / nav_np[0:-1] - 1

        return ret_np

    elif type(nav_ts) is pd.core.frame.DataFrame:

        if df_nav_col == []:
            df_nav_col = range(0, len(nav_ts.columns))
        nav_np = nav_ts.iloc[:, df_nav_col].values
        ret_np = nav2ret(nav_np)
        out_df = nav_ts.copy()
        out_df.iloc[:, df_nav_col] = ret_np

        return out_df

    else:
        raise Exception('Unsupported data type: ' + type(nav_ts))





