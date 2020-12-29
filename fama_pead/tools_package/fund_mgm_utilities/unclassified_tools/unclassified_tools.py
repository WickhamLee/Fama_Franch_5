
# 用来放置一些乱七八糟但是又不知道该放哪里的小工具
# 主要用来解决python中一些烦人的小问题

import math
import datetime
import sys
import collections
import numpy as np
import pandas as pd
import fund_mgm_utilities as fmu

# -------
# isnan
# -------

# 利用pandas.read_csv(open(file_name)) 去读的表格中，如果内容是空的，则会被用nan填充。
# 要判断一个cell是否是nan, 需要先检查type是不是数字，否则直接检查会报错
# 这里将所需要做的判断汇集到一个函数上

def isnan(cell_content):

    comparable_type_list = ['float', 'float64']

    return_value = False
    if type(cell_content).__name__ in comparable_type_list:
        if math.isnan(cell_content):
            return_value = True

    return return_value

# -------
#  now
# -------
# 有时候想打印个时间太麻烦了，需要一长串代码
def now(message = '', convert_to_str = True, time_format = '%Y-%m-%d %H:%M:%S'):

    time_now =  datetime.datetime.now()
    if convert_to_str:
        time_now = message +time_now.strftime(time_format)
    return time_now

# -------
#  today
# -------
# 有时候想打印个今天的日期太麻烦了，需要一长串代码
def today(message = '', convert_to_str = True, str_format = '%Y-%m-%d'):

    date_today =  datetime.datetime.now()
    date_today = date_today.replace(hour = 0, minute = 0, second = 0, microsecond = 0)

    if convert_to_str:
        date_today = message + date_today.strftime(str_format)
    return date_today




# ----------
#  eva_dict
# -----------
# 返回一个list, 里面包含了可以exce的string, 这些string会dict里的每一个变量变成dict scope以外的变量
# 例如： input_dict = dict{'a': 1}
# 该代码将返回： ['a = 1']

def eva_dict(input_dict):
    out_list = []
    for key in input_dict:
        try:
            value_str = key + '=' + str(input_dict[key])
            out_list.append(value_str)
        except:
            data_type =  type(input_dict[key])
            data_type = data_type.__name__
            warnings.warn('eva_dict: 该key的值无法被转换成String： ' + key + '。其数据类型为：' + data_type + '. 该值将无法被evaluate')
    return out_list


# ----------------------
#    将dict转行成str
# ----------------------
# 这样dict就可以在一行内打印完
def dict2str(input_dict, fields_to_disp = None, splitter = ":", field_sep = ' '):
    disp_str = ''
    if fields_to_disp == None:
        fields_to_disp = input_dict.keys()

    i = 0
    for field in fields_to_disp:
        new_str = field + splitter + str(input_dict[field])
        if i == 0:
            disp_str = new_str
        else:
            disp_str = disp_str + field_sep + new_str
        i = i + 1
    return disp_str

# ----------------------
#      p_dict
# ----------------------
# 逐行打印一个dictionary的值
def p_dict(input_dict, indend = ""):
    for key in input_dict.keys():
        if type(input_dict[key]) is dict:
            print(indend + key + ":")
            p_dict(input_dict[key], indend + " ")
        else:
            print(indend + key + ":", input_dict[key])
            
            
# ----------------------
#      p_list
# ----------------------
# 逐行打印一个dictionary的值
def p_list(input_list, indend = ""):
    for line in input_list: print(indend + " " + str(line))
            

# ----------------------
#    将list转行成str
# ----------------------
# 这样dict就可以在一行内打印完
def list2str(input_list, splitter = ":"):
    disp_str = ''
    for i, item in enumerate(input_list):
        if i == 0:
            disp_str = item
        else:
            disp_str = disp_str + splitter + item

    return disp_str


# -------------------------
#   返回当前函数的名字
# -------------------------
# scope_level: 1 = 当前函数
#              2 = 上一层函数
def get_func_name(scope_level = 1):
    return sys._getframe(scope_level).f_code.co_name


# -----------------------------------------------------------
#   返回一个dataframe, 含有各个元素的出现频率,按从高往低排序
# ----------------------------------------------------------
def tabulate(data_np):
    counter = collections.Counter(data_np)
    freq_np = np.array(list(dict(counter).values()))
    unique_np = np.array(list(dict(counter).keys()))

    table_dict = {'value': unique_np, 'freq': freq_np}
    table_df = pd.DataFrame(table_dict)
    table_df = table_df[['value', 'freq']]
    table_df.sort_values(by = 'freq', ascending=False, inplace=True)
    table_df.index = range(table_df.shape[0])


    return table_df


# --------------------
#   返回一个unique的值
# --------------------
#支持的数据类型：list
#待完成工作：
#1.支持其他数据类型，比如numpy

def get_unique(user_input):
    if type(user_input) is list:
        new_list = []
        for item in user_input:
            if item not in new_list:
                new_list.append(item)

    else:
        raise Exception ('暂不支持数据类型' + type(user_input))

    return new_list


# --------------------
#   rank_no_equal
# --------------------
    
def rank_no_equal(val):
    raw_rank = pd.DataFrame(val).iloc[:, 0].rank()
    raw_rank = (raw_rank + np.array(range(raw_rank.shape[0])) / (1 + raw_rank.shape[0])).rank(ascending = False)
    
    return raw_rank.rank(ascending = False).astype(int)
    


# ------------------------
#   integer_distribution
# ------------------------
    
def integer_distribution(prob_dist, total_ct):
    
    raw_dist = np.round(total_ct * prob_dist , 0).astype(int)
    
    gap = total_ct - raw_dist.sum()
    
    # If there are task left, evenly distribute it among the most powerful ones
    if gap > 0:
        highest_power = max(prob_dist)
        while gap > 0:
            for i in range(len(prob_dist)):
                if prob_dist[i] == highest_power:
                    raw_dist[i] = raw_dist[i] + 1
                    gap = gap - 1
                    if gap >= 0:
                        break
    
    # If there are too much, take off task from the weakest ones
    elif gap < 0:
        power_rank = fmu.rank_no_equal(prob_dist)
        while gap < 0:
            for rank in range(min(power_rank), max(power_rank) + 1):
                loc = power_rank == rank
                if raw_dist[loc] > 0:
                    raw_dist[loc] = raw_dist[loc] - 1
                    gap = gap + 1
                    if gap <= 0:
                        break

    return raw_dist

# ------------------------
#   to_pdtime
# ------------------------

# Converst a time value to pandas datetimte
    
def to_pdtime(raw_time):
    if not(type(raw_time) is pd._libs.tslibs.timestamps.Timestamp):
        raw_time = pd.to_datetime(raw_time)
    return raw_time
