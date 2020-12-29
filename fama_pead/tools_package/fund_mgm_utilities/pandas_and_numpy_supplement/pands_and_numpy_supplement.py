
# 写注释

import fund_mgm_utilities as fmu
import numpy as np
import pandas as pd


# 还真需要单独写一个函数。。。不可想象
def concat_2_dfs(left_df, right_df):

    column_list = np.concatenate([left_df.columns.values, right_df.columns.values])
    combined_df = pd.DataFrame.merge(left_df, right_df, left_index = True, right_index = True)
    combined_df = combined_df[column_list]                # 列需要重新排序
    return combined_df

# -----------------------------------------------------------------------------
# 在将dataframe输出到csv的同时检查文件路径是否存在，如果不存在的话则先创建一个
# -----------------------------------------------------------------------------

def df_to_csv(input_dataframe, path, index = False, header = None, encoding = 'GBK'):

    folder_path = fmu.get_folder_name_from_path(path)

    fmu.create_folder(folder_path)

    input_dataframe.to_csv(path, index = index, header = header, encoding = encoding)

    return True

# -----------------------------------------------------------------------------
#           提取df的一列，将其转换成数字，并将nan转换成0
# -----------------------------------------------------------------------------

def df_col_2_np(df, col_name, convert_2_float = True, nan_2_zero = True):

    out_np = df[col_name].values

    if convert_2_float:
        out_np = out_np.astype('float')

    if nan_2_zero:
        out_np[np.isnan(out_np)] = 0

    return out_np


# -----------------------------------------------------------------------------
#           两个 numpy array 相除， 除数为0反回nan, 或者其他值
# -----------------------------------------------------------------------------
# 这样做的好处是被除的时候不会发出警告
def div(numerator, divisor, out_value = float('nan')):
    true_divisor = divisor.copy()
    true_divisor[true_divisor == 0] = out_value
    return numerator / true_divisor


# -----------------------------------------------------------------------------
#       将一个标记一个1维nummpy boolean  附近的值为True
# -----------------------------------------------------------------------------
# current_loc:      1维nummpy boolean        当前位置
# neighor_range:    int                    “附近”的距离设定


def mark_neighbors(current_loc, neighor_range):
    neighbor_loc = current_loc.copy()
    for i in range(len(current_loc)):
        if current_loc[i]:
            for j in range(1, neighor_range + 1):
                if i - j >= 0:
                    neighbor_loc[i - j] = True
                if i + j < current_loc.shape[0]:
                    neighbor_loc[i + j] = True
    return neighbor_loc


# -----------------------------------------------------------------------------
#                               是否为数数字
# -----------------------------------------------------------------------------
def isnumeric(input):
    try:
        if type(input) is str:
            return False
        else:
            # 可以应对numpy.float64等情况
            return type(float(input)) is float
    except:
        return False


# -----------------------------------------------------------------------------
#         读取时间序列文件专用，可以自动侦测编码以及是否有标题行
# -----------------------------------------------------------------------------
# 核心目的是把各种读取方法都尝试一遍

def pd_readcsv_ts(file_path, open_first = [True, False], header = 'auto', encoding = [[], 'GBK', 'GB1312', 'utf-8'], sep = ',', skipinitialspace = True,  return_read_method = False):

    # 用户知道该怎么读取,那就不尝试各种方法了
    if (not type(encoding) is list) and (not type(open_first) is list):
        read_success, df = pd_readcsv_ts_check_open_and_encoding(file_path, open_first=open_first, encoding=encoding, sep = sep, skipinitialspace = skipinitialspace,  return_data=True, header = None)
        used_open_method = open_first
        used_encoding = encoding

    else:
        
        if not (type(open_first) is list):
            open_first = [open_first]

        if not (type(encoding) is list):
            encoding = [encoding]

        # 开始遍历所有方法
        read_success = False
        for i in range(len(open_first)):
            for j in range(len(encoding)):
                if not read_success:
                    read_success, df = pd_readcsv_ts_check_open_and_encoding(file_path, open_first=open_first[i], encoding=encoding[j], sep = sep, skipinitialspace = skipinitialspace,  return_data=True, header = None)
                    used_open_method = open_first[i]
                    used_encoding = encoding[j]


    if read_success:
        # 自动查询是否存在标题行
        if header == 'auto':
            # 第一行不是数字，则必须为标题行
            if not fmu.isnumeric(df.iloc[0,1]):
                # 重新读取数据. 如果标题行在第二行应该会有问题，不过没有人会把标题放在第二行
                used_header = 0
                df = pd_read_csv_open_and_encoding(file_path, used_open_method, used_encoding, sep = sep, skipinitialspace = skipinitialspace,  header=used_header)
            else:
                used_header = None
        else:
            used_header = header

        if not return_read_method:
            return df
        else:
            return df, used_open_method, used_encoding, used_header

    else:

        print('pd_readcsv_ts: 文件读失败 ' + file_path)
        print('pd_readcsv_ts: 下面是最后一次尝试使用的读取方法产生的错误信息（前面用过的方法都不行，最后一次这个方法也不行： ')
        print('used_open_method: ' + str(used_open_method))
        print('used_encoding: ' + str(used_encoding))


        if used_open_method:
            pd.read_csv(open(file_path), encoding = encoding)
        else:
            pd.read_csv(file_path, encoding=encoding)#??????


# ----------------------------------------
#   判断某种编码是否能成功运行pd.read_csv
# ----------------------------------------
def pd_readcsv_ts_check_open_and_encoding(file_path, open_first = True, encoding = [], sep = ',', skipinitialspace = True,  return_data = True, header = 0):

    # 测试是否能成功读取
    try:
        df = pd_read_csv_open_and_encoding(file_path, open_first, encoding, sep = sep, skipinitialspace = skipinitialspace,  header = header)
        read_success = True
    except:
        read_success = False

    # 返回读取尝试结果
    if not read_success: df = []

    if return_data: return read_success, df
    else:           return read_success

# ----------------------------------------
#      将一个if条件塞进函数里
# ----------------------------------------
# 该函数存在的目的主要为了其他方便
def pd_read_csv_open_and_encoding(file_path, open_first, encoding, sep = ',', skipinitialspace = True,  header = 0):
    if open_first:
        if encoding == []:
            return pd.read_csv(open(file_path), header = header, sep = sep, skipinitialspace = skipinitialspace)
        else:
            return pd.read_csv(open(file_path, encoding = encoding), header = header, sep = sep, skipinitialspace = skipinitialspace)
    else:
        if encoding == []:
            return pd.read_csv(file_path, header = header, sep = sep, skipinitialspace = skipinitialspace)
        else:
            return pd.read_csv(file_path, encoding = encoding, header = header, sep = sep, skipinitialspace = skipinitialspace)


# ----------------------------------------
#      返回第一个最大或者最小值
# ----------------------------------------

# 从一维Numpy array里找出最大或者最小值


def first_extreme(one_D_numpy_array, dir = 'max'):

    if type(one_D_numpy_array) is list:
        one_D_numpy_array = np.array(one_D_numpy_array)

    if dir == 'max':
        extreme_loc_bol = one_D_numpy_array == np.nanmax(one_D_numpy_array)
    elif dir == 'min':
        extreme_loc_bol = one_D_numpy_array == np.nanmin(one_D_numpy_array)

    extreme_loc_num = np.where(extreme_loc_bol)

    # 提取第一个值
    extreme_loc_num = extreme_loc_num[0][0]

    if sum(extreme_loc_bol) > 1:
        extreme_loc_bol[extreme_loc_num + 1: ] = False

    return extreme_loc_num, extreme_loc_bol


# ----------------------------------------
#             将数字正则化
# ----------------------------------------

def normalize(np_array):
    arr_max = np.max(np_array)
    arr_min = np.min(np_array)

    if arr_max > arr_min:
        np_array_norm = (np_array - np.min(np_array)) / (np.max(np_array) - np.min(np_array))
    else:
        np_array_norm = np_array * 0

    return np_array_norm


# ----------------------------------------
#             list 的 Boolean indexing
# ----------------------------------------

def list_bool(value_list, bool_list, flip = False):

    value_np = np.array(value_list)
    bool_list = np.array(bool_list)
    if flip:
        bool_list = np.logical_not(bool_list)
    value_np_wanted= value_np[bool_list]

    return value_np_wanted

# ----------------------------------------
#                list_ind
# ----------------------------------------
# List indicing
    
def list_ind(list1, list2):
    new_list = []
    for ind in list2:
        new_list.append(list1[ind])
    return new_list

# ----------------------------------------
#           numpy weekday
# ----------------------------------------
def np_weekday(numpy_datetime):
    return pd.to_datetime(numpy_datetime).weekday()


# ----------------------------------------
#          将矩阵复制到Excel
# ----------------------------------------
# Credit : https://stackoverflow.com/questions/22488566/how-to-paste-a-numpy-array-to-excel

import win32clipboard as clipboard
def c2c(array):
    """
    Copies an array into a string format acceptable by Excel.
    Columns separated by \t, rows separated by \n
    """
    if type(array) is pd.core.frame.DataFrame:
        array = array.values.copy()

    # Create string from array
    line_strings = []

    if len(array.shape) == 2:
        for line in array:
            line_strings.append("\t".join(line.astype(str)).replace("\n",""))
    elif len(array.shape) == 1:
        for line in array:
            if type(line) is str:
                line_strings.append(line)
            elif type(line) is float:
                line_strings.append(str(line))
            else:
                line_strings.append(line.astype(str))
    else:
        raise Exception("只支持1、2维矩阵，输入的矩阵维度是：" + str(len(array.shape)))

    array_string = "\r\n".join(line_strings)

    # Put string into clipboard (open, clear, set, close)
    clipboard.OpenClipboard()
    clipboard.EmptyClipboard()
    clipboard.SetClipboardText(array_string)
    clipboard.CloseClipboard()


# ----------------------------------------
#      比较两个Dataframe的差距
# ----------------------------------------

# 返回：一个新的dataframe, 行列与new_df一样。
# 如果只有新的有数据，或者新数据与老数据差距不一样，则返回新数据

def get_df_diff(old_df, new_df, diff_tol=1e-8):
    new_info_df = pd.DataFrame(index=new_df.index, columns=new_df.columns)
    for time in new_df.index:
        for item in new_df.columns:
            info_is_new = False

            if not fmu.isnan(
                    new_df.loc[time, item]):  # If the information is observed in the current information report
                if (time in old_df.index) and (
                        item in old_df.columns):  # Check if it exist in the previous information report
                    if fmu.isnan(old_df.loc[time, item]):  # if old info does not exists
                        info_is_new = True  # then current info is auto new
                    elif abs(new_df.loc[time, item] - old_df.loc[
                        time, item]) > diff_tol:  # else check if much has changed
                        info_is_new = True  # if change is big then then info is new as well
                else:
                    info_is_new = True  # Old info is not within report scoping. Current info is new
            if info_is_new:
                new_info_df.loc[time, item] = new_df.loc[time, item]

    return new_info_df

# ------------------------------------------------------------------------------------------------------------
#                              Converts a one column dataframe to a dictionary
# ------------------------------------------------------------------------------------------------------------
def one_col_df_to_dict(df):
    out_df = {}

    for key in df.index:
        out_df[key] = df.loc[key].values[0]

    return out_df

# ------------------------------------------------------------------------------------------------------------
#                              Load a two column csv and convert it into a dictionary
# ------------------------------------------------------------------------------------------------------------

def load_table_into_dict(path):

    return one_col_df_to_dict(pd.read_csv(open(path), index_col=0, header = None))



def list_intersect(a, b):
    """ return the intersection of two lists """
    return list(set(a) & set(b))

def list_union(a, b):
    """ return the union of two lists """
    return list(set(a) | set(b))

def list_same(a, b):
    if len(a) != len(b):
        return False
    else:
        for ele_a, ele_b in zip(a, b):
            if ele_a != ele_b:
                return False
        return True

# ---------------------------
#    reindex_df_columns
# ---------------------------

# Change the order columns of old_df to new_columns
# For columns that exist in new_columns but not in old_df, fill it fill_value
    
def reindex_df_columns(old_df, new_columns, fill_value = np.nan):
                             
    if not(list_same(new_columns, old_df.columns)):
        new_cols_not_in_old = set(new_columns) - set(old_df.columns)
        
        new_np = np.zeros((len(old_df.index), len(new_columns)))
        new_np[:] = fill_value
        
        for i, new_column in enumerate(new_columns):
            if not(new_column in new_cols_not_in_old):
                 new_np[:, i] = old_df[new_column].values
            
        new_df = pd.DataFrame(new_np, index = old_df.index, columns = new_columns)
        
        return new_df
    
    else:
    
        return old_df.copy()
    
def top_n(input_arr, n):
    return np.argpartition(input_arr, -n)[-n:]    
    
def multi_dic_to_df(list_dic):
    out_mtx = pd.DataFrame()
    
    for i, v in enumerate(list_dic):
        out_mtx = pd.concat([out_mtx, pd.DataFrame(v, index = ["请注意修改index为年份"])], axis = 0)
        
    return out_mtx