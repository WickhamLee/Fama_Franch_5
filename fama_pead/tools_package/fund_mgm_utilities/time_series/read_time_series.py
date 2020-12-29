# 目的： 从文件读取时间序列，并完成日期处理、对齐，填充等工作


# 编辑记录


#             时间            |              修改内容
#    20181217 19:59 GMT +8                 创建此文件


import pandas as pd
import numpy as np
import datetime as dt
import os
import math
import inspect
import warnings
import shutil
import hashlib
import fund_mgm_utilities as fmu

# 函数名：read_time_series
# 输入：  file_path_list         list                   所需要读取的时间序列的文件完整路径列表                    '' (默认值）
#        do_not_read_file_list  list,                  跳过列表中的这些文件                                      [] (默认值）
#        only_read_these_files_list     list           只读该列表中出现的文件名                                  [] (默认值）
#        ignore_if_file_path_contain    list           文件路径中出现过这个，则不读取                            [] (默认值）
#                                                                                                                [] (默认值）
#        file_list_method:      string,                如何决定文件列表：                                        file_path_list （默认值,某些情况会根据用户输入进行智能判断，详情参见代码）  用户给定一个文件列表
#                                                                                                                all_files_in_folder     读取文件夹下所有的数据文件，不包含子文件夹
#                                                                                                                all_files_in_subfolders 读取文件夹下所有的数据文件，包含子文件夹
#        folder_path:           string,                如果需要读取文件夹下所有文件，则在此输入文件夹路径        默认值：''
#        sep                    string                 列之间用什么区分的                                        默认值：','
#                                                                                                                其他常见值 ' ', '\t'
#        skipinitialspace       boolean                有些csv文件空白太多，需要把这个设置成True                 默认值：True
#        encoding               string                 中文编码                                                  默认值：GBK

#        check_file_type        boolean,               是否检查文件类型                                          默认值：True
#        check_for_empty_fies   boolean                检查文件大小是否大约0,忽略空文件                          默认值：True
#        file_type：            list,                   只读取哪些文件类型                                        [csv, txt]  (默认值）
#        source_time_format     string,                原始数据里时间的格式                                     ‘guess' （默认值, 根据数据源自动匹配)   原始数据里关于时间的标记可能有多重形式。有的可能是excel时间计数，有的可能是yyyymmdd格式，必须告知后才知道如何转换。其他支持格式有待编写
#                                                                                                                do_not_change： 别做任何改变
#        target_time_method     string,                对齐的时间序列对齐到哪些日期？                            fix_freq:          固定频率出现
#                                                                                                                union（默认值）:   所有原始日期和合集
#                                                                                                                given              指定日期list。用target_time_list来指定需要将日期对齐到哪些日子
#        target_time_list       list                   指定将日期对齐到这些日子                                  [] (默认值)                 只有 target_time_method = ‘given'时才会用到
#        time_freq              double,                如果需要对齐到固定频率，频率是几天？                      7（默认值)
#        starttime              pd.Timestamp object    对齐时间序列开始时间                                      ''(默认值) 设定为此默认值时，用所有输入时间序列中最早的数据作为开始时间
#        start_hour             int                    修改starttime的小时属性                                   ''(默认值) 读取分钟线时可能会用到这个属性。如果不设定这个参数，starttime有可能等于原始数据中最早的一点，如果这一点时12点话，而时间又用fix_freq生成，通常不如3点好。
#        start_minute           int                    修改starttime的分钟属性                                   ''(默认值)
#        start_second           int                    修改starttime的秒属性                                     ''(默认值)
#        start_weekday          int                    将starttime推迟至星期几开始                               ''

#        endtime                pd.Timestamp object    对齐时间序列结束时间                                      ''(默认值) 设定为此默认值时，用所有输入时间序列中最晚的数据作为结束时间
#        date_column_no         integer 或者 list      日期在原始数据的第几列                                    0 (默认值）
#        data_column_no         integer 或者 list      所需要的数据（比如净值）在第几列                          1 (默认值)
#        header                 integer 或者 string    原始数据有没有标题行                                      'auto' (默认值) 'auto'： 自动智能判断。如果有非数字标题会被自动识别。 None ： 没有标题。 0： 标题在第一行。 1： 标题在第二行
#        one_df_stores          string                 不同的列之间的区别是是什么                                若data_column_no长度大于1，即需要同时读取多个field,那么数据以什么形式展示呢？
#                                                                                                                everything           所有数据放在一个dataframe里面
#                                                                                                                one_field (默认值）   则会生成多个dataframe, 每个dataframe对应一个field，不同的列展示来自不同文件的数据
#                                                                                                                one_file             每个dataframe对应一个file,不同的列展示不同的field的数据
#        field_names_list       list                   每列分别叫什么名字                                        ''(默认值） 默认用数字命名
#        fill_missing_with      string, math.nan       遇到空缺值怎么处理                                        "last_known"  用最后一个知道的值填充
#                                                                                                                math.nan      填为空白
#        sort_data              boolean                是否需要对数进行排序                                      True          改成False可以提速，但原始数据中若有没有排好序的则可能出问题
#        return_what            string                 只返回数据还是其他中间变量也一起返回                      "data_only"   只返回数据
#        time_on_index          boolean                是否只把时间写在index 里                                  False
#        check_for_duplicates   boolean                是否自动删除日期重复的数据
#        ！！！缓存机制注释需要写


# 输出： 一段对齐的时间序列，以dataframe的形式储存

# 需要解决的问题
# 10. 将一些文件处理函数剥离出去
# 11. 缓存读取标准化，方便给其他函数用
# 12. data_col_no 允许用名称来引索，而不是只能用数字
# 14. 不少辅助函数具有通用性，需要独立出去
# 15. 如果日期读取失败，则跳过某个时间序列。

# ！！！ 尾部数据缺失时是否给警告



def read_time_series(file_path_list='', file_list_method='', check_file_type = True, check_for_empty_fies = True, do_not_read_file_list=[],
                     only_read_these_files_list = [], ignore_if_file_path_contain = [],
                file_type=['csv', 'txt'], folder_path='', source_time_format='guess',
                starttime='', start_hour = '', start_minute = '', start_second = '', start_weekday = '', weekday_filter = [],
                endtime='', target_time_method='union', time_freq=7, target_time_list = [], time_col_name = 'time',
                date_col_no=0, data_col_no=1, header = 'auto', one_df_stores='everything', field_names_list = '',
                sep= ',', skipinitialspace=True, encoding = 'GBK',
                fill_missing_with='last_known', sort_data= True, duplicated_data = 'give_warning', save_to_local_path = '',
                save_output = False, cache_folder = '', task_ID = '', load_from_cache = False,
                print_setting = dict(),
                return_what = 'data_only', time_on_index = False, check_for_duplicates = False, extrapolate_last = False):



    '------------------------- 设定函数如何输出运行状态 ---------------------------'
    # 处理输出运行状态相关的事情
    print_setting = fmu.initialize_print_setting(print_setting)
    fmu.print_status(print_setting, 'begin')
    '------------------------- 数据更新函数的标准部分： 结束 ---------------------------'



    # 如果需要保存参数，则记录所有输入变量，以检查是否有缓存存在
    data_read_from_cache = False
    if save_output | load_from_cache:

        if  os.path.isdir(cache_folder):
            fucntion_inputs_hash = hash_function_input(locals(), inspect.signature(read_time_series))
            output_folder = cache_folder + '\\' + fucntion_inputs_hash
            if task_ID != '':
                output_folder = output_folder + '_' + task_ID
            cache_exist = os.path.isdir(output_folder)
        else:
            raise Exception('临时储存变量文件夹不存在：' + cache_folder)
            exit()

        if load_from_cache:
            if cache_exist:

                # ----------------------------------------- 非标准化内容 ----------------------------------------
                if one_df_stores == 'everything':
                    merged_df_list = fmu.pd_readcsv_ts(output_folder + '\\read_time_series_output_' + str(1) + '.csv')
                    merged_df_list.iloc[:, 0] = pd.to_datetime(merged_df_list.iloc[:, 0])            # 这里对日期做了一些假设，估计以后要改
                elif one_df_stores == 'one_file' or one_df_stores == 'one_field':
                    merged_df_list = []
                    file_path_list, _ = fmu.get_all_folder_files(output_folder)
                    for i in range(len(file_path_list)):
                        ith_file_path = output_folder + '\\read_time_series_output_' + str(i) + '.csv'
                        ith_df = fmu.pd_readcsv_ts(ith_file_path)
                        ith_df.iloc[:, 0] = pd.to_datetime(ith_df.iloc[:, 0])                        # 这里对日期做了一些假设，估计以后要改
                        merged_df_list.append(ith_df)
                # ----------------------------------------- 非标准化内容 ----------------------------------------

                data_read_from_cache = True
            else:
                warning_message = ("read_time_series: 尝试读取本地已缓存的数据但缓存不存在。数据将采用正常办法读取：" + output_folder)
                warnings.warn(warning_message)




    # ----------------------------------------------------------------------------------------------------------------
    #                                       生成需要被读取的文件列表
    # ----------------------------------------------------------------------------------------------------------------
    if not data_read_from_cache:

        sub_path_list = ''
        # 一些情景可以根据用户输入自动选择 file_list_method

        # 未给定文件列表，指定了文件夹，未指定拉文件方法， 则只读取该文件夹下文件，但不读取子文件夹
        if file_path_list == '' and folder_path !=  '' and file_list_method == '':
            file_list_method = "all_files_in_folder"


                                                                            #!!! Need flie name list
        # 给定了文件列表，未给定文件路径，则假设读取文件列表内的所有文件
        elif file_path_list != "" and folder_path == '':
            file_list_method = "file_path_list"
            file_names_list = []
            for file_path in file_path_list:
                file_names_list.append(fmu.get_file_name_from_path(file_path))

        # 生成需要被读取的文件名称以及完整路径列表：
        if file_list_method == 'all_files_in_folder':                       # 读取文件件下所有的文件，不包括子文件夹下的
            file_path_list, file_names_list = fmu.get_all_folder_files(folder_path)

        # 读取包含子文件夹内的所有文件
        elif file_list_method == 'all_files_in_subfolders':                 # 读取文件夹下的所有文件，包括子文件夹下的
            file_path_list, file_names_list, sub_path_list = fmu.get_all_subfolder_files(folder_path)

        elif file_list_method == 'file_list':                               # 用户直接给出完整的文件路径列表
            file_names_list = []
            for file_path in file_path_list:
                file_names_list.append(fmu.get_file_name_from_path(file_path))

        # 从文件列表删除不符合需求的文件：
        file_is_wanted = []                                     # list, boolean, 储存每个文件是否符合需求
        if check_file_type:

            # 检查文件类型
            for file_name in file_names_list:
                file_extension, file_name_proper = fmu.get_file_extension(file_name)
                if file_extension in file_type:
                    file_is_wanted.append(True)
                else:
                    file_is_wanted.append(False)

            # 检查负面清单
            if not(do_not_read_file_list == []):
                j = -1
                for file_name in file_names_list:
                    j = j + 1
                    if file_name in do_not_read_file_list:
                        file_is_wanted[j] = False

            # 检查正面清单
            if not(only_read_these_files_list == []):
                j = -1
                for file_name in file_names_list:
                    j = j + 1
                    if not (file_name in only_read_these_files_list):
                        file_is_wanted[j] = False

            # 检查文件路径中是否含有用户指定不要的东西
            if not ignore_if_file_path_contain == []:
                j = -1
                for file_path in file_path_list:
                    j = j + 1
                    for bad_str in ignore_if_file_path_contain:
                        if bad_str in file_path:
                            file_is_wanted[j] = False
                            break

            # 检查是否有些文件时空的， 或者不存在的文件
            if check_for_empty_fies:
                j = -1
                for file_path in file_path_list:
                    j = j + 1
                    if not(os.path.isfile(file_path)):
                        file_is_wanted[j] = False
                        warnings.warn('这个文件不存在，将被忽略' + file_path)
                    elif os.path.getsize(file_path) == 0:
                        file_is_wanted[j] = False
                        warnings.warn('这个文件大小为0' + file_path)

        # ------------------------------
        # 没有符合要求的文件则啥都不返回
        # ------------------------------
        if sum(file_is_wanted) == 0:
            print("警告：read_time_series 未找到符合要求的数据文件，返回空!!!")
            merged_df_list = []
            file_path_list = []
            return merged_df_list, file_path_list

        file_names_list = remove_list_elements(file_names_list, file_is_wanted)
        file_path_list = remove_list_elements(file_path_list, file_is_wanted)
        if sub_path_list != '':
            sub_path_list = remove_list_elements(sub_path_list, file_is_wanted)

        # ----------------------------------------------------------------------------------------------------------------
        #                                       将所有原始数据读取成dataframe
        # ----------------------------------------------------------------------------------------------------------------

        file_ct = len(file_path_list)
        df_list = []
        source_time_format_user_setting = source_time_format
        df_raw=[]
        for i in range(0, file_ct):

            # 输出状态
            fmu.print_status(print_setting, 'mid', "准备尝试读取这个文件: " + file_path_list[i] + ' ' + str(i) + '/' + str(file_ct))

            # 读取数据
            ith_df_raw = fmu.pd_readcsv_ts(file_path_list[i], header=header, sep=sep, skipinitialspace=skipinitialspace)                            # pd_readcsv_ts 可以自动处理一些读取数据时遇到的复杂情况
            df_raw.append(ith_df_raw)
            # 提取需要的数据所在的列, 这段逻辑有点乱
            # 其实先分开处理完时间和数据，再横向拼接dataframe是更清晰的逻辑，但是这里先这样忍者吧。以后再慢慢处理
            data_idx = []
            if type(date_col_no) is int or type(date_col_no) is str:
                date_col_no = [date_col_no]
            data_idx.append(date_col_no[0])

            if type(data_col_no) is int or type(data_col_no) is str :
                data_col_no = [data_col_no]

            for jth_col_no in data_col_no:
                data_idx.append(jth_col_no)

            # 提取需要的数据所在的列
            try:
                ith_df = ith_df_raw.iloc[:, data_idx].copy()                # 这里不加copy, 它其实也是在做Copy. 到后面对ith_df做修改时python会不停的出警告
            except:
                a = 1
            # -------------
            # 调整时间格式
            # -------------
            ## This might gives huge error on named columns!!!!
            ith_df.iloc[:, 0] = parse_time_format(ith_df.copy().iloc[:, 0], source_time_format)

            # 原始数据中用Last Known替换NaN
            if fill_missing_with == 'last_known':
                ith_df.fillna(method='ffill', inplace=True)

            # 处理重复数据的问题
            if duplicated_data != 'do_nothing':
                repeat_dt_count = len(ith_df.iloc[:, 0]) - len(ith_df.iloc[:, 0].unique())
                if  repeat_dt_count > 0:
                    if duplicated_data == 'keep_first_only':                                    # 只保留第一个数据
                        ith_df.drop_duplicates(keep= 'first', inplace = True)
                    elif duplicated_data == 'keep_last_only':                                   # 只保留最后一个数据
                        ith_df.drop_duplicates(keep='last', inplace=True)
                    elif duplicated_data == 'give_warning':                                     # 给个警告就行了
                        warnings.warn('read_time_series: 数据日期有重复。数据来自于这个文件： ' + file_path_list[i] + ' 重复次数: ' + str(repeat_dt_count))

            # -------------
            #  给列取名字
            # -------------

            # 生成两个list, 储存不含文件后缀的文件名，和文件后缀。后期给dataframe内的列取名时会需要用到这些信息
            file_name_proper_list = []                          # 不含后缀的文件名
            file_extension_list = []                            # 文件后缀
            file_name_id_list = []                              # 这个用来区分不同的文件. 后面给Dataframe不同的列取名时也使用这个List里的string

            for file_name in file_names_list:
                file_extension, file_name_proper = fmu.get_file_extension(file_name)
                file_name_proper_list.append(file_name_proper)
                file_extension_list.append(file_extension)

            # 当读取了一个文件夹内的所有子文件夹文件时，文件名有可能重复。此时需要
            # 需要加上子路径，才能确保不同文件名之间的标记不同
            if file_list_method == 'all_files_in_subfolders':
                j = -1
                for file_name_proper in file_name_proper_list:
                    if file_name_proper_list.count(file_name_proper) > 1:
                        j = j + 1
                        full_file_name = sub_path_list[j].replace('\\', '_') + file_name_proper
                        full_file_name = full_file_name[1: ]
                    else:
                        full_file_name = file_name_proper
                    file_name_id_list.append(full_file_name)

            # 不然的话用直接不含后缀的文件名来区分不同的列
            else:
                file_name_id_list = file_name_proper_list

            ith_df_col_names = [time_col_name]
            # 如果用户没有给field名称， 则用数字编号
            if field_names_list == '':
                field_names_list = []
                for j in range(len(data_col_no)):
                    field_names_list.append(str(j))

            # 只有一个field, 用文件名区分
            if len(data_col_no) == 1:
                ith_df_col_names.append(file_name_id_list[i])

            # 多个field， 一个文件，用field名称区分
            elif len(file_names_list) == 1 and len(data_col_no) > 1:
                for j in range(len(data_col_no)):
                    ith_df_col_names.append(field_names_list[j])

            # 多个文件，多个field, 用文件名 + field
            elif len(file_names_list) > 1 and len(data_col_no) > 1:
                for j in range(len(data_col_no)):
                    ith_df_col_names.append(file_name_id_list[i] + '_' + field_names_list[j])

            ith_df.columns = ith_df_col_names
            df_list.append(ith_df)





        # -------------------
        # 决定开始和结束时间
        # -------------------
        # 若用户未指定开始和结束时间，则自定选择原始数据中最大的范围
        auto_set_starttime = True if starttime == '' and source_time_format != "do_not_change" else False                   # 如果需要读取函数别碰原始时间，则不自动选取开始和结束时间
        auto_set_endtime = True if endtime == '' and source_time_format != "do_not_change" else False

        for i in range(file_ct):
            this_df = df_list[i]
            if auto_set_starttime:
                if i == 0:
                    starttime = pd.Timestamp('2200-01-01')

                if this_df.iloc[:, 0].shape[0] > 0:
                    try:
                        starttime = min(starttime, min(this_df.iloc[:, 0]))
                    except:
                        a = 1

            if auto_set_endtime:
                if i == 0:
                    endtime = pd.Timestamp('1700-01-01')

                if this_df.iloc[:,0].shape[0] > 0:
                    try:
                        endtime = max(endtime, max(this_df.iloc[:,0]))
                    except:
                        a = 1


        # 修改开始时间的日内位置
        if not(start_hour == ''): starttime = starttime.replace(hour = start_hour)                      # 小时
        if not(start_minute == ''): starttime = starttime.replace(minute = start_minute)                # 分钟
        if not(start_second == ''): starttime = starttime.replace(second = start_second)                # 秒
        if not (start_weekday == ''):                                                                   # 星期几开始
            current_weekday = starttime.weekday() + 1
            if start_weekday >= current_weekday:
                starttime = starttime + dt.timedelta(start_weekday - current_weekday)
            else:
                starttime = starttime + dt.timedelta(start_weekday - current_weekday + 7)

        # -------------------
        #  生成目标时间序列
        # -------------------
        # 这里决定最终输出的Dataframe的时间点在哪

        # 固定频率的时间
        if target_time_method == "fix_freq":
            target_time_list = gen_time_list(starttime, endtime, time_freq)
            target_time_df = pd.DataFrame(target_time_list, columns=[time_col_name])

        # 所有原始数据的时间集合
        elif target_time_method == "union":
            for j in range(len(df_list)):
                if j == 0:
                    target_time_df = df_list[0][time_col_name].to_frame()
                else:
                    target_time_df = pd.merge(target_time_df, df_list[j][time_col_name].to_frame(), on=time_col_name, how='outer')

            target_time_df.sort_values(by=time_col_name, ascending=True, inplace=True)

        # 用户指定好了时间
        elif target_time_method == "given":
            target_time_df = pd.DataFrame(target_time_list)
            target_time_df.columns = ['time']

        else:
            raise Exception('不知道这种时间对齐方式是什么意思' + target_time_method)


        # 剔除不符合开始结束时间范围的目标时间
        in_bound_rows = (target_time_df >= starttime) & (target_time_df <= endtime)
        target_time_df = target_time_df[in_bound_rows .iloc[:, 0]]
#        剔除日期中双休日，后续可将交易日历加入！！！
        if weekday_filter!=[]:
            target_time_df = target_time_df[(target_time_df['time'].dt.dayofweek + 1).isin(weekday_filter)]

        # 如果用户需要，给数据先排序
        if sort_data:
            j = - 1
            for each_df in df_list:
                j = j + 1
                sorted_df = each_df.sort_values(by=time_col_name, ascending=True)
                df_list[j] = sorted_df

        # 如果用户需要，移除有重复的数据点
        if check_for_duplicates:
            for j, each_df in enumerate(df_list):
                df_with_no_dup = each_df.drop_duplicates(subset = time_col_name)                        # !!! 这段代码需要实现
                df_list[j] = df_with_no_dup

        # -------------------
        #     对齐数据
        # -------------------

        # 遇到没有数据的日期，用过去最近的数据填充
        if fill_missing_with == 'last_known':
            for i in range(len(df_list)):
                # 输出状态
                fmu.print_status(print_setting, 'mid', "合并完第 " + str(i) + " 条时间序列")

                if i == 0:
                    merged_df = pd.merge_asof(target_time_df, df_list[0], on=time_col_name, direction='backward')
                else:
                    merged_df = pd.merge_asof(merged_df, df_list[i], on=time_col_name, direction='backward')

        # 遇到没有数据的日期，用Nan填充
        elif math.isnan(fill_missing_with):
            for i in range(len(df_list)):
                # 输出状态
                fmu.print_status(print_setting, 'mid', "合并完第 " + str(i) + " 条时间序列")
                if i == 0:
                    merged_df = pd.merge(target_time_df, df_list[0], on=time_col_name, how='left')
                else:
                    merged_df = pd.merge(merged_df, df_list[i], on=time_col_name, how='left')

        # 如果用户需要知道哪些数据是“真是数据”，哪些是根据过去的数据外推的
        if return_what == 'mark_exterpolated_data':
            if fill_missing_with == 'last_known':
                for i in range(len(df_list)):
                    if i == 0:
                        merge_df_no_extrapolation = pd.merge(target_time_df, df_list[0], on=time_col_name, how='left')
                    else:
                        merge_df_no_extrapolation = pd.merge(merge_df_no_extrapolation, df_list[i], on=time_col_name, how='left')
                merge_df_no_extrapolation = merge_df_no_extrapolation.set_index('time')
                not_extrploated_df = merge_df_no_extrapolation.notnull()
            elif math.isnan(fill_missing_with):
                not_extrploated_df = merged_df.notnull()

            

        # ----------------------------------------------------------------------------------------------------------------
        #                                                 尾端日期处理
        # ----------------------------------------------------------------------------------------------------------------

        # 对于最后一个日期，需要做一些特殊处理。举例来说，如果目标日期里，第x天的净值已经来源于原始数据里的最后一天，则后面的目
        #  标日期返回数字应该是nan，而非原始数据里的最后一天的值。这样做可以避免目标日期最后一系列日期都在原始数据最后一天之后
        #  时，对齐出一个长期不变的净值曲线的情况。
        # 这种情况等同于在对未来做预测，我们应该只预测一次。若强行需要预测多日，应该用使用历史波动率进行模拟等方法，而非一直假设
        # 未来净值没有任何波动。
        if not extrapolate_last:
            
            target_time_ct = len(target_time_df)
    
            for j in range(1, len(df_list) + 1):
    
                # 输出状态
                fmu.print_status(print_setting, 'mid', "处理完第 " + str(j) + " 条时间序列的尾部")
    
                if df_list[j - 1][time_col_name].shape[0] > 0:
                    this_list_max_time = max(df_list[j - 1][time_col_name])
    
                    if not type(this_list_max_time) is str:            # 如果原始数据里并非数据，则不做比较
                        bad_rows = target_time_df.iloc[:, 0] >= this_list_max_time
                        bad_rows[bad_rows.idxmax()] = False
                        merged_df.loc[bad_rows.values, merged_df.columns[j]] = math.nan



        # 是否只希望把日期放在Index上
        if time_on_index:
            merged_df.index = merged_df[time_col_name]
            merged_df.index.name = time_col_name
            merged_df.drop(columns = time_col_name, inplace=True)


        # ----------------------------------------------------------------------------------------------------------------
        #                               对于多个文件，多个field的情况，进行数据切割
        # ----------------------------------------------------------------------------------------------------------------
        # 若需要从多个文件读取多个field, 则用户可能希望每个文件或者field的数据自成一个dataframe, 最终返回一个list of dataframe
        # 这种情况下，需要将上面的dataframe进行切割

        # 一个field的内容放在一个dataframe
        idx_shift = 1
        if time_on_index: idx_shift = 0

        if one_df_stores == 'one_field':
            merged_df_list = []
            for j in range(len(field_names_list)):

                if time_on_index:
                    this_df_idx = []
                else:
                    this_df_idx = [0]

                for k in range(len(file_names_list)):
                    this_df_idx.append(k * len(field_names_list) + j + idx_shift)
                merged_df_list.append(merged_df.iloc[:, this_df_idx])

        # 一个文件的内容放在一个dataframe
        elif one_df_stores == 'one_file':
            merged_df_list = []
            for j in range(len(file_names_list)):

                if time_on_index:
                    this_df_idx = []
                else:
                    this_df_idx = [0]

                for k in range(len(field_names_list)):
                    this_df_idx.append(k + idx_shift + j * len(field_names_list))
                merged_df_list.append(merged_df.iloc[:, this_df_idx])

        # 所有数据全部放入一个dataframe,此时不返回一个list而直接返回一个dataframe
        elif one_df_stores == 'everything':
            merged_df_list = merged_df

        # ---------------------------------------------------------------------------------------------------------------------------------
        #                                          将处理完的数据储存至硬盘
        # ---------------------------------------------------------------------------------------------------------------------------------
        # 对于大数据量的读取，拼接，对齐，会消耗比较长的时间，此时最好能将数据存入硬盘，这样下次再读取一样需求的时候，先检查本地是否有缓存
        # 这种情况下，需要将上面的dataframe进行切割

        if save_output:

            # 清除现有缓存数据
            if cache_exist:
                shutil.rmtree(output_folder, ignore_errors=True)

            os.mkdir(output_folder)

            # ----------------------------------------- 非标准化内容 ----------------------------------------
            if type(merged_df_list) is list:
                for i in range(len(merged_df_list)):
                    merged_df_list[i].to_csv(output_folder + '\\read_time_series_output_' + str(i) + '.csv', index = False)
            else:
                merged_df_list.to_csv(output_folder + '\\read_time_series_output_1.csv', index = False)
            # ----------------------------------------- 非标准化内容 ----------------------------------------
    


    # 输出状态
    fmu.print_status(print_setting, 'end')
    

    # 整理好数据，返回给函数呼叫方
    if return_what == 'data_only':
        return_content = merged_df_list
    elif return_what == 'file_path_list':
        return_content = (merged_df_list, file_path_list)
    elif return_what == 'mark_exterpolated_data':
        return_content = (merged_df_list, not_extrploated_df)

    return return_content



# ================================================================================================================
#
#                                               下面为辅助函数
#
# ================================================================================================================


# excel_date_to_datetime
# 目的: 将Excel 日期改成python 标准日期
#   参见： https://stackoverflow.com/questions/38454403/convert-excel-style-date-with-pd
#   输入： excel_series 一个包含有excel日期的series 或者list
#   输出： 转换成了datetime的日期列表

def excel_date_to_datetime(excel_series):
    return pd.TimedeltaIndex(excel_series, unit='d') + dt.datetime(1899, 12, 30, 0, 0, 0)


def matlab_date_to_datetime(matlab_series):
    return pd.TimedeltaIndex(matlab_series - 693960, unit='d') + dt.datetime(1899, 12, 30, 0, 0, 0)

#  产生等频率的时间序列
def gen_time_list(starttime, endtime, time_freq):
    time_list = [starttime]
    current_time = starttime
    while current_time < endtime:
        current_time = current_time + dt.timedelta(time_freq)
        time_list.append(current_time)
    return time_list



# 将 yyyymmdd 分解成 yyyy, mm, dd， yyyymmddd 是 integer
def parse_yyyymmdd(yyyymmdd):
    year = math.floor(yyyymmdd / 10000)
    month = math.floor(yyyymmdd / 100) - year * 100
    day = math.floor(yyyymmdd) - year * 10000 - month * 100

    return year, month, day



# 输入两个长度一样的list, 第二个是一个Boolean， 从第一个list中提取被第二个标记为True的
def remove_list_elements(list, index_of_wanted):
    new_list = []
    i = 0
    for wanted in index_of_wanted:
        if wanted:
            new_list.append(list[i])
        i = i + 1
    return new_list

# --------------------------------
#   根据样本string来猜测日期格式
# --------------------------------
# throw_exception_if_failed: 如果猜不到，是直接报错还是返回一个string告诉用户出错了
def guess_date_format(sample_date_str, throw_exception_if_failed = False):

        guessed_format = 'guess_failed'
        sample_date_str = str(sample_date_str)

        # 涉及到日期的部分
        if "-" in sample_date_str:
            date_part = '%Y-%m-%d'
        elif "/" in sample_date_str:
            date_part = '%Y/%m/%d'
        elif len(sample_date_str) == 8:
            date_part = '%Y%m%d'
        else:
            date_part = ''

        # 涉及到时间的部分
        time_part = ''
        colon_ct = sample_date_str.count(":")
        if colon_ct == 2:
            time_part = "%H:%M:%S"
        elif colon_ct == 1:
            time_part = "%H:%M"


        # 合并日期和时间部分
        if date_part != '' and time_part == '':
            guessed_format = date_part
        elif date_part == '' and time_part != '':
            guessed_format = time_part
        elif date_part != '' and time_part != '':
            guessed_format = date_part + ' ' + time_part
        elif date_part == '' and time_part == '':
            # 其他特殊格式
            try:
                sample_date_str_num = float(sample_date_str)
                if sample_date_str_num < 50000:
                    guessed_format = 'excel'
                elif sample_date_str_num < 800000:
                    guessed_format = 'matlab'
                elif sample_date_str_num > 10000000 and sample_date_str_num < 30000000:
                    guessed_format = 'yyyymmdd_bad_rounding'
            except:
                guessed_format = 'guess_failed'

        if guessed_format == 'guess_failed' and throw_exception_if_failed:
            raise Exception("guess_date_format: 无法猜出时间格式: " + sample_date_str)
        else:
            return guessed_format


# -----------------------------------------
#         将时间转换成标准格式
# -----------------------------------------

def parse_time_format(raw_time_series, format_given):

    # 猜测数据格式
    if format_given == 'guess':
        if raw_time_series.shape[0] > 0:
            time_sample = raw_time_series.iloc[0]
            actual_format = guess_date_format(time_sample, throw_exception_if_failed=True)
            actual_format_is_standard = is_time_format_standard(actual_format)
        else:
            actual_format = '%Y%m%d'                                                                                                       # 哪怕没数据，这里也需要把列的格式转换成数据，不然后面对齐的时候会报错。在没数据的时候随便转转就行了，格式无所谓。
            actual_format_is_standard = True

    # 若不猜测格式，则将用户数标准化
    else:
        actual_format_is_standard = is_time_format_standard(format_given)
        actual_format,_ = convert_time_format_to_standard(format_given)

    if not actual_format_is_standard:
        if actual_format == 'excel':
            parsed_time = excel_date_to_datetime(raw_time_series)
        elif actual_format == 'matlab':
            parsed_time = excel_date_to_datetime(raw_time_series - 693960)                                                       #!!! This might need to be checked
        elif actual_format == ['yyyymmdd', 'hhmmss']:
            parsed_time = pd.to_datetime(raw_time_series.iloc[:, 0] * 1000000 + raw_time_series.iloc[:, 1], format='%Y%m%d%H%M%S')  # !!1 这里估计要测试
        elif actual_format == 'yyyymmdd_bad_rounding':
            parsed_time = pd.to_datetime(raw_time_series.values.astype(int).astype(str), format='%Y%m%d')
        elif actual_format == 'do_not_change':
            parsed_time = raw_time_series                                                                                       #   别碰原始数据

    else:
        try:
            parsed_time = pd.to_datetime(raw_time_series, format=actual_format)                                                      # 这是是pandas可以直接处理的
        except:
            try:
                parsed_time = pd.to_datetime(raw_time_series)
            except:
                print("错误： python和手动猜测的日期格式都不对, 第一个时间" + str(time_sampl) + "，手动猜测日期格式：" + actual_format)
            
    return parsed_time


# -----------------------------------------
# 检查输入的日期格式是否标为标准Python格式
# -----------------------------------------
def is_time_format_standard(time_format):
    non_standard_list = ['excel', 'matlab', ['yyyymmdd', 'hhmmss'], 'yyyymmdd', 'yyyy-mm-dd', 'yyyy/mm/dd', 'yyyymmdd_bad_rounding']
    if time_format in non_standard_list:
        format_is_stadard = False
    else:
        format_is_stadard = True

    return format_is_stadard

# -----------------------------------------
#           将日期格式标准化
# -----------------------------------------
def convert_time_format_to_standard(time_format):
    convert_dict = {'yyyymmdd': '%Y%m%d',
                    'yyyy/mm/dd': '%Y/%m/%d',
                    'yyyy-mm-dd': '%Y-%m-%d'}

    if type(time_format) is list:
        conversion_success = False
    else:
        conversion_success = time_format in convert_dict.keys()

    if conversion_success:
        converted_format = convert_dictp[time_format]
        conversion_success = True
    else:
        converted_format = time_format
        conversion_success = False

    return converted_format, conversion_success




# ------------------------------------------------
#                   缓存相关函数
# ------------------------------------------------
# 目的：某些函数运行一次需要很长时间（比如读取大量的数据后再对其）
#       可以考虑运行一次后将结果储存在硬盘上
#       下次运行时，直接读取储存结果，只要确保两次函数的输入都一样就行

# ！！！ 这些函数需要独立到另外一个模块里去


def get_func_input(output_from_locals, output_form_signature):
    function_input_dict = dict()
    for key in output_form_signature.parameters.keys():
        function_input_dict[key] = output_from_locals[key]
    return function_input_dict


def hash_dict(input_dict):
    key_str = ''
    value_str = ''
    for key in input_dict:
        key_str = key_str + key
        try:
            value_str = value_str + str(input_dict[key])
        except:
            data_type =  type(input_dict[key])
            data_type = data_type.__name__
            value_str = value_str + 'str_conversion_failed, data type is ' + data_type
            warnings.warn('hash_dict: 该key的值无法被转换成String： ' + key + '。其数据类型为：' + data_type + '. 签名结果可能不唯一。它的值是')

    hash_1 = hashlib.sha1(key_str.encode()).hexdigest() + hashlib.sha1(value_str.encode()).hexdigest()
    hash_value = hashlib.sha1(hash_1.encode()).hexdigest()
    return hash_value

def hash_function_input(output_from_locals, output_form_signature):
    function_input_dict = get_func_input(output_from_locals, output_form_signature)
    hash_value =  hash_dict(function_input_dict)
    return hash_value


# --------------------------------------
#       删除开始和结尾的重复性数据
# --------------------------------------

def remove_begin_end_idle(ts_np_input, return_one_var = True):

    ts_np = ts_np_input.copy()

    end_loc_list = []
    end_value_list = []

    if len(ts_np.shape) == 1:
        ts_np = np.expand_dims(ts_np, axis = 1)

    [date_ct, ts_ct] = ts_np.shape

    for j in range(ts_ct):

        found_first_begin = False
        found_begin_end = False

        found_first_end = False
        found_end_end = False

        end_loc_list.append(-1)
        end_value_list.append(-1)

        for i in range(date_ct):

            # 删除早期不变的数据
            if not np.isnan(ts_np[i,j]) and not found_first_begin:

                found_first_begin = True
                begin_value = ts_np[i,j]

            if (found_first_begin and not found_begin_end):

                if ts_np[i,j] == begin_value:
                    ts_np[i, j] = float('nan')
                else:
                    found_begin_end = True
                    ts_np[i-1, j] = begin_value

            # 删除尾部不变的数据
            if not np.isnan(ts_np[-(i + 1), j]) and not found_first_end:

                found_first_end = True
                end_value = ts_np[-(i + 1), j]

            if found_first_end and not found_end_end:
                if ts_np[-(i + 1), j] == end_value:
                    ts_np[-(i + 1), j] = float('nan')
                else:
                    found_end_end = True
                    ts_np[-(i), j] = end_value
                    end_loc_list[-1] = date_ct - i
                    end_value_list[-1] = end_value

    if return_one_var:
        return ts_np
    else:
        other_stuff = {'end_value_list': end_value_list, 'end_loc_list': end_loc_list}
        return ts_np, other_stuff






# ---------------------------------
#    将起点抬高至某条以便于比较
# ---------------------------------


def rescale_ts(ts_np_input):

    ts_np = ts_np_input.copy()
    (date_ct, ts_ct) = ts_np.shape


    # --------------------
    #    找出尾部节点
    # --------------------

    end_value_list = []
    end_loc_list = []


    for j in range(ts_ct):
        found_end = False
        end_value_list.append(-1)
        end_loc_list.append(-1)
        for i in range(date_ct):
            if not np.isnan(ts_np[-(i + 1), j]) and not found_end:
                end_loc_list[-1] = date_ct - i
                end_value_list[-1] = ts_np[-(i + 1), j]
                found_end = True


    # --------------------
    # 决定一定时间段内的主轴
    # --------------------
    main_ts = np.zeros(date_ct) - 1
    for i in range(date_ct):

        # Main is missing
        if main_ts[i] < 0:

            # Check if previous main is active
            new_main_needed = False
            if i == 0 or main_ts[i-1] < 0:
                new_main_needed = True
            elif np.isnan(ts_np[i, int(main_ts[i - 1])]):
                new_main_needed = True

            # new_main_needed
            if new_main_needed:
                candidate_ret = []
                for j in range(ts_ct):
                    # Find the one whose life represents the average return, and use that guy
                    if i < end_loc_list[j] and not np.isnan(ts_np[i, j]):
                        avg_ret = (end_value_list[j] - ts_np[i, j]) / (end_loc_list[j] - i)
                    else:
                        avg_ret = float('nan')
                    candidate_ret.append(avg_ret)

                if all(np.isnan(candidate_ret)):
                    main_ts[i] = -1
                    # Signal at the next time stop that we are still searching for candidates

                else:
                    main_loc, _ = fmu.first_extreme(np.abs(candidate_ret - np.nanmean(candidate_ret)), dir = 'min')
                    main_ts[i] = main_loc
            else:
                main_ts[i] = main_ts[i-1]


    # --------------------
    # 进行等比例放大
    # --------------------

    scale_value = []

    for i in range(date_ct):

        # 记录缩放目标
        if main_ts[i] < 0:
            if i == 0:
                scale_value.append(float('nan'))
            else:
                scale_value.append(scale_value[-1])
        else:
            scale_value.append(ts_np[i, int(main_ts[i])])

        # 缩放
        for j in range(ts_ct):

            # 如果是时间序列的第一天，则进行缩放
            need_rescale = False
            if i == 0:
                if not(np.isnan(ts_np[i, j])):
                    need_rescale = True
            elif np.isnan(ts_np[i - 1, j]) and not np.isnan(ts_np[i, j]):
                need_rescale = True

            # 进行缩放
            if need_rescale and not np.isnan(scale_value[i]):
                ts_np[:, j] = ts_np[:, j] * scale_value[i] / ts_np[i, j]

    return ts_np


# ------------------
#   加入一个平均值
# ------------------
# !!! 这个函数应该被移走
def nav_attach_mean(nav_np, where = 'left'):

    ret_ts = fmu.nav2ret(nav_np)
    avg_ret = np.nanmean(ret_ts, axis=1)
    avg_ret[np.isnan(avg_ret)] = 0
    avg_nav = remove_begin_end_idle(np.cumprod(avg_ret + 1))

    if where == 'left':
        ts_np_with_avg = np.concatenate([avg_nav, nav_np], axis=1)
    else:
        ts_np_with_avg = np.concatenate([nav_np, avg_nav], axis=1)

    return ts_np_with_avg


