import fund_mgm_utilities as fmu
import pandas as pd
import numpy as np
import collections
import warnings
import os


# ----------------------------------
#     手动更新列表里的所有文件
# ----------------------------------
# 允许用户互动式的手动录入数据至 csv 文件
# 如果文件分散在各地，可以避免到处找他们并一个个打开的繁琐工作


def update_ts_mannually(update_file_path_list, header_list = "auto"):

    new_df_list = []
    new_data_ct = 0
    ts_change_ct = 0

    for i in range(len(update_file_path_list)):

        i_file_path = update_file_path_list[i]

        if type(header_list) is list:
            header_setting = header_list[i]
        else:
            header_setting = 'auto'

        data_df, new_data_df = update_one_ts(i_file_path, header_setting = header_setting)
        new_df_list.append(new_data_df)

        if new_data_df.shape[0] > 0:
            ts_change_ct = ts_change_ct + 1
            new_data_ct = new_data_ct + new_data_df.shape[0]


    print('----------------------------')
    print('|       本次更新总结       |')
    print('----------------------------')
    for i in range(len(update_file_path_list)):
        print('文件： ' + update_file_path_list[i])
        print('新增数据： ')
        print(new_df_list[i])
        print(' ')

    print('总共有' + str(ts_change_ct) + "个文件发生变化")
    print('总共新增' + str(new_data_ct) + "条数据")


    print('----------------------------')
    print('|所有文件数据数据更新完毕! |')
    print('----------------------------')

    return True




# ----------------------------------------------
#    以与用户互动的方式更新一个dataframe的数据
# ----------------------------------------------

def update_one_ts(file_path, header_setting = "auto"):

    print('------------------')
    print('|  手动数据更新  |')
    print('------------------')

    print('准备手动更新该文件：'  + file_path)
    if os.path.exists(file_path):
        data_df, used_open_method, used_encoding, used_header = fmu.pd_readcsv_ts(file_path, header=header_setting, return_read_method = True)
        print('原始数据读取成功！')
    else:
        print('文件不存在！')



    continue_update_this_ts = True

    new_date_list = []
    new_nav_list = []
    while continue_update_this_ts:

        date_np = pd.to_datetime(data_df.iloc[:, 0]).values

        # 检查日期是否有重复复，如果有重复则警告并退出
        date_freq = fmu.tabulate(list(date_np))

        if any(date_freq['freq'] > 1):
            warnings.warn('以下日期有重复：')
            print(date_freq.loc[date_freq['freq'] > 1, :])
            continue_update_this_ts = False
            input('-->请手动修复原始数据后再使用本工具更新。按回车退出该时间序列的更新')

        else:
            nav_np = data_df.iloc[:, 1].values

            print('可以更新储存在这个文件里的数据了：' + file_path)
            print('目前最后5天的净值为：')
            print(data_df.iloc[max(-5, -data_df.shape[0] - 1):])

            today_date = fmu.today()
            user_input = input('-->请输入想添加的净值和日期，用逗号间隔。例如： 2019/1/1, 1.5， 若不输入日期，则默认净值为今日(' + today_date + ')的净值: ')

            input_split = user_input.split(',')
            if len(input_split) == 1:
                input_date = np.datetime64(fmu.today(convert_to_str=False))
                input_nav = user_input
            else:
                input_date = np.datetime64(pd.to_datetime(input_split[0]))
                input_nav = float(input_split[1])

            # -------------------------
            #      若用户想替换
            # -------------------------
            if input_date in date_np:

                replace_loc = (date_np == input_date)
                crn_value = nav_np[replace_loc]

                print('您输入的日期: ' + str(input_date) + ' 已存在于当前数据里面，其净值为：' + str(crn_value[0]))
                user_replace = input('-->请确认是否要替换掉原来的数据(y=是，其他输入=否): ')

                # 打印更新日前后的净值

                if user_replace.lower() == 'y':
                    replace_loc = (date_np == input_date)
                    date_np[replace_loc] = input_date
                    nav_np[replace_loc] = input_nav
                    new_data_entered = True

                    new_date_list.append(input_date)
                    new_nav_list.append(input_nav)

                else:
                    new_data_entered = False

            # ----------------------------------
            #      不替换现有数据，而是新增数据
            # ----------------------------------
            else:
                date_np = np.append(date_np, input_date)
                nav_np = np.append(nav_np, input_nav)
                new_data_entered = True

                new_date_list.append(input_date)
                new_nav_list.append(input_nav)

            columns = data_df.columns
            data_df = fmu.nav_list_2_dataframe(list(date_np), list(nav_np), date_format=None, columns = columns)
            data_df.sort_values(by = columns[0], inplace = True, ascending=True)

            if new_data_entered:
                print('新数据已更新，更新日期前后一段时间的净值为：')

                new_date_loc = (data_df.iloc[:, 0].values == input_date)
                neighbor_loc = fmu.mark_neighbors(new_date_loc, 2)

                neighbor_df = data_df.iloc[neighbor_loc, :]
                print(neighbor_df)

            else:
                print('您已决定放弃本次输入的数据')

            continue_update_this_ts = input("是否需要继续给这个文件添加新数据(y=是，其他输入=否): ")
            continue_update_this_ts = (continue_update_this_ts.lower() == 'y')


    print('该文件的新数据已录入完毕：' + file_path)

    # ---------------------------------------
    #    让用户决定是否将录入的数据写入文件
    # ---------------------------------------

    new_data_ct = len(new_date_list)
    print('您总共新增了（包括替换的）' + str(new_data_ct) + '个数据点：')

    if new_data_ct > 0:

        new_data_df = fmu.nav_list_2_dataframe(new_date_list, new_nav_list, date_format=None)
        print(new_data_df)

        output_data = input("是否将数据写至文件 (y=是，其他输入=否): ")
        if output_data.lower() == 'y':
            if used_header == 0:
                header_out = True
            else:
                header_out = used_header

            data_df.to_csv(file_path, header=header_out, index = False, encode='GBK')
            print('写入成功！')
        else:
            print('您已放弃将新数据写入文件')
            new_data_df  = pd.DataFrame([])

    else:
        new_data_df = pd.DataFrame([])
        print('原数据文件将不会有任何变化')

    return data_df, new_data_df


