import fund_mgm_utilities as fmu
import pandas as pd
import numpy as np
import os
import warnings
# 目的： 处理合并时间序列相关事宜





# -------------------------------
#       合并两个时间序列
# -------------------------------
# 主要用来比较两个时间序列是否一致
# max_error: 最大允许误差。 csv读写时会产生整数进位误差，经过各类计算后可能放大。默认值为:[1e-8](经验参数)
def merge_2_ts(df_new, df_old, give_warning_not_error = True, df_new_fix_time_format = True, df_old_fix_time_format = True, return_data_summary = False, max_error = float(1e-8), handle_conflict = 'give_error'):

    output_col_name = df_new.columns
    df_new.columns = ['日期', '新数据']
    df_old.columns = ['日期', '老数据']



    if df_new_fix_time_format:
        df_new['日期'] = pd.to_datetime(df_new['日期'])

    if df_old_fix_time_format:
        df_old.iloc[:, 0] = pd.to_datetime(df_old.iloc[:, 0])


    merged_df = pd.merge(df_new, df_old, how = 'outer', on = '日期')
    merged_df.sort_values(by = '日期', ascending = True, inplace = True)


    # 精确区分： 仅属于新数据或者老数据的日期
    #           新老数据都有的日期，以及这些日期的一致性
    left_has_data = np.logical_not(np.isnan(merged_df['新数据'].values))
    right_has_data = np.logical_not(np.isnan(merged_df['老数据'].values))

    left_only = np.logical_and(left_has_data, np.logical_not(right_has_data))
    right_only = np.logical_and(right_has_data, np.logical_not(left_has_data))
    both_has = np.logical_and(left_has_data, right_has_data)

    diff = merged_df['新数据'].values[both_has] / merged_df['老数据'].values[both_has] - 1
    full_diff = np.zeros(merged_df.shape[0])
    full_diff[both_has] = diff

    both_has_but_diff = full_diff > max_error

    merged_df['相对误差'] = full_diff
    merged_df['误差过大'] = both_has_but_diff



    # 新老数据的优先权。如果某些日期新老数据都有，用谁的
    if handle_conflict  == 'give_error':
        conflict_use = 'use_new'                         # 默认设置
    else:
        conflict_use = handle_conflict

    # 这样后面的稍微快一点
    if conflict_use == 'use_new':
        conflict_use_new = True
    elif conflict_use == 'use_old':
        conflict_use_new = False


    if sum(both_has_but_diff) > 0:

        print('merge_2_ts： 新老数据在某些日期不完全一致: ')
        print(merged_df.iloc[both_has_but_diff, :])

        if handle_conflict == 'give_error':
            raise Exception('merge_2_ts： 新老数据在某些日期不完全一致')
        else:
            warnings.warn('merge_2_ts： 新老数据在某些日期不完全一致，解决方案：' + conflict_use)

    # 合并数据
    merge_nav = []
    for i in range(merged_df.shape[0]):
        if left_only[i]:
            merge_nav.append(merged_df.iloc[i, 1])
        elif right_only[i]:
            merge_nav.append(merged_df.iloc[i, 2])
        elif both_has[i]:
            if conflict_use_new:
                merge_nav.append(merged_df.iloc[i, 1])
            else:
                merge_nav.append(merged_df.iloc[i, 2])



    data_dict = {output_col_name[0]: list(merged_df.loc[:, '日期']), output_col_name[1]: merge_nav}
    data_df = pd.DataFrame(data_dict)


    data_df = data_df[output_col_name]

    # 返回状态
    if return_data_summary:

        data_summary = {"老数据": df_old.shape[0],
                "新数据": df_new.shape[0],
                "仅新数据有": sum(left_only),
                "仅老数据有": sum(right_only),
                "新老数据都有": sum(both_has)}

        return data_df, data_summary
    else:
        return data_df

# ----------------------------------------------------------
# 比较一个dataframe和某个本地文件储存的dataframe是否一致
# ----------------------------------------------------------

def merge_ts_to_flie(new_df, out_file_path, print_setting = [], handle_conflict = 'give_error'):
    print_setting = fmu.initialize_print_setting(print_setting)
    fmu.print_status(print_setting, "begin", " 准备将新数据与该文件内的数据合并 " + out_file_path)

    # 检查输入的数据是否有日期重复日，如果有，直接报错，因为
    # 当日期有重复时，merge一次会让合并完的数据量暴增

    # 当降噪生产线参数改了后，需要覆盖之前的降噪时间序列，不需要时需comment out
    # if '降噪' in out_file_path:
    #     handle_conflict = 'use_new'

    repeat_ct_new = len(new_df.iloc[:, 0]) - len(new_df.iloc[:, 0].unique())
    if repeat_ct_new > 0:
        raise Exception('merge_ts_to_flie:  新数据有重复： ' + str(repeat_ct_new) + '。 老数据地址是： ' + out_file_path)


    # 如果本地文件不存在，或者大小为0，则假设本地不存在有效数据
    has_valid_local_data=False
    if os.path.isfile(out_file_path):
        if os.path.getsize(out_file_path) > 0:
            has_valid_local_data = True

    if has_valid_local_data:
        old_df = fmu.pd_readcsv_ts(out_file_path)

        repeat_ct_old = len(old_df.iloc[:, 0]) - len(old_df.iloc[:, 0].unique())
        if repeat_ct_old > 0:
            raise Exception('merge_ts_to_flie:  老数据有重复： ' + str(repeat_ct_old) + '。 老数据地址是： ' + out_file_path)


        merged_df, data_summary = merge_2_ts(new_df.copy(), old_df.copy(), df_new_fix_time_format = False, df_old_fix_time_format = True, return_data_summary=True, handle_conflict = 'use_new')
        fmu.print_status(print_setting, "mid", " 新老数据合并完毕, 没有发现净值不一致的日期。各类数据数量: " + fmu.dict2str(data_summary))
    else:
        merged_df = new_df
        fmu.df_to_csv(merged_df, out_file_path, index=False)
        fmu.print_status(print_setting, "mid", " 本地数据文件不存在!已将新数据写入本地")


    fmu.print_status(print_setting, "end")
    return merged_df