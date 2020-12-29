import pandas as pd
import numpy as np
import fund_mgm_utilities as fmu

import os
project_folder = os.path.realpath(__file__)
project_folder = project_folder.replace(project_folder.split('\\')[-1], "")



# -------------------------------------------------------
#               时间序列分类算法 展示
# -------------------------------------------------------
data = pd.read_csv(open(project_folder + r"\performance.csv"), encoding = 'utf-8', index_col = 0, header = 0)

# 读取数据

# 日期
date_ts = data.index

# 时间序列名称
ts_names = list(data.columns)

# 净值历史
ts_np = data.values

ts_np = fmu.remove_begin_end_idle(ts_np)                # 移除开始和结尾的不变净值。 如果净值一开始长期不变，或者最后阶段长期不变，则移除他们。我们不希望比较建仓期的相关性


# 计算相关性

ret_np = ts_np[1:,] / ts_np[:-1,] - 1

ts_ct = ret_np.shape[1]

corr_mtx = np.zeros([ts_ct, ts_ct])
for i in range(ts_ct):
    for j in range(ts_ct):
        x = ret_np[:, i]
        y = ret_np[:, j]
        valid = np.logical_not(np.isnan(x) | np.isnan(y))
        corr_value = np.corrcoef(x[valid], y[valid])
        corr_value =  corr_value[0, 1]
        if np.isnan(corr_value):
            corr_value = -1

        corr_mtx[i, j] = corr_value

print("correlation calculted")

# 进行聚类分析
min_corr_list = []

founder_method = 'avg_dist'
recruit_criteria = 'all_dist_close'

group_info_list = fmu.classify_multi_layer(corr_mtx, min_corr_list, founder_method=founder_method, recruit_criteria=recruit_criteria, item_names = ts_names, auto_min_corr = True)

print("clustering order calcualted")

ordered_name_list = fmu.get_orderd_name_list(group_info_list)


ts_names_sorted = np.array(ts_names)[ordered_name_list]
sorted_corr = corr_mtx[ordered_name_list,:][:, ordered_name_list]

for i in range(sorted_corr.shape[0]):
    for j in range(sorted_corr.shape[1]):
        if sorted_corr[i,j] == -1:
            sorted_corr[i, j] = float('nan')

out_df = pd.DataFrame(sorted_corr, columns = ts_names_sorted)
out_df.index = ts_names_sorted

out_df.to_csv(project_folder + r'\sorted_result.csv', encoding = 'GBK')




