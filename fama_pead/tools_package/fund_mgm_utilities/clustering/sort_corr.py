import pandas as pd
import numpy as np
import matplotlib as mpl
import fund_mgm_utilities as fmu

# -------------------------------------------------------
#                   最优展示顺序
# -------------------------------------------------------

def get_orderd_name_list(group_info_list):

    ordered_name = []

    for group_info in group_info_list:

        if group_info['next_level'] == []:

            ordered_name = ordered_name + list(group_info['name'])

        else:

            ordered_name = ordered_name + get_orderd_name_list(group_info['next_level'])

    return ordered_name


# --------------------------------------
#   找到一个最受欢迎的成员，建立一个新的小组
# --------------------------------------

def find_new_group_founder(corr_mtx, method = 'avg_dist'):

    if method == 'avg_dist':
        proximity = np.average(corr_mtx, axis = 1)
        new_mem_num, new_mem_bool = fmu.first_extreme(proximity, dir = 'max')

    else:
        raise Exception('find_new_group_founder：不支持该招募方法: ' + str(method))

    return new_mem_num, new_mem_bool



# --------------------------------------
#           招募一个新的成员
# --------------------------------------

def recruit_new_member_4_group(group_bool, corr_mtx, min_corr = 0, recruit_criteria = 'all_dist_close'):

    pop_ct = corr_mtx.shape[0]
    idx = np.array(range(pop_ct))

    if recruit_criteria == 'all_dist_close':

        # 如果有人可招， 则尝试招人
        if any(np.logical_not(group_bool)):

            recruit_corr = corr_mtx[group_bool, :][:, np.logical_not(group_bool)]
            recruit_proximity = np.average(recruit_corr, axis = 0)

            # 与组内成员的最低相关性
            if recruit_corr.shape[0] > 1:
                min_corr_list = np.min(recruit_corr, axis = 0)
            else:
                min_corr_list = recruit_proximity

            # 与组内成员的最低相关性不能低于某个阈值
            recruit_qualified = min_corr_list > min_corr
            recruit_proximity[np.logical_not(recruit_qualified)] = float('nan')

            idx_recruit = idx[np.logical_not(group_bool)]
            if np.any(recruit_qualified):

                # 招最和睦的人
                new_recruit_num, new_recruit_bool = fmu.first_extreme(recruit_proximity, dir = 'max')

                # 将坐标转换为全局坐标
                new_recruit_num = idx_recruit[new_recruit_num]
                new_recruit_bool = np.zeros(pop_ct) > 0
                new_recruit_bool[new_recruit_num] = 0

                found_new_member = True

            else:
                # 有人可招，但与组内的成员都合不来
                found_new_member = False
        else:
            # 无人可招
            found_new_member = False

        # 没招到人
        if not(found_new_member):
            new_recruit_num = -1
            new_recruit_bool = np.zeros(pop_ct) > 0
            found_new_member = False


    # 与成员的平均相关性大于最低值
    elif method == 'all_dist_close':
        raise Exception('recruit_new_member_4_group: 该招募方法还没写完： all_dist_close')

    else:
        raise Exception('find_new_group_founder：不支持该招募方法: ' + str(method))

    # 返回要么新成员的位置，要么空集，表示无法找到新人
    return found_new_member, new_recruit_num, new_recruit_bool


# --------------------------------------
#           建立一个新的群体
# --------------------------------------
# 主流方法:
# 先找一个最受欢迎的人做创始人，然后开始招人，新人必须要和现有成员和睦相处。
# 一直招人，招到没法招为止

# 允许用户对一些细节进行修改，比如创始人的选法和招人方法


def establish_new_group(corr_mtx, min_corr, founder_method = 'avg_dist', recruit_criteria = 'all_dist_close'):

    # 找创始人
    founder_num, group_bool = find_new_group_founder(corr_mtx, method = founder_method)
    group_num = [founder_num]

    # 招人，一直招到没有合适的人可以招为止
    found_new_member = True

    while found_new_member:
        found_new_member, new_mem_num, new_member_bool = recruit_new_member_4_group(group_bool, corr_mtx,
                                                                                    min_corr=min_corr,
                                                                                    recruit_criteria=recruit_criteria)
        if found_new_member:
            group_num.append(new_mem_num)
            group_bool[new_mem_num] = True

    return group_num, group_bool


# --------------------------------------
#           进行一层聚类
# --------------------------------------

# 对于n个元素， 若它们之间的相似度用corr_mtx矩阵标记（越大越相似），则构建n组， n <=m, 每个元素刚好属于一组，且组员之间的相似度大于min_corr
# 通过调整 founder_method 和 recruit_criteria 可修改分类方式


def classify_one_layer(corr_mtx, min_corr, founder_method='avg_dist', recruit_criteria='all_dist_close', item_names = []):

    pop_ct = corr_mtx.shape[0]

    if type(item_names) is list:
        item_names = np.array(range(pop_ct))

    idx = np.array(range(pop_ct))

    group_list = []

    # 建立一个新的群体
    ungrouped_bool = np.zeros(pop_ct) == 0  # 未被分类的成员

    while any(ungrouped_bool):  # 只要有成员没有被分类，则继续分类

        ungrouped_corr_mtx = corr_mtx[ungrouped_bool, :][:, ungrouped_bool]

        # 构建新的群体
        group_num, group_bool = establish_new_group(ungrouped_corr_mtx, min_corr=min_corr, founder_method='avg_dist',
                                                    recruit_criteria='all_dist_close')

        # 将局部坐标转换成全局坐标
        group_num_global = idx[ungrouped_bool][group_num]
        group_bool_global = np.zeros(pop_ct) > 0
        group_bool_global[group_num_global] = True
        group_name_list = item_names[group_num_global]

        # 记录分类结果
        group_info = {'name': group_name_list, 'num': group_num_global, 'bool': group_bool_global}
        group_list.append(group_info)

        # 标记已分类成员
        ungrouped_bool[group_num_global] = False

    # 按组员数量排序

    return group_list



# --------------------------------------
#           进行多层聚类
# --------------------------------------

# 对于n个元素， 若它们之间的相似度用corr_mtx矩阵标记（越大越相似），则构建n组， n <=m, 每个元素刚好属于一组，且组员之间的相似度大于min_corr
# 通过调整 founder_method 和 recruit_criteria 可修改分类方式
# 对于每一组内部再次使用相同方法进行分类


def classify_multi_layer(corr_mtx, min_corr_list, founder_method='avg_dist', recruit_criteria='all_dist_close', item_names = [], auto_min_corr = False):


    # 如果指定的最阈值已经探索完毕，且用户批准自动探索，则继续探索
    if min_corr_list == [] and auto_min_corr:
        min_corr = recommand_min_corr(corr_mtx, founder_method = founder_method, recruit_criteria = recruit_criteria)
    else:
        min_corr = min_corr_list[0]

    pop_ct = corr_mtx.shape[0]

    if type(item_names) is list:
        item_names = np.array(range(pop_ct))

    idx = np.array(range(pop_ct))

    group_list = []

    # 建立一个新的群体
    ungrouped_bool = np.zeros(pop_ct) == 0  # 未被分类的成员

    while any(ungrouped_bool):  # 只要有成员没有被分类，则继续分类

        ungrouped_corr_mtx = corr_mtx[ungrouped_bool, :][:, ungrouped_bool]

        # 构建新的群体
        group_num, group_bool = establish_new_group(ungrouped_corr_mtx, min_corr=min_corr, founder_method='avg_dist',
                                                    recruit_criteria='all_dist_close')

        # 将局部坐标转换成全局坐标
        group_num_global = idx[ungrouped_bool][group_num]
        group_bool_global = np.zeros(pop_ct) > 0
        group_bool_global[group_num_global] = True
        group_name_list = item_names[group_num_global]


        # 进行下一层分类（如果没有分完所有层级，而且当前组员数量大于1）
        stop_digging = True
        if pop_ct > 1:              # 首先要有人可以分

            if len(min_corr_list) > 1:                   # 用户指定了分级，还没分完
                stop_digging = False
            elif len(min_corr_list) == 1:                # 指定分级用完了，但允许自动分级，且不知道自动分级效果好坏
                if auto_min_corr:
                    stop_digging = False
            elif len(min_corr_list) == 0:                # 自动分级，且上一层的自动分级有意义，那么这一层还值得试探一下
                if min_corr > np.min(corr_mtx):
                    stop_digging = False

        if not(stop_digging):
            next_level_group_info_list = classify_multi_layer(corr_mtx[:, group_num_global][group_num_global, :],
                                 min_corr_list[1:], founder_method=founder_method,
                                 recruit_criteria=recruit_criteria,
                                 item_names=item_names[group_num_global], auto_min_corr = auto_min_corr)

        else:
            next_level_group_info_list = []

        # 记录分类结果
        group_info = {'name': group_name_list, 'num': group_num_global, 'bool': group_bool_global, 'min_corr': min_corr, 'next_level': next_level_group_info_list}
        group_list.append(group_info)


        # 标记已分类成员
        ungrouped_bool[group_num_global] = False

    # 按组员数量排序

    return group_list


# --------------------------------------
#           推荐一个切割点
# --------------------------------------

# 在组数和组内和谐程度之间找一个平衡点

def recommand_min_corr(corr_mtx, res = 20, founder_method='avg_dist', recruit_criteria='all_dist_close'):

    group_ct_list = []
    group_dist_list = []

    max_corr = np.max(corr_mtx)
    min_corr = np.min(corr_mtx)

    corr_sweep_list = [min_corr]
    while(corr_sweep_list[-1]) < max_corr:
        corr_sweep_list.append(corr_sweep_list[-1] + (max_corr - min_corr) / res)

    for min_corr in corr_sweep_list:

        group_info_list = classify_multi_layer(corr_mtx, [min_corr], founder_method=founder_method, recruit_criteria=recruit_criteria, auto_min_corr = False)
        group_ct = len(group_info_list )

        group_corr_sum = []
        group_area_list = []

        for group_info in group_info_list:
            group_corr_sum.append(np.sum(corr_mtx[group_info['num'], :][:, group_info['num']]))
            group_area_list.append(len(group_info['num']) * len(group_info['num']))

        group_dist = np.sum(group_corr_sum) / np.sum(group_area_list)

        group_ct_list.append(group_ct)
        group_dist_list.append(group_dist)

    group_ct_list = np.array(group_ct_list)
    group_dist_list = np.array(group_dist_list)

    group_ct_list = fmu.normalize(group_ct_list)
    group_dist_list = fmu.normalize(group_dist_list)

    min_corr_quality = group_dist_list - group_ct_list

    best_min_corr_loc, _ = fmu.first_extreme(min_corr_quality)

    # 此时增加新组无用
    best_min_corr = corr_sweep_list[best_min_corr_loc]

    return best_min_corr


