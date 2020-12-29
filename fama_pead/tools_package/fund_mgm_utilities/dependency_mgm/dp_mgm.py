import pandas as pd
import numpy as np
import math
import fund_mgm_utilities as fmu



# 目的：当有多个任务待完成，而其中某些任务必须在另一些任务完成后才能被完成时，如何合理的安排任务顺序，使任务完成不发生冲突？

# 待解决问题

# 同时完成和依赖性会有些冲突，依赖性的完整描述应该是： 依赖于这些任务 或者 其他任务同组任务被完成。 忽略第二个条件会导致依赖性检查误判


# ------------------------------------
#   将依赖性列表转换成任务顺序
# ------------------------------------
# 输入参数： task_list 任务列表
#            dp_list 每个任务的依赖列表
#            sep: 依赖列表中的区分符号
#            cannot_run_task: 标注一些不能跑的任务



def dp_list_2_task_order(task_list, dp_list, group_list = [], sep = ',', canrun_mark = [], show_status = False):

    dp_mtx = dp_list_2_dp_mtx(task_list, dp_list, sep)
    full_task_order = dp_mtx_2_task_order(dp_mtx)

    # 对于可以一起完成的任务，只执行一次
    if group_list == []:
        return full_task_order
    else:
        short_task_order = task_list_remove_repeat(full_task_order, group_list, canrun_mark = canrun_mark, show_status = show_status)
        return short_task_order, dp_mtx

# ------------------------------------
#   将依赖性列表转换成依赖性矩阵
# ------------------------------------

def dp_list_2_dp_mtx(task_list, dp_list, sep = ','):

    task_ct = len(task_list)
    
    if task_ct != len(set(task_list)):
        raise ValueError('任务列表有重复!')

    dp_mtx = np.zeros([task_ct, task_ct])
        
    for i in range(task_ct):
    
        ith_dp_list = dp_list[i]
        
        # 如果输入的原始内容还不是list
        if not(type(ith_dp_list) is list):
            if fmu.isnan(ith_dp_list):                      # 如果没有依赖性
                ith_dp_list = []
            else:
                ith_dp_list = ith_dp_list.split(sep)
    
        for dp_task in ith_dp_list:
            if dp_task not in task_list:
                raise ValueError('任务 ' + task_list[i] + ' 依赖 ' + dp_task + ' 但 ' + dp_task + '并不在任务列表中。任务列表是 ' + str(task_list))
            dp_mtx[i, task_list.index(dp_task)] = 1

    return dp_mtx

# ------------------------------------------------
#           删除不需要的重复性任务
# ------------------------------------------------

# 当完成A时可以同时把B也做完时，做完A后将不会执行B

# ordered_task_list: 安排好的任务列表，如[3, 0, 2, 1]
# group_list: 每个任务是否属于某一同时被完成的分组标记。如果两个任务会被同时完成，则会被编为同一组，否则为nan. 如[1, nan, 1, nan]

# 输出：简化后的任务顺序：例如： [3, 0 , 1]
def task_list_remove_repeat(task_order, group_list, show_status = False, canrun_mark = []):

    tasks_to_remove = []


    if show_status:
        print('原始任务（移除等效任务前）顺序')
        print(task_order)

    for i in range(len(task_order)):
        group_for_i = group_list[task_order[i]]

        if not fmu.isnan(group_for_i) and not (i in tasks_to_remove):
            for j in range(i + 1, len(task_order)):
                if group_list[task_order[j]] == group_for_i and (task_order[j] not in tasks_to_remove):

                    # 任务i已经被完成，而任务j排在任务i后面，但又和任务i为等效任务，我们则可以考虑移除任务j。

                    canrun_mark_creates_issue = False
                    if canrun_mark != []:
                        if canrun_mark[task_order[i]] == False and canrun_mark[task_order[j]] == True:
                            canrun_mark_creates_issue = True

                    if canrun_mark_creates_issue:
                        tasks_to_remove.append(task_order[i])
                        if show_status:
                            print('既然已经完成了 ' + str(task_order[i]) + '，那么就不需要完成 :' + str(task_order[j]))
                            print('但是任务 ' + str(task_order[i]) + ' 处于不可执行状态，而任务 :' + str(task_order[j]) + ' 可以被执行，我们决定移除前者。')
                            print('请注意做依赖性检查，或者保证所有任务都处于可以执行状态')

                    else:
                        tasks_to_remove.append(task_order[j])
                        if show_status:
                            print('既然已经完成了 ' + str(task_order[i]) + '，那么就不需要完成 :' + str(task_order[j]))


    shortened_task_list = [x for x in task_order if x not in tasks_to_remove]

    return shortened_task_list

# ----------------------------------
#       重构完整的任务列表
# ----------------------------------

# 如果任务顺序是A，C, 而完成A的同时会完成B，则完整任务顺序是: A, B, C
# 输入：
# short_task_order = [0, 2]
# group_list = [1,1, nan]
# 输出：
# full_task_order = [0, 1, 2]

def short_task_order_2_full(short_task_order, group_list):

    total_tasks_ct = len(group_list)

    full_task_order = []                                                       # 重新构建一个完整的任务列表

    for i in range(len(short_task_order)):

        ith_task = short_task_order[i]
        ith_group = group_list[ith_task]

        full_task_order.append(ith_task)                                       # 先完整ith_task, 然后再去完整剩下的会和i同时被完成的

        if not fmu.isnan(ith_group):                                           # 如果这个任务属于某一组
            for j in range(total_tasks_ct):                                    # 则检查这一组内的其他任务
                if group_list[j] == ith_group and j not in short_task_order:   # 是否在当前的简化任务列表中
                    full_task_order.append(j)                                  # 如果不在简化的任务列表中， 则降它补回至完整任务列表

    return full_task_order



# ------------------------------------------------
#   在依赖矩阵的基础上生成合适的任务执行顺序
# ------------------------------------------------

def dp_mtx_2_task_order(dp_mtx):

    task_ct = dp_mtx.shape[0]
    task_order = []
    for i in range(task_ct):
        if i not in task_order:
            task_order, stack_check = add_dp_to_task_order(i, dp_mtx, task_order, task_ct, 0)  # 先查查完成任务j前需要完成其他哪些任务
            task_order.append(i)                                                             # 再完成任务 i

    return task_order


# --------------------------------------------------------------------
#  子函数： 在已完成的任务基础上，查询完成某一任务前需要先完成哪些任务
# ---------------------------------------------------------------------
# 返回新的完整的任务顺序列表

def add_dp_to_task_order(task_no, dp_mtx, completed_task_order, task_ct, stack_check):

    if task_no not in completed_task_order:

        # 确保不被循环依赖卡死
        stack_check = stack_check + 1
        if stack_check > (task_ct + 1):
            raise ValueError('总共只有' + str(task_ct + 1) + ' 个任务需要完成，但是在搜索依赖性的时候已经搜索了 ' + str(stack_check) + '层了。检查是否存在循环依赖! ' \
                             + '目前可以正常更新的是' + str(completed_task_order) + " 但是在更新任务 " + str(task_no) + " 时侦测到了循环依赖")

        for j in range(task_ct):
            # 需要完成任务j, 且任务j不在已完成的任务列表中
            if dp_mtx[task_no, j] == 1 and j not in completed_task_order:
                completed_task_order, stack_check = add_dp_to_task_order(j, dp_mtx, completed_task_order, task_ct, stack_check)         # 先查查完成任务j前需要完成其他哪些任务
                completed_task_order.append(j)                                                                                        # 再完成任务 j

    return completed_task_order, stack_check


# ----------------------------------------
#       检查任务顺序是否有问题
# ----------------------------------------
def check_task_order_4_dp_mtx(dp_mtx, task_order, group_list = []):
    order_is_good = True
    message = "任务顺序检查无误，未发现在更新中遇依赖于未完成的任务的情况"

    task_ran_ct = len(task_order)
    total_task_ct = dp_mtx.shape[0]
    for i in range(task_ran_ct):

        ith_task_no = task_order[i]
        ith_pre_comp = task_order[0: i]

        for j in range(total_task_ct):
            if dp_mtx[ith_task_no, j] == 1 and j not in ith_pre_comp:

                # 任务 ith_task_no 对 任务 j 有依赖性，但是j却不在已完成的列表中ith_pre_comp
                # 此时需要检查是否有和j等效的任务已经被完成 （如果任务a,b,c中任何一个任务被完成都会导致a,b,c被全部完成， 则称a,b,c为等效任务）
                equivelent_task_completed = False
                for completed_task in ith_pre_comp:
                    if group_list[completed_task] == group_list[j]:
                        equivelent_task_completed = True                        # 某个已完成的任务和有待完成的任务j等效，所以不存在依赖性问题
                        break

                if not equivelent_task_completed:
                    order_is_good = False
                    message = '更新顺序有问题: 在完成 ' + str(ith_task_no) + ' 时我们需要 ' + str(j) + ' 但此时仅完成了: ' + str(ith_pre_comp)
                    return order_is_good, message

    return order_is_good, message


# -----------------------------------------------------------------------------------------------------------------
#                           如果某一个任务的值发生改变，计算一下其它哪些任务的值也会改变
# -----------------------------------------------------------------------------------------------------------------

def calc_impact_path(dp_mtx, initial_impact):
    task_ct = dp_mtx.shape[0]
    crn_impact_vec = np.empty((task_ct), dtype=bool)
    crn_impact_vec[:] = False
    crn_impact_vec[initial_impact] = True

    impact_record = crn_impact_vec

    it_ct = 0
    while crn_impact_vec.any():
        it_ct = it_ct + 1
        crn_impact_vec = np.any(dp_mtx[:, crn_impact_vec], axis=1)
        impact_record = np.logical_or(impact_record, crn_impact_vec)
        if it_ct > task_ct:
            raise Exception('经过' + str(it_ct) + '次迭代后仍然没有查遍所有的依赖性，请检查依赖性矩阵是否有循环逻辑')

    return impact_record


# -----------------------------------------------------------------------
#  计算一个影响矩阵。矩阵的第i, j 元素 = True 表示若i发生变化最终j也会发生变化
# -----------------------------------------------------------------------
def calc_impact_matrix(dp_mtx):
    impact_mtx = np.full_like(dp_mtx, fill_value=False)

    for task_no in range(dp_mtx.shape[0]):
        impact_mtx[:, task_no] = calc_impact_path(dp_mtx, task_no)

    return impact_mtx


# --------------------------------------
#        从任务依赖性列表计算出影响矩阵
# --------------------------------------

def dp_list_2_impact_matrix(task_list, task_depend_list):
    dp_mtx = fmu.dp_list_2_dp_mtx(task_list, task_depend_list) == 1
    return calc_impact_matrix(dp_mtx)


# ----------------------------------------
#       需要测试时可以运行此函数
# ----------------------------------------

def dp_mgm_demo():

    para_path = fmu.__path__[0] + r'\dependency_mgm\dependency_mgm_test_input.csv'

    # 读取参数
    para_df = pd.read_csv(open(para_path))

    # 将参数格式换成成可接受格式
    task_list = list(para_df['任务'])
    dp_list = list(para_df['依赖于'])
    group_list = list(para_df['同时完成分组'])

    print('任务列表：')
    print(task_list)

    print("依赖性列表: ")
    print(dp_list)

    print("重复性分组列表: ")
    print(group_list)

    # 生成依赖性矩阵
    dp_mtx = dp_list_2_dp_mtx(task_list, dp_list)


    print("依赖性矩阵：")
    print(dp_mtx)

    # 使用依赖性矩阵生成任务顺序
    task_order = dp_mtx_2_task_order(dp_mtx)

    print("任务顺序：")
    print(task_order)

    # 检查任务熟顺序是否合理
    order_good, message = check_task_order_4_dp_mtx(dp_mtx, task_order)

    print("任务顺序合理性检查：")
    print(message)


    print("删除重复后的任务顺序：")
    task_order_no_repeat = task_list_remove_repeat(task_order, group_list)

    print(task_order_no_repeat)

    print("从简化任务列表重新构建的完整任务顺序：")
    task_order_full = short_task_order_2_full(task_order_no_repeat, group_list)
    print(task_order_full)

    # 检查任务熟顺序是否合理
    order_good, message = check_task_order_4_dp_mtx(dp_mtx, task_order_full)

    print('再次检查任务顺序：')
    print(message)



