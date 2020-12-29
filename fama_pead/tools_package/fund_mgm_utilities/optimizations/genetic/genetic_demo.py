# 遗传算法使用示例

import math
import numpy as np
import random
import fund_mgm_utilities as fmu
import matplotlib.pyplot as plt
import pandas as pd

# --------
# 目标函数
# --------
def obj_func(input_arr):
    output = 0
    for i in range(len(input_arr)):
        this_x = input_arr[i] - i
        output = output + math.pow(-1 * this_x, i) * math.sin(this_x)
    return output


# ----------------
# 函数式限制条件
# ----------------
# a < g(x) <= b
def constrain_1(input_arr):
    output = 0
    for i in range(len(input_arr)):
        this_x = math.pow(input_arr[i], 2)
        output = output + math.pow(-1 * this_x, i) * math.sin(this_x)
    return output

def constrain_2(input_arr):
    output = 0
    for i in range(len(input_arr)):
        this_x = input_arr[i] - i
        output = output + math.sqrt(abs(math.pow(-1 * this_x, i) * math.sin(this_x)))
    return output


func_csn_list = [constrain_1, constrain_2]
func_lb_list = [0, -float('inf')]
func_ub_list = [float('inf'), 20000]


# -------------
# 线性限制条件
# -------------
# lb < x < ub
input_ct = 10
lb = np.zeros(input_ct) * 0 - 20
ub = np.zeros(input_ct) * 0 + 20



# ------------------------------------------------------------------------------------------------------------------------------
#                                                   呼叫遗传算法
# ------------------------------------------------------------------------------------------------------------------------------
print(fmu.now('Begin running genetic algo: '))

initial_guess = np.zeros(input_ct) + 1

default_para = {'pop_ct': 100, 
                    'max_generation':100, 
                    'max_make_baby_attempts':20,
                    'kid_per_family':2, 
                    'max_initialization_attempts':20,
                    'elite_rate':0.1, 
                    'mutate_rate':0.2, 
                    'explore_step_size':[], 
                    'explore_surroundings_last_n_gen':[], 
                    'explore_step_ct': 20,
                    'repeat_ct' : 2, 
                    'initial_guess' : [], 
                    'use_prev_best' : False,
                    'record_progress':False, 
                    'display_progrss':True,
                    'must_satisfy_constrains': True}  
best_gene, best_fit, progress_report = fmu.genetic_opt(obj_func, lb, ub,
                      func_csn_list, func_lb_list, func_ub_list,
                   default_para)

print(fmu.now('Finished running genentic algo: '))

print('Log of final fit: ' + str(math.log(best_fit)))
print('Final fit:' + str(best_fit))
print('Best gene: ' + str(best_gene))

# plt.plot((np.array(progress_report['best_fit_hist'])))
# plt.plot((np.array(progress_report['avg_fit_hist'])))
# plt.plot((np.array(progress_report['std_fit_hist'])))

# plt.show()

