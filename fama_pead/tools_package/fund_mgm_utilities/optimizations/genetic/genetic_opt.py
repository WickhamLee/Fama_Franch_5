import math
import numpy as np
import warnings
import random
import fund_mgm_utilities as fmu
import matplotlib.pyplot as plt


# =======================================================================================
#                                   遗传算法优化
# =======================================================================================

# 目的：一个通用的优化算法，其中混入了一部分爬坡式贪心算法的元素，用于解决非线性凹优化问题

# ----------------------------------
# 检查函数式限制条件是否已经被满足
# ----------------------------------
def function_constrain_satisfied(func_csn_list, func_lb_list, func_ub_list, func_input):
    constrain_satisfied = True
    for i in range(len(func_csn_list)):
        i_func = func_csn_list[i]
        i_output = i_func(func_input)
        i_satisfied = (i_output <= func_ub_list[i]) & (i_output >= func_lb_list[i])
        if not i_satisfied:
            constrain_satisfied = False
            break
    return constrain_satisfied


# ----------------
# 随即生成一代人
# ----------------
def initialize_pop(func_csn_list, func_lb_list, func_ub_list, lb, ub, pop_ct, max_initialization_attempts, must_satisfy_constrains, initial_solution=[]):

    ini_pop_list = []
    for i in range(0, pop_ct):
        gen_attempt = 0
        while gen_attempt <= max_initialization_attempts:
            gene_i = np.random.uniform(lb, ub)
            if function_constrain_satisfied(func_csn_list, func_lb_list, func_ub_list, gene_i):
                break
            else:
                gen_attempt = gen_attempt + 1

        if gen_attempt > max_initialization_attempts and not function_constrain_satisfied(func_csn_list, func_lb_list, func_ub_list, gene_i):
            if i == 0:
                if initial_solution == []:
                    if must_satisfy_constrains:
                        raise Exception('We can not even find a single value that satisfy the constrains, what kind of wierd solution space you have got?')
                    else:
                        print('We can not even find a single value that satisfy the constrains, but we are going to go head with the algo anyway. Constrain ignored this time')
                else:
                    gene_i = initial_solution
            else:
                gene_i = prev_gene

        ini_pop_list.append(gene_i)
        prev_gene = gene_i

    return ini_pop_list


# ------------
#  生一个孩子
# -----------
def make_a_bady(mom, dad, max_make_baby_attempts, func_csn_list, func_lb_list, func_ub_list):

    this_baby_attempt = 1
    while this_baby_attempt <= max_make_baby_attempts:
        kid = np.random.uniform(mom, dad)
        if function_constrain_satisfied(func_csn_list, func_lb_list, func_ub_list, kid):
            break
        else:
            this_baby_attempt = this_baby_attempt + 1

    if this_baby_attempt > max_make_baby_attempts:
        kid = mom

    return kid


# ----------
# 生一代孩子
# ----------
def produce_offsprings(current_pop, max_make_baby_attempts, kid_per_family, func_csn_list, func_lb_list, func_ub_list):
    kid_list = []
    pop_ct = len(current_pop)
    for i in range(pop_ct):
        mom = current_pop[i]
        dad = current_pop[random.randint(0, pop_ct - 1)]
        for j in range(kid_per_family):
            kid = make_a_bady(mom, dad, max_make_baby_attempts, func_csn_list, func_lb_list, func_ub_list)
            kid_list.append(kid)
    return kid_list


# ------------------------
#  从一代人中筛选一部分人
# ------------------------
def keep_best_genes(population, obj_func, keep_ct):

    if keep_ct > 0:
        # 给人群打分
        fit_list = calc_pop_fitness(population, obj_func)

        # 筛选出合适的基因
        survived_gene_no = sorted(range(len(fit_list)), key=lambda i: fit_list[i])[-keep_ct:]
        survived_gene_list = [population[i] for i in survived_gene_no]
    else:
        survived_gene_list = []

    return survived_gene_list

# ------------------------
# 给一代人中的每个人打分
# ------------------------
def calc_pop_fitness(pop, obj_func):
    fit_list = []
    for i in range(len(pop)):
        fit_i = obj_func(pop[i])
        fit_list.append(fit_i)
    return fit_list


# ----------------------
#  让一个人切探索周边
# ----------------------
# For each member of the population, move each component by a certain distance between 0 and explore_step_size, for explore_step_ct number of times
# Keep the best moved result that still satisfied constrain, if not, revert to the beginning

# We have tried a more targeted grandient descend method, but, 1: numerial slope takes forever, 2: it doesn't imporove the overall value

def explore_surrounding(gene, explore_step_size, explore_step_ct, obj_func, lb, ub, func_csn_list, func_lb_list, func_ub_list):
    crn_best_fit = obj_func(gene)
    crn_best_gene = gene
    var_ct = len(gene)

    for i in range(explore_step_ct):

        move = np.random.uniform(np.zeros(var_ct)-explore_step_size, np.zeros(var_ct)+explore_step_size)
        new_gene = gene + move

        current_move_broke_bounds = True
        current_move_improved_fit = False

        if all(new_gene >= lb):
            if all(new_gene <= ub):
                if function_constrain_satisfied(func_csn_list, func_lb_list, func_ub_list, new_gene):
                    current_move_broke_bounds = False
                    current_move_improved_fit = True
                    new_fit = obj_func(new_gene)
                    if new_fit > crn_best_fit:
                        crn_best_gene = new_gene
                        crn_best_fit = new_fit

        if current_move_broke_bounds:
            explore_step_size = explore_step_size / math.sqrt(2)
        else:
            if current_move_improved_fit:
                explore_step_size = explore_step_size * math.sqrt(2)#explore_step_size * math.sqrt(2)

    return crn_best_gene

# -----------------------------
#  让一代人中的每个人去探索周边
# -----------------------------
def pop_explore_surroundings(pop, explore_step_size, explore_step_ct, obj_func, lb, ub, func_csn_list, func_lb_list, func_ub_list):
    for i in range(len(pop)):
        pop[i] = explore_surrounding(pop[i], explore_step_size, explore_step_ct, obj_func, lb, ub, func_csn_list, func_lb_list, func_ub_list)
    return pop


# ----------------------
#     演化一次
# ----------------------
def genetic_opt_once(obj_func, lb, ub,
             func_csn_list, func_lb_list, func_ub_list,
             pop_ct=100, max_generation=100, max_make_baby_attempts=20, kid_per_family=2, max_initialization_attempts=20, must_satisfy_constrains = True,
             elite_rate=0.1, mutate_rate=0.2, explore_step_size=[], explore_surroundings_last_n_gen=[], explore_step_ct= 20,
             initial_guess = [],
             record_progress=False, display_progrss=True):

    # 初始化周边探索设定
    # ！！！ 参数初始化设定需要用dictionary整理，目前的方法太乱了
    if explore_surroundings_last_n_gen == []:
        explore_step_size = (ub - lb)/100
        explore_surroundings_last_n_gen = math.floor(max_generation/10)

    # 初始化人群
    current_pop = initialize_pop(func_csn_list, func_lb_list, func_ub_list, lb, ub, pop_ct, max_initialization_attempts, must_satisfy_constrains)
    
    # 如果用户提供了一个初始解，则将初始解加入。
    if initial_guess != []:
        current_pop[0] = initial_guess


    # 决定下一代人中，多少来自：自然生产，精英直接无性繁殖，变异
    pop_size = len(current_pop)
    elite_ct = math.floor(pop_size * elite_rate)
    mutate_ct = math.floor(pop_size * mutate_rate)
    keep_kid_ct = pop_size - elite_ct - mutate_ct

    if record_progress:
        best_fit_hist = []
        avg_fit_hist = []
        std_fit_hist = []

    # 开始进化
    for i in range(max_generation):

        # 基因混合，进行随机式探索
        off_springs_list = produce_offsprings(current_pop, max_make_baby_attempts, kid_per_family, func_csn_list, func_lb_list, func_ub_list)                                           # 自然生产的所有下一代：孩子

        survived_kid_list = keep_best_genes(off_springs_list, obj_func, keep_kid_ct)                                                                                                    # 存活的下一代
        elite_list = keep_best_genes(current_pop, obj_func, elite_ct)                                                                                                                   # 精英
        mutated_list = initialize_pop(func_csn_list, func_lb_list, func_ub_list, lb, ub, mutate_ct, max_initialization_attempts, must_satisfy_constrains, initial_solution=off_springs_list[0])                  # 变异

        # 探索周边, 类似于gradient descend
        if max_generation - i < explore_surroundings_last_n_gen:
            elite_list = pop_explore_surroundings(elite_list, explore_step_size, explore_step_ct, obj_func, lb, ub, func_csn_list, func_lb_list, func_ub_list)                          # 改进后的精英
            survived_kid_list = pop_explore_surroundings(survived_kid_list, explore_step_size, explore_step_ct, obj_func, lb, ub, func_csn_list, func_lb_list, func_ub_list)            # 改进后的孩子
            mutated_list = pop_explore_surroundings(mutated_list, explore_step_size, explore_step_ct, obj_func, lb, ub, func_csn_list, func_lb_list, func_ub_list)                      # 改进后的变异人

        current_pop = survived_kid_list + elite_list + mutated_list                                                                                                                     # 下一代人

        # 记录中间结果
        if record_progress:
            this_gen_fitness = calc_pop_fitness(current_pop, obj_func)                                                                                                                  # 这代最优x
            this_gen_best_fit = max(this_gen_fitness)                                                                                                                                   # 这代最优f(x)
            this_gen_avg_fit = np.mean(this_gen_fitness)
            this_gen_std_fit = np.std(this_gen_fitness)

            best_fit_hist.append(this_gen_best_fit)
            avg_fit_hist.append(this_gen_avg_fit)
            std_fit_hist.append(this_gen_std_fit)

            if display_progrss:
                print('第' + str(i) + '代最优解： ' + str(this_gen_best_fit))

    # 输出最终的最优解
    final_gen_fitness = calc_pop_fitness(current_pop, obj_func)
    final_gen_best_fit = max(final_gen_fitness)
    best_gene_final_gen = current_pop[final_gen_fitness.index(final_gen_best_fit)]

    if record_progress:
        progress_report = {'best_fit_hist': best_fit_hist,
                           'avg_fit_hist': avg_fit_hist,
                           'std_fit_hist': std_fit_hist}
    else:
        progress_report = dict()

    return best_gene_final_gen, final_gen_best_fit, progress_report

# ---------------------
#       主函数
# ---------------------
# 遗传算法优化

# 输入参数
#   参数                                   描述               类型
# obj_func：                             目标函数            function
# lb, ub:                              最大，最小值           list


# 输出参数
# best_gene_final_gen            最优解的目标函数输出值       float


# 待开发： 子函数更多的注释
#          输入参数用dict来维护
#def geneetic_opt
    



def genetic_opt(obj_func, lb, ub,
             func_csn_list, func_lb_list, func_ub_list, para):

    # Merge user input para with default paras
    para = {**get_gene_default_para(), ** para}
    
    # Release parameters
    # A better way to do this would have been to use classes
    # But the code was written before Sam is used to dealing with classes in python
    # You are welcome to refractor this if you have the time
    # Exec does to work here. The released variable remaines unaccessible 
    pop_ct = para['pop_ct']
    max_generation = para['max_generation']
    max_make_baby_attempts = para['max_make_baby_attempts']
    kid_per_family = para['kid_per_family']
    max_initialization_attempts = para['max_initialization_attempts']
    elite_rate = para['elite_rate']
    mutate_rate = para['mutate_rate']
    explore_step_size = para['explore_step_size']
    explore_surroundings_last_n_gen = para['explore_surroundings_last_n_gen']
    explore_step_ct = para['explore_step_ct']
    repeat_ct = para['repeat_ct' ]
    initial_guess = para['initial_guess'] 
    use_prev_best = para['use_prev_best']
    record_progress = para['record_progress']
    display_progrss = para['display_progrss']
    must_satisfy_constrains = para['must_satisfy_constrains']
        
        
    # 多演化几次，以期待最优结果
    for i in range(repeat_ct):

        if i > 0 and use_prev_best:
            initial_guess = best_gene_final_gen_i

        best_gene_final_gen_i, final_gen_best_fit_i, progress_report_i = genetic_opt_once(obj_func, lb, ub,
                         func_csn_list, func_lb_list, func_ub_list,
                         pop_ct=pop_ct, max_generation=max_generation, max_make_baby_attempts=max_make_baby_attempts, kid_per_family=kid_per_family, max_initialization_attempts=max_initialization_attempts, must_satisfy_constrains = must_satisfy_constrains,
                         elite_rate=elite_rate, mutate_rate=mutate_rate, explore_step_size=explore_step_size, explore_surroundings_last_n_gen=explore_surroundings_last_n_gen, explore_step_ct=explore_step_ct,
                         initial_guess = initial_guess,
                         record_progress=record_progress, display_progrss=display_progrss)

        if i == 0:
            best_gene_final_gen = best_gene_final_gen_i
            final_gen_best_fit = final_gen_best_fit_i
            progress_report = progress_report_i
        elif final_gen_best_fit_i > final_gen_best_fit:
            best_gene_final_gen = best_gene_final_gen_i
            final_gen_best_fit = final_gen_best_fit_i
            progress_report = progress_report_i

        if display_progrss:
            print("遗传算法完成第" + str(i+1) + '次演化尝试，当前最优解：' + str(final_gen_best_fit))

    return best_gene_final_gen, final_gen_best_fit, progress_report


# -----------------------------
#    get_gene_default_para
# -----------------------------
# Returns the default parameters for the genetic algorithm
def get_gene_default_para():
    
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
                    'must_satisfy_constrains': True}        # If set to false, after max_initialization_attempts failed attemps, it will keep going instead of throwing an error
    
    return default_para
    
    
    
    
    
