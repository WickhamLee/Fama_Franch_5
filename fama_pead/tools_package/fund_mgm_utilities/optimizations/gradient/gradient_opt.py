import numpy as np
from math import *
from functools import reduce

# =======================================================================================
#                                       梯度上升算法
# =======================================================================================

# 功能：梯度上升算法，用于寻找目标函数一定范围内的极大值点
# 输入：
#           obj_func()          function          目标函数f(x1,x2,...,xn)
#           lb                  1*n array         (x1,x2,...,xn)的下界
#           ub                  1*n array         (x1,x2,...,xn)的上界
#           csn_func_list       list              约束函数列表
#           func_lb_list        list              约束函数的下界列表
#           func_ub_list        list              约束函数的上界列表
#           para                dict              算法参数
#
# 输出：
#           x_old               1*n array         目标函数取极大值时的(x1,x2,...,xn)


# -------------------------------------
#                主函数
# -------------------------------------

def gradient_opt (obj_func, lb, ub, csn_func_list, func_lb_list, func_ub_list, para):
    
    # 合并输入的参数与默认参数
    para = {**get_grad_default_para(), ** para} 
    
    learning_rt = para['learning_rt']         # 学习率
    repeat_ct = para['repeat_ct']             # 循环次数
    precision = para['precision']             # 精度
    ini_values = para['ini_values']           # 初始值
    ini_ct = para['ini_ct']                   # 取几组初始值
        
    # 初始化自变量
    x_old_list = get_grad_x_values(lb, ub, ini_values, ini_ct) 
    x_opt = x_old_list[0].copy()
    
    for ct in range(ini_ct):
        
        x_old = x_old_list[ct].copy()
        x_new = x_old.copy()
        
        # 初始化因变量、约束函数
        y_old = obj_func(x_old)
        csn_func = func_lb_list.copy()    
    
        # 梯度上升算法
        for r in range(repeat_ct):           
        
            for i in range(len(x_old)):
                x_move = x_old.copy()
                x_move[i] += precision
                dy = obj_func(x_move) - obj_func(x_old)
                x_new[i] = x_old[i] + learning_rt * dy / precision
            
            y_new = obj_func(x_new)
        
            for func_no in range(len(csn_func_list)):
                csn_func[func_no] = (csn_func_list[func_no])(x_new)
        
            # 判断新的一组自变量是否符合约束条件，并有所优化
            if (lb < x_new).all() & (x_new < ub).all() & (func_lb_list < csn_func).all() & (
                    csn_func < func_ub_list).all() & (y_new > y_old):                    
                x_old = x_new.copy()
                y_old = obj_func(x_new)
            
            else: 
                break     
        
        # 在每组初始值的优化结果中，取最好的那一组
        if y_old > obj_func(x_opt):
            x_opt = x_old.copy()
        
    print("当前最优解：", obj_func(x_opt))
    
    return x_opt


# -------------------------------------
#         get_grad_default_para
# -------------------------------------
# 返回默认参数

def get_grad_default_para():
    
    default_para = {'learning_rt': 0.5,
                    'repeat_ct': 100, 
                    'precision': 0.0001, 
                    'ini_values': "rnd", 
                    'ini_ct': 1}
    
    return default_para

    
# -------------------------------------
#           get_grad_x_values
# -------------------------------------
# 给定(x1,x2,...,xn)的下界、上界、取值方法，返回(x1,x2,...,xn)的一组值

def get_grad_x_values(lb, ub, get_x_method, ini_ct):
    
    x_old_list = []
    x_old = lb.copy() + np.nan
    
    if get_x_method == "avg":
        # 取上下界的均值
        for ct in range(ini_ct):
            x_old = (lb + ub) / 2
            x_old_list.append(x_old)
    
    elif get_x_method == "rnd":
        # 在取值范围内随机取多组数
        for ct in range(ini_ct):
            for i in range(len(lb)):
                x_old[i] = np.random.uniform(low = lb[i], high = ub[i]) 
            x_old_list.append(x_old)
            
    elif get_x_method == "crnd":
        # 取中心对称的点
        x_mid = x_old.copy()
        x_vec = x_old.copy()
        x_mid = (lb + ub) / 2
        
        if ini_ct % 2 == 1:
            x_old_list.append(x_mid)       
            ini_ct -= 1
            
        r = min(ub - lb) / 2  
        
        while ini_ct > 0:                
            for i in range(len(lb)):
                x_vec[i] = np.random.uniform(low = - 1, high = 1)
                
            x_vec = x_vec / np.linalg.norm(x_vec) * r
            x_old_list.extend([x_mid + x_vec, x_mid - x_vec])
            ini_ct -= 2
            
    elif get_x_method == "urnd":
        # 取尽量均匀的的点
        
        # 计算每个维度能分成几段  
        n = int(ini_ct ** (1 / len(x_old))) + 1
        k = int(np.log(ini_ct) // np.log(n))
        x_ct = np.ones(len(x_old))
        x_ct[: k] = n
        x_ct = x_ct.tolist()
        
        # 尽量为更多的维度分段
        product = reduce(lambda x, y: x * y, x_ct)
        while ini_ct // product > 1:
            x_ct[x_ct.index(1)] = ini_ct // product
            product = reduce(lambda x, y: x * y, x_ct)
        
        x_rem = int(ini_ct - product)
        
        # 按分好的段均匀生成初始值
        for i in range(int(product)):
            product = 1
            x_old = lb.copy()
            for j in range(len(lb)):
                x_old[j] = lb[j] + (ub[j] - lb[j]) / (x_ct[j] + 1) * (1 + (i // product) % x_ct[j])
                product = product * x_ct[j]
            x_old_list.append(x_old)
        
        # 为多余的点分配初始值
        if x_rem % 2 == 1:
            x_old = lb.copy()
            for i in range(len(lb)):
                # 去除随机性
                # x_old[i] = np.random.uniform(low = lb[i], high = ub[i]) 
                x_old[i] = (ub[i] - lb[i]) / len(lb) * i + lb[i]
            x_old_list.append(x_old)      
            x_rem -= 1
                
        x_mid = lb.copy()
        x_mid = (lb + ub) / 2
        x_vec = lb.copy()   
        r = min(ub - lb) / 2  
        
        while x_rem > 0:                
            for i in range(len(lb)):
                #x_vec[i] = np.random.uniform(low = - 1, high = 1) 
                # 去除随机性
                x_vec[i] = ub[i] - (ub[i] - lb[i]) / len(lb) * i
            x_vec = x_vec / np.linalg.norm(x_vec) * r
            x_old_list.extend([x_mid + x_vec, x_mid - x_vec])
            x_rem -= 2
            
    else:
        raise Exception("无效取值方法：" + get_x_method)
    return x_old_list