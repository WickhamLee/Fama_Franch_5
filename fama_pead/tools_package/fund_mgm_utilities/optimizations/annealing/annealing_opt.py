import numpy as np

# =======================================================================================
#                                       模拟退火算法
# =======================================================================================

# 功能：简化的模拟退火算法，用于寻找目标函数一定范围内的极大值点
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

def annealing_opt (obj_func, lb, ub, csn_func_list, func_lb_list, func_ub_list, para):
    
    # 合并输入的参数与默认参数
    para = {** get_annealing_default_para(), ** para} 
    
    T0 = para['T0']                           # 初始温度
    Tmin = para['Tmin']                       # 终止温度
    repeat_ct = para['repeat_ct']             # 循环次数
    precision = para['precision']             # 精度
    ini_values = para['ini_values']           # 初始值
  
    t = 0
    T = T0
    
    # 初始化自变量
    x_old = get_annealing_x_values(lb, ub, ini_values)    

    # 初始化因变量、约束函数
    y_old = obj_func(x_old)
    csn_func = func_lb_list.copy() 
    
    
    # 模拟退火算法
    while T >= Tmin:     
        
        for i in range(repeat_ct):      
            x_new = x_old + np.random.uniform(low = -1, high = 1, size = len(x_old)) * T * precision
            y_new = obj_func(x_new)
            
            for func_no in range(len(csn_func_list)):
                csn_func[func_no] = (csn_func_list[func_no])(x_new)
            
            # 判断新的一组自变量是否符合约束条件，并有所优化
            if (lb < x_new).all() & (x_new < ub).all() & (func_lb_list < csn_func).all() & (
                                             csn_func < func_ub_list).all() & (y_new > y_old):                    
                x_old = x_new
                y_old = obj_func(x_new)
                                     
        t += 1
        T = T0/(1+t)
        
        
    print("当前最优解：", obj_func(x_old))
    
    return x_old


# -------------------------------------
#      get_annealing_default_para
# -------------------------------------
# 返回默认参数

def get_annealing_default_para():
    
    default_para = {'T0': 100, 
                    'Tmin': 10,
                    'repeat_ct' : 10,
                    'precision': 1/1000, 
                    'ini_values': "avg"}
    
    return default_para
    

# -------------------------------------
#        get_annealing_x_values
# -------------------------------------
# 给定(x1,x2,...,xn)的下界、上界、取值方法，返回(x1,x2,...,xn)的一组值

def get_annealing_x_values(lb, ub, get_x_method):
    
    x_old = lb.copy() + np.nan
    
    if get_x_method == "avg":
        # 取上下界的均值
        x_old = (lb + ub) / 2
    
    elif get_x_method == "rnd":
        # 在取值范围内随机取一组数
        for i in range(len(lb)):
            x_old[i] = np.random.uniform(low = lb[i], high = ub[i]) 
            
    else:
        raise Exception("无效取值方法：" + get_x_method)
    
    return x_old