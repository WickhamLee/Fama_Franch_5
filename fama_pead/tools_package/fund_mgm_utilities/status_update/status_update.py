import fund_mgm_utilities as fmu

# 用来集中处理运行耗时较长的函数时，输出中间状态的问题

# 初始化用户设定。用户设定的了，按用户的来。没有的，按默认的来。

# style_name: 一些提前预设好的输出风格的名称， 例如, print_all 表示输出所有状态
def initialize_print_setting(user_setting = dict(), override_function_name = True, style_name = ''):

    if style_name == '':
        print_setting_default = {"print_mid_status": False,
                                "indent": 0,
                                "current_level": 0,
                                "print_func_name": True,
                                'print_begin_status': False,
                                'print_end_status': False,
                                'show_time': True,
                                'begin_end_line_divider': True}                # 在输出开始和结束状态时增加一行分割线。在反复运行多个同等级函数时，这个设置有助于区分不同的函数运行状态

    elif style_name == 'print_all':

        print_setting_default = {"print_mid_status": True,
                                "indent": 0,
                                "current_level": 0,
                                "print_func_name": True,
                                'print_begin_status': True,
                                'print_end_status': True,
                                'show_time': True,
                                'begin_end_line_divider': True}


    user_setting = {**print_setting_default, **user_setting}               # 合并这两个dictionary
    if override_function_name:
        user_setting['func_name'] = fmu.get_func_name(2)                   # 默认呼叫函数名为目前呼叫 initialize_print_setting 这个函数的函数

    # 是否空几格，以及是否输出函数名称
    indent_string = ''
    if user_setting['print_begin_status'] or user_setting['print_mid_status'] or user_setting['print_end_status']:

        for i in range(user_setting['indent']):
            indent_string = indent_string + ' '

        if user_setting['print_func_name']:
            indent_string = indent_string + user_setting['func_name'] + ': '

    user_setting['indent_string'] = indent_string

    return user_setting


# ---------------------------------------------------------------
#                           输出状态
# ---------------------------------------------------------------

def print_status(print_setting, which_part, end_string = ''):

    # 当执行的任务涉及到多层函数呼叫的时候，层数达到一定深度时也许我们就不想输出状态了。因为多层的呼叫可能涉及的都是一些不重要的细节
    # 这里处理用户的层数设定。
    # print_setting['current_level' 表示当前已经到第几层了
    # print_setting['max_level']: 表示状态输出的最深层数

    if not print_setting == []:
        print_print_depth_exceeded = False
        if 'max_print_level' in print_setting.keys() and 'current_level' in print_setting.keys():
            if print_setting['current_level'] > print_setting['max_print_level']:
                print_print_depth_exceeded = True

        if not print_print_depth_exceeded:
            begin_str = print_setting['indent_string']
            if print_setting['show_time']:
                begin_str = begin_str + fmu.now()

            if which_part == 'mid' and print_setting['print_mid_status']:
                if type(end_string) is str:
                    print(begin_str + ":" + end_string)
                else:
                    print(begin_str)
                    print(end_string)

            elif which_part == 'begin' and print_setting['print_begin_status']:
                if print_setting['begin_end_line_divider'] == True:
                    print(' ')

                if type(end_string) is str:
                    begin_status_str = begin_str + ': 函数开始运行' + end_string
                    print(begin_status_str)
                else:
                    begin_status_str = begin_str + ': 函数开始运行'
                    print(begin_status_str)
                    print(end_string)


            elif which_part == 'end' and print_setting['print_end_status']:
                if type(end_string) is str:
                    end_status_str = begin_str + ': 函数运行完毕' + end_string
                    print(end_status_str)
                else:
                    end_status_str = begin_str + ': 函数运行完毕'
                    print(end_status_str)
                    print(end_string)



# ---------------------------------------------------------------
#                   返回打印设置时的标准空格数量
# ---------------------------------------------------------------

# 这个数字决定了，如果A函数呼叫了B函数，且A和B都输出运行状态，则B在输出状态前
# 前面留的空格数量比A多多少个
def get_print_sd_indent():
    return 5


# ---------------------------------------------------------------
#            将PrintSetting里的indent数量提高一个级别
# ---------------------------------------------------------------
def incre_print_indent(print_setting, how_many_levels = 1):
    print_setting['indent'] = print_setting['indent'] + get_print_sd_indent() * how_many_levels
    print_setting['current_level'] = print_setting['current_level'] + how_many_levels
    return print_setting


