
# 用来放置一些乱七八糟但是又不知道该放哪里的小工具
# 主要用来解决python中一些烦人的小问题

import math
import datetime

# -------
# isnan
# -------

# 利用pandas.read_csv(open(file_name)) 去读的表格中，如果内容是空的，则会被用nan填充。
# 要判断一个cell是否是nan, 需要先检查type是不是数字，否则直接检查会报错
# 这里将所需要做的判断汇集到一个函数上

def isnan(cell_content):

    comparable_type_list = ['float', 'float64']

    return_value = False
    if type(cell_content).__name__ in comparable_type_list:
        if math.isnan(cell_content):
            return_value = True

    return return_value

# -------
#  now
# -------
# 有时候想打印个时间太麻烦了，需要一长串代码
def now():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')