import datetime
import pandas as pd


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#       Deals with time related functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# -------
#  now
# -------
# 有时候想打印个时间太麻烦了，需要一长串代码
def now(message = '', convert_to_str = True, time_format = '%Y-%m-%d %H:%M:%S'):

    time_now =  datetime.datetime.now()
    if convert_to_str:
        time_now = message +time_now.strftime(time_format)
    return time_now

# -------
#  today
# -------
# 有时候想打印个今天的日期太麻烦了，需要一长串代码
def today(message = '', convert_to_str = True, str_format = '%Y-%m-%d'):

    date_today =  datetime.datetime.now()
    date_today = date_today.replace(hour = 0, minute = 0, second = 0, microsecond = 0)

    if convert_to_str:
        date_today = message + date_today.strftime(str_format)
    return date_today

# -------------------
#  find_next_weekday
# -------------------
    
# Find the next date that is the nths day of a week, where n = weekday
    
def find_next_weekday(current_date, weekday):
    
    if weekday <= 7:
        crn_weekday = current_date.weekday() + 1
        
        if weekday > crn_weekday:
            days_shift = weekday - crn_weekday
        else:
            days_shift = 7 - (crn_weekday - weekday)
            
        return  current_date + + pd.Timedelta(days=days_shift)
    
    else:
        
        raise Exception("Weekday must be less than or equal to 7. You entered " + str(weekday))
    
    
    
    