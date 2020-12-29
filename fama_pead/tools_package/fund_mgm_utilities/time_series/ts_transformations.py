import pandas as pd
import numpy as np
import fund_mgm_utilities as fmu
from numba import jit


# ----------------
# 将收益增加或减少
# ----------------
def shift_return(current_return, time_hist, shift_amount):
    time_diff = time_hist.diff().dt.days.values / 365.25
    time_diff[np.isnan(time_diff)] = 0
    shifted_return = current_return + shift_amount * time_diff

    return shifted_return


# ----------------
# Rolling Maximum
# ----------------

# Input variables: vec: a one dimensional array. Must be one dimension or numba will fail
#                  window: look back period
#                  min_periods: minimum number of numbers

# The basic idea is to check the current max aganist previous max
# If not then check if first value in previous window equal previous max
# If yes then it is possible that the max has exited the window and we need to calculate the maximum using the slow way

# Output variables: a one dimensional vector the same as as the input vector
    
@jit(nopython=True)
def mv_max(vec_in, window, min_periods = 1):
    
    vec = vec_in.copy()
    
    vec[np.isnan(vec)] = -np.inf
    
    row_ct = len(vec)
    roll_max_np = np.zeros_like(vec) + np.nan

    m1 = np.max(vec[:min_periods])
    roll_max_np[min_periods - 1] = m1
    
    # Before there is enough data for one window
    for i in range(min_periods, window):
        y2 = vec[i]
        if y2 > m1: 
            m1 = y2
        
        roll_max_np[i] = m1
    
    # Begin Rolling
    for i in range(window, row_ct):
        ip1 = i + 1
        y1 = vec[i-window]
        x2 = vec[i]

        if x2 >= m1:
            roll_max_np[i] = x2
        elif x2 < m1:
            if y1 != m1:
                roll_max_np[i] = m1
            else:  
                roll_max_np[i] = np.max(vec[ip1 - window: ip1])
        m1 = roll_max_np[i]
        
    roll_max_np[roll_max_np == -np.inf] = np.nan

    return roll_max_np


# ----------------
# Rolling Minimum
# ----------------

# Input variables: vec: a one dimensional array. Must be one dimensions or numba will fail
#                  window: look back period
#                  min_periods: minimum number of numbers

# Output variables: a one dimensional vector the same as as the input vector
    

@jit(nopython=True)
def mv_min(vec_in, window, min_periods = 1):
    
    vec = vec_in.copy()
    
    row_ct = len(vec)
    roll_min_np = np.zeros_like(vec) + np.nan

    vec[np.isnan(vec)] = np.inf


    m1 = np.min(vec[:min_periods])
    roll_min_np[min_periods - 1] = m1
    
    # Before there is enough data for one window
    for i in range(min_periods, window):
        y2 = vec[i]
        if y2 < m1: m1 = y2
        
        roll_min_np[i] = m1
    
    # Begin Rolling
    for i in range(window, row_ct):
        ip1 = i + 1
        y1 = vec[i-window]
        x2 = vec[i]
        if x2 <= m1: roll_min_np[i] = x2
        elif x2 > m1:
            if y1 != m1: roll_min_np[i] = m1
            else:  roll_min_np[i] = np.min(vec[ip1 - window: ip1])
        m1 = roll_min_np[i]
        
    roll_min_np[roll_min_np == np.inf] = np.nan

    return roll_min_np



# ----------------
# Rolling Average
# ----------------

# Input variables: vec: a one dimensional array. Must be one dimensions or numba will fail
#                  window: look back period
#                  min_periods: minimum number of numbers

# The basic idea is to calculate the cummulative sum, and use the difference to calculate 
# the moving sum. The use division to get the moving average

# Output variables: a one dimensional vector the same as as the input vector
    
@jit(nopython = True)
def mv_avg(vec, window, min_periods = 1):
    vec_insert = insert_zeros_front(vec, 1)
    
    roll_avg = np.zeros_like(vec) + np.nan  
    
    # Calculate the cummulative sum
    cumsum_vec = np.cumsum(vec_insert) 
    
    # Beginning of the sequence
    avg_value = cumsum_vec[min_periods: window + 1] /  np.arange(min_periods, window + 1)
    roll_avg[min_periods-1:window] = avg_value
    
    # End of the sequence
    main_seq = (cumsum_vec[window:] - cumsum_vec[:-window]) / float(window)
    roll_avg[window-1:] = main_seq
    
    return roll_avg


# ----------------
# Rolling Sum
# ----------------

# Input variables: vec: a one dimensional array. Must be one dimensions or numba will fail
#                  window: look back period
#                  min_periods: minimum number of numbers

# The basic idea is to calculate the cummulative sum, and use the difference to calculate 
# the moving sum.

# Output variables: a one dimensional vector the same as as the input vector
@jit(nopython = True)
def mv_sum(vec, window, min_periods = 1):
    vec_insert = insert_zeros_front(vec, 1)
    
    roll_sum = np.zeros_like(vec) + np.nan  
    
    # Calculate the cummulative sum
    cumsum_vec = np.cumsum(vec_insert) 
    
    # Beginning of the sequence
    sum_value = cumsum_vec[min_periods: window + 1] 
    roll_sum[min_periods-1:window] = sum_value
    
    # End of the sequence
    main_seq = cumsum_vec[window:] - cumsum_vec[:-window]
    roll_sum[window-1:] = main_seq
    
    return roll_sum



# ----------------------
#   Add zeros in front
# ----------------------
# Add count many zeros to the front of the one dimensional vector vec
@jit(nopython = True)
def insert_zeros_front(vec, count):    
    row_ct = vec.shape[0]
    vec_insert = np.zeros(row_ct + count)
    vec_insert[count:] = vec
    return vec_insert



# ------------------
#  find_last_known
# ------------------
# Similarly to the result of excel funtion match(idx_target, idx_source, 1)
    
@jit(nopython = True)
def find_last_known(idx_target, idx_source):

    source_ct = idx_source.shape[0]
    crn_search_start_row = 0
    out_idx = np.zeros_like(idx_target) - 1

    for target_row, target_value in enumerate(idx_target):
        
        for source_row in range(crn_search_start_row, source_ct):
            
            if idx_source[source_row] > target_value:
                if source_row > 0:
                    crn_search_start_row = source_row
                    out_idx[target_row] = source_row - 1
                break
            # If we reached the end of the source, and found that the current target is at least as big as the source
            # Since the target can only get bigger, we know the answer to other targets
            if source_row == source_ct - 1:
                idx_source[source_row] <= target_value
                out_idx[target_row:] = source_row
                break
            
    return out_idx




# ---------------
#   mv_argmax
# ---------------
# How many days ago did that last max value occur. 
# Only look for the last window data points
# !!! This will need to be fixed
@jit(nopython=True)
def mv_argmax(vec, window, min_periods = 1):
    
    row_ct = len(vec)
    roll_max_np = np.zeros_like(vec) + np.nan
    
    m1 = np.max(vec[:min_periods])
    roll_max_np[min_periods - 1] = m1
    
    roll_argmax = np.zeros_like(vec) + np.nan
    roll_argmax[min_periods - 1] = 0
    
    # Before there is enough data for one window
    for i in range(min_periods, window):
        y2 = vec[i]
        if y2 > m1: 
            m1 = y2
            roll_argmax[i] = 0
            roll_max_np[i] = m1
        else:
            roll_argmax[i] = roll_argmax[i - 1] + 1
            
    # Begin Rolling
    for i in range(window, row_ct):
        ip1 = i + 1
        y1 = vec[i-window]
        x2 = vec[i]
        if x2 >= m1: 
            roll_argmax[i] = 0
            roll_max_np[i] = x2
        elif x2 < m1:
            if y1 != m1: 
                roll_argmax[i] = roll_argmax[i - 1] + 1
                roll_max_np[i] = m1
            else:  
                roll_argmax[i] = window - np.argmax(vec[ip1 - window: ip1]) - 1
                roll_max_np[i] = np.max(vec[ip1 - window: ip1])
                
        m1 = roll_max_np[i]
        
    return roll_argmax



# ---------------
#   mv_argmin
# ---------------
# How many days ago did that last min value occur. 
# Only look for the last window data points
    
@jit(nopython=True)
def mv_argmin(vec, window, min_periods = 1):
    
    row_ct = len(vec)
    roll_min_np = np.zeros_like(vec) + np.nan
    
    m1 = np.min(vec[:min_periods])
    roll_min_np[min_periods - 1] = m1
    
    roll_argmin = np.zeros_like(vec) + np.nan
    roll_argmin[min_periods - 1] = 0
    
    # Before there is enough data for one window
    for i in range(min_periods, window):
        y2 = vec[i]
        if y2 < m1: 
            m1 = y2
            roll_argmin[i] = 0
            roll_min_np[i] = m1
        else:
            roll_argmin[i] = roll_argmin[i - 1] + 1
            
    # Begin Rolling
    for i in range(window, row_ct):
        ip1 = i + 1
        y1 = vec[i-window]
        x2 = vec[i]
        if x2 <= m1: 
            roll_argmin[i] = 0
            roll_min_np[i] = x2
        elif x2 > m1:
            if y1 != m1: 
                roll_argmin[i] = roll_argmin[i - 1] + 1
                roll_min_np[i] = m1
            else:  
                roll_argmin[i] = window - np.argmin(vec[ip1 - window: ip1]) - 1
                roll_min_np[i] = np.min(vec[ip1 - window: ip1])
                
        m1 = roll_min_np[i]
        
    return roll_argmin



    


