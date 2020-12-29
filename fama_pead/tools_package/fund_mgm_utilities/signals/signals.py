from numba import jit
import pandas as pd
import numpy as np
from numba import guvectorize
import fund_mgm_utilities as fmu




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                       Common backtesting signals
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ------
# Cross
# ------
def cross(ts1, ts2):
    return ref(ts1 < ts2, 1, fill_missing = False) & (ts1 >= ts2)


# ---------
#  barlast
# ---------
    
# Return an array that is the same size bool_mtx
# Each element represents the number of rows since the last true value
def barlast(bool_mtx):
    barlast_np = barlast_numb(bool_mtx)
    return barlast_np.astype(int)
    
@jit(nopython = True)
def barlast_numb(bool_mtx):
    
    time_ct = bool_mtx.shape[0]
    asset_ct = bool_mtx.shape[1]
    
    barlast_np = np.zeros((time_ct,asset_ct))
    
    for i in range (1, time_ct):
        for j in range (asset_ct):
            if bool_mtx[i-1,j] == 0:
                barlast_np[i,j] = barlast_np[i-1,j] + 1
                
    return barlast_np


# --------
#   ref
# --------
# Data and idx_shift are arrays of the same size
# For each element in data, find an element that is n rows above it, where n is the corresponding element in idx_shift

def ref(data, idx_shift, fill_missing = np.nan):
    if type(idx_shift) == np.ndarray:
        return ref_dynamic(data, idx_shift, fill_missing)
    else:
        return ref_static(data, idx_shift, fill_missing)


@jit(nopython = True, nogil = True, fastmath = True)
def ref_dynamic(data, idx_shift, fill_missing = np.nan):
    
    time_ct = data.shape[0]
    asset_ct = data.shape[1]
    
    out_mtx = np.full_like(data, fill_missing)

    for j in range(asset_ct):
        for i in range(time_ct):
            row_no = i - idx_shift[i, j]        
            if row_no >= 0:
                out_mtx[i,j] = data[row_no, j]
    
    return out_mtx

def ref_static(data, idx_shift, fill_missing = np.nan):

    out_mtx = np.full_like(data, fill_missing)
    
    if idx_shift > 0:
        out_mtx[idx_shift:, :] = data[:-idx_shift, :]
    elif idx_shift < 0:
        out_mtx[:idx_shift, :] = data[-idx_shift:, :]
    else:
        out_mtx = data.copy()
        
    return out_mtx


# =============================================================================
# @guvectorize(['void(double[:,:], intp[:], double[:], double[:,:])'], '(n,m),(),()->(n,m)',nopython=True)
# def ref_static2(data, idx_shift, fill_missing, result):
#     wid = idx_shift[0]
#     fill_missing = fill_missing[0]
#     result[:wid-1, :] = fill_missing
#     result[wid:, :] = a[:-wid, :]
#     
# =============================================================================
    


# --------
#  count
# --------
    
# Might need a static case fast solution using pd.rolling
def count(bool_mtx_np, win_mtx_np):  
    bool_mtx_np[np.isnan(bool_mtx_np)] = 0
    if type(win_mtx_np) == np.ndarray:
        return dynamic_window_sum(bool_mtx_np, win_mtx_np)
    else:
        return pd.DataFrame(bool_mtx_np).rolling(window = win_mtx_np).sum().values

@jit(nopython = True)
def dynamic_window_sum(value4sum, win_mtx_np):

    out_mtx = np.full_like(value4sum *1.0, np.nan)

    for j in range(out_mtx.shape[1]):
        prv_sum_lost = True
        for i in range(win_mtx_np.shape[0]):
            if win_mtx_np[i, j] > 0 and i - win_mtx_np[i, j] >= -1:   
                if prv_sum_lost:       

                    out_mtx[i, j] = np.sum(value4sum[i - win_mtx_np[i, j]+1 : i+1, j])
                    prv_sum = out_mtx[i, j]
                    
                else:
                    # Fast method
                    crn_sum = prv_sum + value4sum[i, j]
                    
                    last_wid_start = i-win_mtx_np[i-1, j]
                    wid_adj_len = win_mtx_np[i, j] - (win_mtx_np[i - 1, j] + 1) # If zero then win len increased by 1, no adjustment necessary
                    
                    if wid_adj_len > 0:
                        crn_sum = crn_sum + np.sum(value4sum[last_wid_start - wid_adj_len : last_wid_start, j])
                    elif wid_adj_len < 0:
                        crn_sum = crn_sum + np.sum(value4sum[last_wid_start : last_wid_start - wid_adj_len, j])
                    
                    out_mtx[i, j] = crn_sum
                    prv_sum = crn_sum
                prv_sum_lost = False
            else:
                if win_mtx_np[i, j] == 0:                                     # This is to conform with inherited special definition
                    out_mtx[i, j] = np.sum(value4sum[0 : i+1, j])  
                prv_sum_lost = True
                        
    return out_mtx
    

# -----------
#   min_roll
# -----------
@jit(nopython = True)
def min_roll(input_np, min_win):
    out = np.empty_like(input_np) + np.nan
    for j in range(input_np.shape[1]):
        out[:, j] = fmu.mv_min(input_np[:, j], min_win)
    return out


# -----------
#   max_roll
# -----------
def max_roll(input_np, max_win):
    return pd.DataFrame(input_np).rolling(window = max_win).max().values


# -----------
#   all_roll
# -----------
def all_roll(input_np, all_wid):
    return pd.DataFrame(input_np).rolling(window = all_wid, min_periods = 1).sum().values == all_wid

# -----------
#   ma
# -----------
def ma(input_np, ma_wid):
    return pd.DataFrame(input_np).rolling(window = ma_wid).mean().values

# -----------
#   std_roll
# -----------
def std_roll(input_np, std_wid, cap_value = None):
    std = pd.DataFrame(input_np).rolling(window = std_wid).std().values
    if not(cap_value is None):
        std[std < cap_value] = cap_value
    return std


# -----------
#   max_pair
# -----------
def max_pair(input_np1, input_np2):
    return np.maximum(input_np1, input_np2)

# --------------
#   count_since
# --------------
# Since x have occured, how many times have y occured
def count_since(x, y):
    return count(x, barlast(y))

# --------------
#  occur number
# --------------
# Since x have occured, if y occured today, how many times is this y
def occur_number(x, y):
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            a = 1
    return 1


# -------------
#  rowwise_pct
# -------------
# Find the rowwise percentile of data_np
def rowwise_pct(data_np):
    
    rank_loc = np.argsort(data_np, 1) 
    
    rank = rank_loc2rank(rank_loc)
                
    valid_data_ct = np.sum(~np.isnan(data_np), 1)
    
    pct = rank / np.expand_dims(valid_data_ct, 1)
    pct[np.isnan(data_np)] = np.nan
    
    return pct

# -------------
#  rank_loc2rank
# -------------
# rank_loc stores the location of value of each rank
# This function converts it to the rank of each location
    
def rank_loc2rank(rank_loc):
    rank = rank_loc.copy()
    ordered_array = np.arange(rank_loc.shape[1])
    for row_no in range(rank_loc.shape[0]):
        rank[row_no, rank_loc[row_no, :]] = ordered_array
    return rank

# -------------
#  reduce_freq
# -------------
# Reduce the frequency which data_np changes
@jit(nopython = True)
def reduce_freq(data_np, adj_freq, start_cycle_no = 1):
    
    data_out_np = data_np.copy()
    cycle_no = start_cycle_no
    for row_no in range(1, data_out_np.shape[0]):
        if cycle_no > 0:
            data_out_np[row_no,:] = data_out_np[row_no - 1,:]

        cycle_no += 1
        if cycle_no > adj_freq:
            cycle_no = 0
    
    return data_out_np


