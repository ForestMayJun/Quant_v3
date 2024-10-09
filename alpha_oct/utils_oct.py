
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

def split_ret_inday(p_inday:pd.DataFrame, time_split):
    '''
    Paras:
    p_inday: MultiIndex对象, 分钟级别价格
    time_split: 日内收益分隔的时间点 
    '''

    left_p = p_inday[:, time_split]
    right_p = p_inday[time_split:]

    def p_to_ret(p, shift_min=0):
        afno_ret = p.groupby(level='Date').apply(lambda x: x.iloc[:, -1] / x.iloc[:, shift_min] -1)
        afno_ret.index.names = ['a', 'Date', 'InsrumentID']
        afno_ret.index = afno_ret.index.droplevel('a')
        afno_ret = afno_ret.unstack()
        
        return afno_ret
    
    left_ret = p_to_ret(left_p, shift_min=1) # 有些股票第一分钟没数据
    right_ret = p_to_ret(right_p)


def inverse(price:pd.DataFrame, l_min=30):
    l_ret = price.groupby('Date', axis=0).apply(lambda x: x.iloc[l_min, :] / x.iloc[0, :] - 1)
    r_ret = price.groupby('Date', axis=0).apply(lambda x: x.iloc[-1, :] / x.iloc[l_min, :] -1)

    sym = l_ret * r_ret
    sym = sym / abs(sym)

    num = sym.shape[1] - sym.isna().sum(axis=1)
    inv_per = sym.sum(axis=1) / num
    
    return inv_per, sym, num
    
        