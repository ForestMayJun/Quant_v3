
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import multiprocessing
import matplotlib.pyplot as plt

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

def multiporcessing_generate_factor(data):
    '''
    传入multiindex对象, level1-InstrumentID, level2-日频data, columns为日频价量数据, 可以多列
    '''
    shift_day = 0
    r_win = 20
    n_processing = 64

    def _handle_stockly(data_stock: pd.DataFrame, r_win=r_win):
        def _f(x: pd.Series):
            pass

        return data_stock['close'].rolling(r_win).apply(_f)

    def process_group(group, r_win=15):
        instrument_id, group_data = group
        return (instrument_id, _handle_stockly(group_data, r_win))

    def _price_vol_weight_corr_v1(data: pd.DataFrame, n_processes=n_processing):
        grouped = list(data.groupby(level='InstrumentID'))
        with multiprocessing.Pool(processes=n_processes) as pool:
            results = list(tqdm(pool.imap(process_group, grouped), total=len(grouped)))

        factor_list = []
        for instrument_id, factor in results:
            factor_df = pd.DataFrame(factor.values, index=factor.index.get_level_values('date'), columns=[factor.index[0][0]])
            factor_list.append(factor_df)

        factor = pd.concat(factor_list, axis=1)

        return factor.T
    
    return _price_vol_weight_corr_v1(data)


def draw_price_vol_unique(data:pd.DataFrame):
    '''
    data:MultiIndex对象, 一列价格一列成交量数据
    '''
    p_unique = data['Close'].unique()
    vol_unique = pd.Series(index=p_unique)
    for p in vol_unique.index:
        vol_unique.loc[p] = data['LastVolume'][data['Close'] == p].sum()

    plt.scatter(p_unique, vol_unique)
        
def drop_level1_index(data:pd.DataFrame):
    res = data.copy()
    res.index.names = ['a', 'InstrumentID', 'Date']
    id = res.index.droplevel('a')
    res.index = id
    del data

    return res.unstack(level='InstrumentID')