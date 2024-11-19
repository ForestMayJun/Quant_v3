import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import matplotlib.pyplot as plt
import multiprocessing
import statsmodels.api as sm

import multiprocessing
import statsmodels.api as sm


def ols_resid_2factor(fa1, fa2):
    '''将两个因子做时序中性化取残差'''
    shift_day = 0
    r_win = 20
    n_processing = 64

    u = set(fa1.columns).intersection(set(fa2.columns))
    fa1 = fa1.loc[:, u].unstack().stack(level='InstrumentID').stack(level='date')
    fa2 = fa2.loc[:, u].unstack().stack(level='InstrumentID').stack(level='date')

    concat_fa = pd.concat([fa1, fa2], axis=1)

    def _handle_stockly(data_stock: pd.DataFrame, r_win=r_win):
        '''
        进行中性化计算  
        data_stock:单只股票的日频数据, 两列columns, 第0列作为x第1列作为y进行回归=
        '''
        def f(x, y):
            if x.isna().sum() + y.isna().sum() == 0:
                x = sm.add_constant(x)
                model = sm.OLS(y, x)
                res = model.fit()
                return res.resid[-1]
            else:
                return np.nan

        return data_stock.iloc[:, 0].rolling(r_win).apply(lambda x:f(x, data_stock.iloc[:, 1].loc[x.index]))

    def process_group(group, r_win=15):
        instrument_id, group_data = group
        return (instrument_id, _handle_stockly(group_data, r_win))

    def multiprocessing_factor(data: pd.DataFrame, n_processes=n_processing):
        grouped = list(data.groupby(level='InstrumentID'))
        with multiprocessing.Pool(processes=n_processes) as pool:
            results = list(tqdm(
                pool.imap(process_group, grouped), total=len(grouped)
            ))

        factor_list = []
        for instrument_id, factor in results:
            factor_df = pd.DataFrame(
                factor.values,
                index=factor.index.get_level_values('date'), 
                columns=[factor.index[0][0]]
            )
            factor_list.append(factor_df)

        factor = pd.concat(factor_list, axis=1)

        return factor
    
    return multiprocessing_factor(concat_fa)


def vol_distribution_byprice(data):
    '''
    成交量按价格的分布
    data:分钟频数据
    '''
    n_processing = 64

    def handle_stockly(data_stock: pd.DataFrame):
        '''对单只股票的处理函数, 可以是日频的也可以是分钟频的数据, 返回一个日频的series'''

        def handle_stockly_daily(data_stock_daily:pd.Series):
            g_num = 10
            min_s, max_s = data_stock_daily['Close'].min(), data_stock_daily['Close'].max()

            if min_s == max_s:
                return np.nan
            
            p_range = np.linspace(min_s, max_s, g_num)
            data_stock_daily['cate'] = pd.cut(data_stock_daily['Close'], p_range)
            g_value_vol = data_stock_daily.groupby('cate').sum()['LastVolume']
            p_volmax_iv = g_value_vol.idxmax()
            p_volmax = (p_volmax_iv.left + p_volmax_iv.right) / 2

            return p_volmax

        return data_stock.groupby(level='Date').apply(handle_stockly_daily)


    def process_group(group):
        instrument_id, group_data = group
        return (instrument_id, handle_stockly(group_data))

    def multiprocessing_factor(data: pd.DataFrame, n_processes=n_processing):
        grouped = list(data.groupby(level='InstrumentID'))
        with multiprocessing.Pool(processes=n_processes) as pool:
            results = list(tqdm(
                pool.imap(process_group, grouped), total=len(grouped)
            ))

        factor_list = []
        date_name = 'date' if 'date' in data.index.names else 'Date'
        for instrument_id, factor in results:
            factor_df = pd.DataFrame(
                factor.values,
                index=factor.index.get_level_values(date_name), 
                columns=[instrument_id]
            )
            factor_list.append(factor_df)

        factor = pd.concat(factor_list, axis=1)

        return factor
    
    return multiprocessing_factor(data)

def p_volmax_handle_stockly(data_stock: pd.DataFrame):
    '''
    计算p_volmax
    对单只股票的处理函数, 可以是日频的也可以是分钟频的数据, 返回一个日频的series
    '''

    def handle_stockly_daily(data_stock_daily:pd.Series):
        g_num = 10
        min_s, max_s = data_stock_daily['Close'].min(), data_stock_daily['Close'].max()

        if min_s == max_s:
            return np.nan
        
        p_range = np.linspace(min_s, max_s, g_num)
        data_stock_daily['cate'] = pd.cut(data_stock_daily['Close'], p_range)
        g_value_vol = data_stock_daily.groupby('cate').sum()['LastVolume']
        p_volmax_iv = g_value_vol.idxmax()
        p_volmax = (p_volmax_iv.left + p_volmax_iv.right) / 2

        return p_volmax

    return data_stock.groupby(level='Date').apply(handle_stockly_daily)

