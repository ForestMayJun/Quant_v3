'''
    分钟级别数据的价量量化工具函数
'''
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime, time, timedelta
import sys
import os
import matplotlib.pyplot as plt

sys.path.append('/mnt/datadisk2/aglv/aglv/lab_aglv')
# from forintern.DataDaily import DataDaily
# from forintern.DataMinute_vwap import DataMinute

tqdm.pandas()

def open_return(df):

    df_reset = df.reset_index()
    df_multi = df_reset.set_index(['InstrumentID', 'EndTime'])
    type = 'Open'
    df_unstack = df_multi[type].unstack().T
    df_return = df_unstack.pct_change().fillna(0)
    df_signal = np.where(df_return > 0, 1, np.where(df_return < 0, -1, 0))
    df_signal = pd.DataFrame(df_signal, index=df_return.index, columns=df_return.columns).shift(1).fillna(0)
    df_cash = df_signal * df_unstack
    cash_cumsum = df_cash.sum(axis=1).cumsum()

    return cash_cumsum[-1]


def get_all_minute(type=None):
    '''获取某一年所有的分钟级别数据, 储存为一个三重multi-index df对象, 
    或者获取一个特定类型价格的二重multi-index df对象'''
    year = '2021'
    folder_root = '/mnt/datadisk/pjluo_shared/minute_h5_data/'
    folder = Path(folder_root)
    f_names = [f.name for f in folder.iterdir() if f.is_file() and f.name.endswith('.h5')]
    f_names = [f for f in f_names if f.startswith(year)]
    f_names.sort(key=lambda x: int(x[:8]))
    
    df_dic = {}
    for f in tqdm(f_names):
        df = pd.read_hdf(folder_root + f)
        df_reset = df.reset_index()
        df_multi = df_reset.set_index(['InstrumentID', 'EndTime'])
        if type == None:
            df_dic[f[:8]] = df_multi
        else:
            df_dic[f[:8]] = df_multi[type].unstack().T

    all_minute = pd.concat(df_dic.values(), keys=df_dic.keys())

    if type == None:
        all_minute.index.names = ['Date','InstrumentID', 'EndTime']
    else:
        all_minute.index.names = ['Date', 'EndTime']
        
    return all_minute


def time_list_t0(init='09:30', end='15:00', hour_gap=0, min_gap=10):
    '''
    生成一段均匀间隔的时点,Parameters:
    init:初始时间
    end:结束时间
    hour_gap, min_gap:间隔时间的小时分钟数
    '''
    init_time = datetime.strptime(init, '%H:%M')
    end_time = datetime.strptime(end, '%H:%M')

    noon_begin = datetime.strptime('11:30', '%H:%M')
    noon_end = datetime.strptime('13:00', '%H:%M')

    time_delta = timedelta(hours=hour_gap, minutes=min_gap)

    n_time_number = (end_time - init_time)/time_delta
    time_list = []
    for i in range(int(n_time_number)+1):
        t = init_time + i*time_delta
        if t <= noon_begin or t > noon_end:
            time_list.append(t.strftime('%H:%M'))

    return time_list

def t0_corr(data_minute):
    
    tl = time_list_t0(init='09:40', min_gap=15)

def vwap_corr_stock(df, min_gap=10, type='Close'):
    # tl = time_list_t0(min_gap=min_gap)
    # n_tl = len(tl)
    df = df.sort_values('EndTime')
    n_time = int(len(df)/min_gap)
    v_wap = []
    for i in range(n_time):
        v_wap.append(
            np.dot(df[type][min_gap*i:min_gap*(i+1)], df['LastVolume'][min_gap*i:min_gap*(i+1)]) / np.sum(df['LastVolume'][min_gap*i:min_gap*(i+1)]) if np.sum(df['LastVolume'][min_gap*i:min_gap*(i+1)]) !=0 else np.nan
        )
    v_wap = pd.Series(v_wap)
    v_wap = (v_wap - v_wap.mean()) / (v_wap.std() + 1e-10)

    return v_wap.corr(v_wap.shift(-1))

def vwap_corr_df(df):
    s_list = pd.Series(df.index.unique(), index=df.index.unique())
    return s_list.progress_apply(lambda x:vwap_corr_stock(df.loc[x]))

def q_stats_powersum(df:pd.DataFrame,h=10):
    '''
    df:为multi-index对象, 两级索引['Date', 'EndTime']  
    h:求和的窗口数，  
    返回:$\sum_{k=1}^h q_k^2$
    '''

    def _q_stats_powersum(data:pd.DataFrame, h=h):
        '''时序自相关性Q统计量,针对每天生成因子'''
        n = len(data)
        data = data.pct_change()
        data = (data - data.mean()) / (data.std() +1e-10).fillna(0)
        lagged_data = np.array([data.shift(-k).fillna(0).values for k in range(1, h + 1)])
        rho_squared_sum = np.sum((np.einsum('ijk,jk->ik', lagged_data, data.values))**2 / (n - np.arange(1, h + 1))[:, None], axis=0)
        
        return pd.Series(rho_squared_sum, index=data.columns)
        
    return df.groupby(level='Date').progress_apply(_q_stats_powersum)

def q_stats_std(df:pd.DataFrame,h=10,step=1):
    '''
    df:为multi-index对象, 两级索引['Date', 'EndTime']  
    h:求和的窗口数,默认=h  
    step:滞后时期的回望步长,默认=1  
    返回:$\std_{k=1}^h q_k^2$
    '''

    def _q_stats_std(data:pd.DataFrame, h=h):
        '''时序自相关性Q统计量,针对每天生成因子'''
        n = len(data)
        data = (data - data.mean()) / (data.std() +1e-10)
        lagged_data = np.array([data.shift(-k).fillna(0).values for k in range(1, step*h + 1, step)])
        q_stats = np.einsum('ijk,jk->ik', lagged_data, data.values)
        rho_squared_sum = np.std(q_stats, axis=0)
        
        return pd.Series(rho_squared_sum, index=data.columns)
        
    return df.groupby(level='Date').progress_apply(_q_stats_std)

def skew_daily(df:pd.DataFrame):
    '''日内数据的skew'''
    def _skew_daily(data:pd.DataFrame):
        data = data.pct_change().fillna(0)
        data = (data - data.mean()) / (data.std() + 1e-10).fillna(0)
        return data.skew()
    
    return df.groupby(level='Date').progress_apply(_skew_daily)


def high_std_mean(df:pd.DataFrame, std_window=10, mean_window=5, std_p=0.8):
    '''
    日内高波收益率因子，
    df:价格df, 
    std_window: 向前滚动计算std的窗口分钟数, 
    mean_window: 向前滚动计算mean的窗口分钟数, 
    std_p: 计算std序列高波标准的分位数
    '''
    def _high_std_mean(data:pd.DataFrame, s_w=std_window, m_w=mean_window, std_p=std_p):
        def _for_series(s:pd.Series):
            s_std = s.rolling(s_w).std().fillna(0)
            p_value = s_std.quantile(std_p)
            mean_sum = s[s_std > p_value].rolling(m_w).mean().sum()
            return mean_sum
        
        return data.apply(_for_series)
    
    return df.groupby(level='Date').progress_apply(_high_std_mean)


def long_short_return(df:pd.DataFrame, mean_window=5, sum_window=15):
    '''
    日内单侧行情, 衡量多空博弈力量, 判断长涨长跌
    '''
    def _long_short_return(data:pd.DataFrame):
        data = data.pct_change().fillna(0)
        data_mean = data.rolling(mean_window).mean().fillna(0)
        data_mean_sum = data_mean.rolling(sum_window).sum().fillna(0)
        return data_mean_sum.max() - data_mean_sum.min()
    
    return df.groupby(level='Date').progress_apply(_long_short_return)

def long_short_return_v2(df:pd.DataFrame, window=5):
    '''
    日内单侧行情, 衡量多空博弈力量, 判断急涨急跌
    '''
    def _long_short_return(data:pd.DataFrame):
        data = data.pct_change().fillna(0)
        data_mean_sum = data.rolling(window).sum().fillna(0)
        return data_mean_sum.max() - data_mean_sum.min()
    
    return df.groupby(level='Date').progress_apply(_long_short_return)

def long_short_return_v3(df:pd.DataFrame, window=15):
    '''
    日内单侧行情, 衡量多空博弈力量, 判断急涨急跌
    df:行列均是Multiindex的df对象,列包含价格序列和成交量序列
    '''
    def _long_short_return(data:pd.DataFrame):
        price = data.xs(key='Close', level='Type', axis=1)
        ret = price.pct_change().fillna(0)
        # data_mean_sum = ret.rolling(window).sum().fillna(0)

        vol = data.xs(key='LastVolume', level='Type', axis=1)
        vol_percent = vol / vol.sum()

        vol_ret = ret * vol_percent
        vol_ret_sum = vol_ret.rolling(window).sum()

        return vol_ret_sum.max() - vol_ret_sum.min()
    
    return df.groupby(level='Date').progress_apply(_long_short_return)

def conti_up_down_v1(price:pd.DataFrame, T=30, is_up=True):
    '''
    返回日内连续性涨跌幅,使用矩阵乘法,
    price:Multi-index对象,
    T:最大回溯窗口期,
    is_up:True时计算涨幅, False时计算跌幅,
    '''

    A = np.triu(np.ones((T, T)))
    def _conti_up_down(price_daily, T=T, is_up=is_up):
        return_daily = price_daily.pct_change()
        def _for_stock(r_datly_stock):
            if is_up:
                res = r_datly_stock.rolling(T).apply(lambda x: np.max(np.dot(A, x.fillna(0))))
            else:
                res = r_datly_stock.rolling(T).apply(lambda x: np.min(np.dot(A, x.fillna(0))))
            return res
        
        return return_daily.apply(_for_stock)
    
    return price.groupby(level='Date').progress_apply(_conti_up_down)

def conti_up_down_v2(price: pd.DataFrame, T=30, is_up=True):
    '''
    返回日内连续性涨跌幅,使用卷积计算,
    price:Multi-index对象,
    T:最大回溯窗口期,
    is_up:True时计算涨幅, False时计算跌幅
    '''

    kernel_arrays = [np.ones(i) for i in range(1, T+1)]

    def _conti_up_down(price_daily, T=T, is_up=is_up):
        return_daily = price_daily.pct_change().fillna(0)

        def _for_stock(r_daily_stock):
            r_mat = np.vstack([
                np.pad(np.convolve(r_daily_stock, kernel_arrays[i], mode='valid'), (i, 0), mode='constant') for i in range(len(kernel_arrays))
                ])
            
            if is_up:
                res = np.max(r_mat)
            else:
                res = np.min(r_mat)
            
            return res
        
        return return_daily.apply(_for_stock)
    
    return price.groupby(level='Date').progress_apply(_conti_up_down)

def inday_std_max(price:pd.DataFrame, window=15):
    '''
    日内给定窗口期的波动最大值, 窗口默认值为15分钟
    '''
    def _inday_std_max(daily_price:pd.DataFrame, w=window):
        daily_ret = daily_price.pct_change()
        
        return daily_ret.rolling(w).std().max()
    
    return price.groupby(level='Date').progress_apply(_inday_std_max)

def is_stop_trade(price:pd.DataFrame, std_level=1e-6, back_period=-15, updown_level=0.08):
    '''
    判断某只股票是否涨停跌停, 1为可以正常交易, 0为涨停跌停 
    price:分钟级价格数据, multi-index对象 
    std_level:认为停止交易的数据标准差阈值 
    back_period:回溯时间期, 单位min
    updown_level:认为涨跌停幅度阈值
    '''

    def _is_stop_trade(price_daily:pd.DataFrame):
        return pd.Series(np.where(
            (price_daily.iloc[back_period:].std() < std_level) & (np.abs(price_daily.iloc[-1] / price_daily.iloc[0]) - 1 > updown_level),
            0, 1), index=price_daily.columns)
    
    return pd.DataFrame(price.groupby(level='Date').progress_apply(_is_stop_trade), columns=price.columns)

def get_stop_stock(price:pd.DataFrame, std_level=1e-6, back_period=-15, updown_level=0.08):
    '''
    返回在某个日期涨停跌停的股票
    price:分钟级价格数据, multi-index对象 
    std_level:认为停止交易的数据标准差阈值 
    back_period:回溯时间期, 单位min
    updown_level:认为涨跌停幅度阈值
    '''
    
    def _get_stop_stock(price_daily:pd.DataFrame):
        jg = (price_daily.iloc[back_period:].std() < std_level) & (np.abs(price_daily.iloc[-1] / price_daily.iloc[0]) - 1 > updown_level)
        return jg[jg == True].index.to_list()
    
    return price.groupby(level='Date').progress_apply(_get_stop_stock)

def show_idmax_plot(date_id, factor, price):
    '''
    查看因子值最大的股票的当日以及下一日的情况
    f_idmax:因子数据factor.idxmax()
    '''
    f_idmax = factor.idxmax(axis=1)
    stock= f_idmax.iloc[date_id]
    data = price[stock][date_id*240-10:(date_id+2)*240+10].values
    plt.plot(data)
    plt.title(stock+'-'+f_idmax.index[date_id])
    plt.show()


if __name__ == '__main__':
    x = pd.DataFrame()
    x['a'] = pd.Series(np.random.rand(100))
    x['b'] = pd.Series(np.sin(np.arange(0, 1, 0.01)))
    print(x)
    print(x.apply(q_stats_powersum))
