import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

def open_return(df):
    '''获取开盘价数据'''
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
