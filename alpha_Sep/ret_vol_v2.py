
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
tqdm.pandas()
import matplotlib.pyplot as plt
import multiprocessing
from forintern.DataDaily import DataDaily
datadaily = DataDaily()
close = datadaily.adjclose
vol = datadaily.volume

def get_price_vol(datadaily):
    '''
    获取下面几个函数的数据对象  
    -> MultiIndex对象 level1-InstrumentID level2-date
    '''
    close = datadaily.adjclose.loc[20210101:20231231].T.stack()
    open = datadaily.adjopen.loc[20210101:20231231].T.stack()
    vol = datadaily.volume.loc[20210101:20231231].T.stack()
    data = pd.concat([close,open, vol], axis=1)
    data.columns = ['close', 'open', 'vol']
    
    return data

def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
def relu(x):
    return pd.Series(np.where(x > 0, 1, 0), index=x.index)

def true_relu(x):
    return pd.Series(np.where(x > 0, x, 0), index=x.index)

def neg_relu(x):
    return pd.Series(np.where(x < 0, 1, 0), index=x.index)

def posi_or_neg(x):
    return pd.Series(np.where(x > 0, 1, -1), index=x.index)

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def leaky_relu(x, alpha=0.01):
    return pd.Series(np.where(x > alpha *x , x, alpha * x), index=x.index)

def p_dot(x, y, p=2):
    return np.power(np.dot(np.power(x, p), np.power(y, p)), 1/p)

def _handle_stockly(data_stock:pd.DataFrame, r_win=15):
    data_stock['vol_zs'] = (data_stock['vol'] - data_stock['vol'].rolling(r_win).sum()) / (data_stock['vol'].rolling(r_win).std() + 1e-10)
    data_stock['inday_ret'] = data_stock['close'] / data_stock['open']
    data_stock['night_ret'] = data_stock['open'] / data_stock['close'].shift(1)
    data_stock['weight'] = true_relu((data_stock['inday_ret'] - 1) * 100).rolling(r_win).mean()
    # data_stock['vol_zs_weight'] = relu(data_stock['vol_zs']).rolling(r_win).mean()

    def _f(x):
        y = data_stock['vol_zs'].loc[x.index]
        return np.dot(x, y)
    res = data_stock['weight'].rolling(r_win).apply(_f)

    del data_stock

    return res


def process_group(group, r_win=15):
    instrument_id, group_data = group
    return (instrument_id, _handle_stockly(group_data, r_win))

def price_vol_weight_corr_v1(data: pd.DataFrame, n_processes=64):
    grouped = list(data.groupby(level='InstrumentID'))
    with multiprocessing.Pool(processes=n_processes) as pool:
        results = list(tqdm(pool.imap(process_group, grouped), total=len(grouped)))

    factor_list = []
    for _, factor in results:
        factor_df = pd.DataFrame(factor.values, index=factor.index.get_level_values('date'), columns=[factor.index[0][0]])
        factor_list.append(factor_df)

    factor = pd.concat(factor_list, axis=1)

    return factor

data = get_price_vol(datadaily)
data