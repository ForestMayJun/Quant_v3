
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import multiprocessing
import sys

sys.path.append('/mnt/datadisk2/aglv/aglv/lab_aglv/forintern/')
from DataDaily import DataDaily

datadaily = DataDaily()

def get_price_vol(datadaily):
    '''
    获取下面几个函数的数据对象  
    -> MultiIndex对象 level1-InstrumentID level2-date
    '''
    close = datadaily.adjclose.loc[20210101:].T.stack()
    vol = datadaily.volume.loc[20210101:].T.stack()
    data = pd.concat([close, vol], axis=1)
    data.columns = ['close', 'vol']
    
    return data

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def _handle_stockly(data_stock: pd.DataFrame, r_win=15):
    def _f(x: pd.Series):
        vol = data_stock['vol'].loc[x.index]
        vol = (vol - vol.mean()) / (vol.std() + 1e-10)
        ret = x.pct_change().fillna(0)
        x = (x - x.mean()) / (x.std() + 1e-10)
        
        return np.dot(x * ret, vol)

    return data_stock['close'].rolling(r_win).apply(_f)

def process_group(group):
    return (group[0], _handle_stockly(group[1]))

def price_vol_weight_corr_v1(data: pd.DataFrame, r_win=15, n_processes=4):
    '''
    捕捉量增价涨行情, 计算rolling一段时间的相关系数
    data: MultiIndex对象, l1:InstrumentID, l2:data, 日频数据
    '''
    grouped = list(data.groupby(level='InstrumentID'))[:4]

    with multiprocessing.Pool(processes=n_processes) as pool:
        results = list(tqdm(pool.imap(process_group, grouped), total=len(grouped)))

    factor_list = []
    for instrument_id, factor in results:
        factor_df = pd.DataFrame(factor.values, index=factor.index.get_level_values('date'), columns=[factor.index[0][0]])
        factor_list.append(factor_df)

    factor = pd.concat(factor_list, axis=1)

    return factor.T

data = get_price_vol(datadaily)
price_vol_weight_corr_v1(data)