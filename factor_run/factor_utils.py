'''因子计算工具'''
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from forintern.DataDaily import DataDaily

def rolling(data, f, window=30):
    '''
    pd.series -> pd.series,
    每个元素是rolling前window窗口 -> 
    '''
    res = pd.Series(index=data.index) # 储存数据 iloc[i]为i-window - i时间段的f的值，f:pd.series -> float
    for i in range(window, len(data)):
        res.iloc[i] = f(data.iloc[i-window:i])
    
    return res

def log_return(df:pd.DataFrame):
    '''接受价格df对象,返回对数收益率'''
    df = df.fillna(method='ffill')
    ratio = df.pct_change() + 1
    log_ratio = np.log(ratio).fillna(0)

    return log_ratio

def over_to_normal(df:pd.DataFrame):
    '''假设series服从标准正态分布,将大于2的值归于0'''
    df[df > 2] = 0
    df[df < -2] = 0
    return df

if __name__ == '__main__':
    df = pd.DataFrame([[2, np.nan, 6, 3, np.nan],
                       [4, 7, 0, 1, 2]])
    # print(log_return(s1))
    print(over_to_normal(df))