'''
自定义适应度函数:my_metric\\
使用IC法检测因子:\\
IC = corr(X_t, y_{t+1})
'''

import pandas as pd
from gplearn.fitness import make_fitness
from scipy.stats import spearmanr
import numpy as np


def _my_metric(y, y_pred, w=[]):
    '''
    返回X,y之间的秩相关系数\\
    y已自动.shift(-1)
    '''
    if not isinstance(y, pd.Series) or not isinstance(y_pred, pd.Series):
        # raise TypeError("Both inputs should be pandas Series objects.")
        y, y_pred = pd.Series(y), pd.Series(y_pred)

    if len(y) != len(y_pred):
        raise ValueError('metirc error:X and y must have the same shapr')
    
    df = pd.DataFrame({'y':y,'y_pred':y_pred})
    df = df.dropna()
    
    corr, p = spearmanr(df['y'].iloc[:-1].values, df['y_pred'].shift(-1).iloc[:-1].values) # compare the value of date i with that of i+1 
    
    return abs(corr)

my_metric = make_fitness(function=_my_metric, greater_is_better=True)

# num1 = pd.Series([1, 5, 3, np.nan, 1, 3])
# num2 = pd.Series([2, np.nan, 1, 9, 9, 1])

# print(my_metric(num1, num2))
# nums = pd.Series([1, 3, 4, 3,5])
# print(nums.pct_change())