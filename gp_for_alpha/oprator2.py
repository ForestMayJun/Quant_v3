'''
自定义运算符\\
函数列表：\\
[rank, delay, corr_d, cov_d, scale, delta, signed_power, decay_linear,
ts_min, ts_max, ts_argmin, ts_argmax, ts_rank, ts_sum, ts_product, ts_std]\\
并激活函数\\
初始函数列表为:init_func\\
自定义函数为:user_func\\
合并函数列表为:my_func
'''
import numpy as np
import pandas as pd
from gplearn.functions import make_function
# from assist_func import my_try_except
# from assist_func import test_Xd
# from assist_func import test_Xa




def _rank(X):
    '''返回X向量中每个元素在X中的分位数'''
    X = pd.Series(X)
    ranks = X.rank(method='min')
    percent = (ranks - 1) / (len(X) -1 )
    return percent

def _delay(X, d):
    '''返回向量X前d天的值'''
    try:
        d = abs(int(d[0]))+1 if isinstance(d,(list, np.ndarray)) else abs(int(d))+1  # transform d to int
        if not isinstance(d, int) or d < 0:
            raise TypeError("d must be a non-negative integer.")
        
        X = pd.Series(X)
        return X.shift(d)
    
    except TypeError as e:
        if not 'only size-1 arrays can be converted to Python scalars'  in str(e):
            print(f"TypeError in delay function: {e}")
        return np.zeros_like(X)
    except Exception as e:
        print(f"Error in delay function: {e}")
        return np.zeros_like(X)


def _corr_d(X, Y, d):
    '''计算X,Y最近d天构成的时序列的相关系数'''
    try:
        d = abs(int(d[0]))+1 if isinstance(d,(list, np.array)) else abs(int(d))+1  # transform d to int
        if not isinstance(d, int) or d < 0:
            raise TypeError("d must be a non-negative integer.")
    
        X, Y = pd.Series(X), pd.Series(Y)
        if len(X) != len(Y):
            raise ValueError('X and Y have length error')
        
        return X.rolling(d).corr(Y.rolling(d))
    except TypeError as e:
        if not 'only size-1 arrays can be converted to Python scalars'  in str(e):
            print(f"TypeError in corr_d function: {e}")
        return np.zeros_like(X)
    except Exception as e:
        print(f"Error in corr_d function: {e}")
        return np.zeros_like(X)

def _cov_d(X, Y, d):
    '''计算X,Y最近d天构成的时序列的协方差'''
    try:
        d = abs(int(d[0]))+1 if isinstance(d,(list, np.array)) else abs(int(d))+1  # transform d to int
        if not isinstance(d, int) or d < 0:
            raise TypeError("d must be a non-negative integer.")
    
        X, Y = pd.Series(X), pd.Series(Y)
        if len(X) != len(Y):
            raise ValueError('X and Y have length error')
        
        return X.rolling(d).cov(Y.rolling(d))
    except TypeError as e:
        if not 'only size-1 arrays can be converted to Python scalars'  in str(e):
            print(f"TypeError in cov_d function: {e}")
        return np.zeros_like(X)
    except Exception as e:
        print(f"Error in cov_d function: {e}")
        return np.zeros_like(X)

def _scale(X, a):
    '''返回 a*X / sum(abs(X)), a的缺省值为1,a一般大于0'''
    try:
        a = float(a)
        X = pd.Series(X)
        if not isinstance(a, float):
            raise TypeError("d must be a float.")
        
        return a*X / np.sum(np.abs(X))
    except TypeError as e:
        if not 'only size-1 arrays can be converted to Python scalars'  in str(e):
            print(f"TypeError in scale function: {e}")
        return np.zeros_like(X)
    except Exception as e:
        print(f"Error in scale function: {e}")
        return np.zeros_like(X)

def _delta(X, d):
    '''返回 X.diff(d)'''
    try:
        d = abs(int(d[0]))+1 if isinstance(d,(list, np.array)) else abs(int(d))+1  # transform d to int
        if not isinstance(d, int) or d < 0:
            raise TypeError("d must be a non-negative integer.")
        
        X = pd.Series(X)
        return X.diff(d)
    except TypeError as e:
        if not 'only size-1 arrays can be converted to Python scalars'  in str(e):
            print(f"TypeError in delta function: {e}")
        return np.zeros_like(X)
    except Exception as e:
        print(f"Error in delta function: {e}")
        return np.zeros_like(X)

def _signed_power(X, a):
    '''保留符号乘方'''
    try:
        a = float(a)
        X = pd.Series(X)
        if not isinstance(a, float):
            raise TypeError("d must be a float.")
        
        return np.sign(X) * np.power((np.abs(X)), a)
    except TypeError as e:
        if not 'only size-1 arrays can be converted to Python scalars'  in str(e):
            print(f"TypeError in signed_power function: {e}")
        return np.zeros_like(X)
    except Exception as e:
        print(f"Error in signed_power function: {e}")
        return np.zeros_like(X)

def _decay_linear(X, d):
    '''d天线性加权值,权重依靠距今日期做衰减 d, d-1, .... 加权和使用当天数据'''
    try:
        d = abs(int(d[0]))+1 if isinstance(d,(list, np.array)) else abs(int(d))+1  # transform d to int
        if not isinstance(d, int) or d < 0:
            raise TypeError("d must be a non-negative integer.")
        
        X = pd.Series(X)
        def dot_d(X, d):
            X_d = X.iloc[-d:]
            weight = np.arange(1, d+1, 1) / np.sum(np.arange(1, d+1, 1))
            return np.dot(X_d, weight)
        
        res = pd.Series(index=X.index)
        for i in range(d-1, len(X)):
            res.iloc[i] = dot_d(X.iloc[i-d+1:i+1], d)
    
        return res
    
    except TypeError as e:
        if not 'only size-1 arrays can be converted to Python scalars'  in str(e):
            print(f"TypeError in decay_linear function: {e}")
        return np.zeros_like(X)
    except Exception as e:
        print(f"Error in decay_linear function: {e}")
        return np.zeros_like(X)


def _ind_neutralize(X, inclass):
    '''因子行业中性化'''
    pass

def _ts_min(X, d):
    ''' -> array 过去d天最小时序值, 包含当天值'''
    try:
        d = abs(int(d[0]))+1 if isinstance(d,(list, np.array)) else abs(int(d))+1  # transform d to int
        if not isinstance(d, int) or d < 0:
            raise TypeError("d must be a non-negative integer.")
        
        X = pd.Series(X)
        return X.rolling(d).min()
    except TypeError as e:
        if not 'only size-1 arrays can be converted to Python scalars'  in str(e):
            print(f"TypeError in ts_min function: {e}")
        return np.zeros_like(X)
    except Exception as e:
        print(f"Error in ts_min function: {e}")
        return np.zeros_like(X)

def _ts_max(X, d):
    ''' -> array 过去d天最大时序值, 包含当天值'''
    try:
        d = abs(int(d[0]))+1 if isinstance(d,(list, np.array)) else abs(int(d))+1  # transform d to int
        if not isinstance(d, int) or d < 0:
            raise TypeError("d must be a non-negative integer.")
        
        X = pd.Series(X)
        return X.rolling(d).max()
    except TypeError as e:
        if not 'only size-1 arrays can be converted to Python scalars'  in str(e):
            print(f"TypeError in ts_max function: {e}")
        return np.zeros_like(X)
    except Exception as e:
        print(f"Error in ts_max function: {e}")
        return np.zeros_like(X)

def _ts_argmin(X, d):
    ''' -> array 过去d天最小时序值索引, 包含当天值'''
    try:
        d = abs(int(d[0]))+1 if isinstance(d,(list, np.array)) else abs(int(d))+1  # transform d to int
        if not isinstance(d, int) or d < 0:
            raise TypeError("d must be a non-negative integer.")
        
        res = pd.Series(index=X.index)
        for i in range(1, len(X)):
            if i < d:
                res.iloc[i] = X[:i+1].idxmax()
            else:
                res.iloc[i] = X[i-d+1:i+1].idxmax()

        return res
    
    except TypeError as e:
        if not 'only size-1 arrays can be converted to Python scalars'  in str(e):
            print(f"TypeError in ts_argmax function: {e}")
        return np.zeros_like(X)
    except Exception as e:
        print(f"Error in ts_argmax function: {e}")
        return np.zeros_like(X)

def _ts_argmax(X, d):
    ''' -> array 过去d天最小时序值索引, 包含当天值'''
    try:
        d = abs(int(d[0]))+1 if isinstance(d,(list, np.array)) else abs(int(d))+1  # transform d to int
        if not isinstance(d, int) or d < 0:
            raise TypeError("d must be a non-negative integer.")
        
        res = pd.Series(index=X.index)
        for i in range(1, len(X)):
            if i < d:
                res.iloc[i] = X[:i+1].idxmin()
            else:
                res.iloc[i] = X[i-d+1:i+1].idxmin()
                
        return res
    
    except TypeError as e:
        if not 'only size-1 arrays can be converted to Python scalars'  in str(e):
            print(f"TypeError in ts_argmin function: {e}")
        return np.zeros_like(X)
    except Exception as e:
        print(f"Error in ts_argmin function: {e}")
        return np.zeros_like(X)

def _ts_rank(X, d):
    '''索引i -> 在i - i-d时间段这个数组中, i的分位值'''
    try:
        d = abs(int(d[0]))+1 if isinstance(d,(list, np.array)) else abs(int(d))+1  # transform d to int
        if not isinstance(d, int) or d < 0:
            raise TypeError("d must be a non-negative integer.")
        
        res = pd.Series(index=X.index)
        for i in range(d-1, len(X)):
            ranks = X[i-d+1:i+1].rank(method='min')
            res.iloc[i] = (ranks.iloc[-1] - 1) / (d - 1)

        return res
    
    except TypeError as e:
        if not 'only size-1 arrays can be converted to Python scalars'  in str(e):
            print(f"TypeError in ts_rank function: {e}")
        return np.zeros_like(X)
    except Exception as e:
        print(f"Error in ts_rank function: {e}")
        return np.zeros_like(X)

def _ts_sum(X, d):
    '''索引i -> i位置的过去d天的时序和'''
    try:
        d = abs(int(d[0]))+1 if isinstance(d,(list, np.array)) else abs(int(d))+1  # transform d to int
        if not isinstance(d, int) or d < 0:
            raise TypeError("d must be a non-negative integer.")

        return X.rolling(d).sum()
    
    except TypeError as e:
        if not 'only size-1 arrays can be converted to Python scalars'  in str(e):
            print(f"TypeError in ts_sum function: {e}")
        return np.zeros_like(X)
    except Exception as e:
        print(f"Error in ts_sum function: {e}")
        return np.zeros_like(X)


def _ts_product(X, d):
    '''前i天的连乘'''
    try:
        d = abs(int(d[0]))+1 if isinstance(d,(list, np.array)) else abs(int(d))+1
        X = pd.Series(X)
        return X.rolling(d).prod()
    
    except TypeError as e:
        if not 'only size-1 arrays can be converted to Python scalars'  in str(e):
            print(f"TypeError in ts_prod function: {e}")
        return np.zeros_like(X)
    except Exception as e:
        print(f"Error in ts_prod function: {e}")
        return np.zeros_like(X)

def _ts_std(X, d):
    '''前i天的标准差'''
    try:
        d = abs(int(d[0]))+1 if isinstance(d,(list, np.array)) else abs(int(d))+1
        X = pd.Series(X)
        return X.rolling(d).std()
    
    except TypeError as e:
        if not 'only size-1 arrays can be converted to Python scalars'  in str(e):
            print(f"TypeError in ts_std function: {e}")
        return np.zeros_like(X)
    except Exception as e:
        print(f"Error in ts_std function: {e}")
        return np.zeros_like(X)


rank = make_function(function=_rank, name='rank', arity=1, wrap=False)
delay = make_function(function=_delay, name='delay', arity=2, wrap=False)
corr_d = make_function(function=_corr_d, name='corr_d', arity=3, wrap=False)
cov_d = make_function(function=_cov_d, name='cov_d', arity=3, wrap=False)
scale = make_function(function=_scale, name='scale', arity=2, wrap=False)
delta = make_function(function=_delta, name='delta', arity=2, wrap=False)
signed_power = make_function(function=_signed_power, name='signed_power', arity=2, wrap=False)
# ind_neutralize = make_function(function=_ind_neutralize, name='ind_neutralize', arity=2, wrap=False)
decay_linear = make_function(function=_decay_linear, name='decay_linear', arity=2, wrap=False)
ts_min = make_function(function=_ts_min, name='ts_min', arity=2, wrap=False)
ts_max = make_function(function=_ts_max, name='ts_max', arity=2, wrap=False)
ts_argmin = make_function(function=_ts_argmin, name='ts_argmin', arity=2, wrap=False)
ts_argmax = make_function(function=_ts_argmax, name='ts_argmax', arity=2, wrap=False)
ts_rank = make_function(function=_ts_rank, name='ts_rank', arity=2, wrap=False)
ts_sum = make_function(function=_ts_sum, name='ts_sum', arity=2, wrap=False)
ts_product = make_function(function=_ts_product, name='ts_product', arity=2,wrap=False)
ts_std = make_function(function=_ts_std, name='ts_std', arity=2, wrap=False)

user_func = [rank, delay, corr_d, cov_d, scale, delta, signed_power, decay_linear,
             ts_min, ts_max, ts_argmin, ts_argmax, ts_rank, ts_sum, ts_product, ts_std]

init_func = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'inv', 'sin', 'max', 'min']

my_func = init_func + user_func