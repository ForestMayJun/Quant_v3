'''
运算函数的辅助函数\\  
my_try_except\\
test_Xd\\  
test_Xa  
'''

import numpy as np
import pandas as pd

def my_try_except(*args):
    '''
    封装块 try-except
    Parameters:
    f: function;
    X:array; 
    d:int or float; 
    (Y:array) 
    '''
    
    try:
        f, X, d = args[0], args[1], args[2]
        if len(args) == 3:
            return f(X, d)
        else:
            Y = args[3]
            return f(X, Y, d)
    except TypeError as e:
        if not 'only size-1 arrays can be converted to Python scalars'  in str(e):
            print(f"TypeError in {f.__name__} function: {e}")
        return np.zeros_like(X)
    except Exception as e:
        print(f"Error in {f.__name__} function: {e}")
        return np.zeros_like(X)
    
def test_Xd(X, d):
    '''
    封装代码 
    检测以及标准化X,d
    Parameters:
    X:array
    d:int
    '''
    try:
        X = pd.Series(X)
        d = abs(int(d[0]))  # transform d to int
        if not isinstance(d, int) or d < 0:
            raise TypeError("d must be a non-negative integer.")
    except Exception as e:
        if not 'only size-1 arrays can be converted to Python scalars'  in str(e):
            raise ValueError(f'{e}')

    
    return X, d

def test_Xa(X, a):
    '''
    封装代码
    检测以及标准化X,a
    Parameters:
    X:array
    a:float
    '''
    try:
        X = pd.Series(X)
        a = abs(float(a[0]))
        if not isinstance(a, float):
            raise TypeError("d must be a float.")
    except Exception as e:
        if not 'only size-1 arrays can be converted to Python scalars'  in str(e):
            raise ValueError(f'{e}')

    return X, a