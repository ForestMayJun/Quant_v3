'''数据处理'''

import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DataDaily import DataDaily

# import the raw data
data_daily = DataDaily()

def adj_data(data_list, data=data_daily):
    '''drop the data not in the universe'''
    u = data.universe_all
    u = list(set(u).intersection(set(data_list.columns)))
    return data_list.loc[:, u]

close = adj_data(data_daily.close) # get the exact value of close prince

def price_to_yeild(p):
    '''from the price data to the yeild'''
    p_yeild = p.pct_change()
    p_yeild.iloc[0] = 0

    return p_yeild

# choose a specific stock to train
stock = 'SH600519'
s_close = close[stock]
nan_sum = s_close.isna().sum()
print(f'the stock has {nan_sum} lost value')

s_yeild = price_to_yeild(s_close)

if s_yeild.isna().sum() != 0:
    raise ValueError(f'there is nan value in stock {stock}')

X_train, X_test, y_train, y_test = train_test_split(
    s_close, s_yeild,
    train_size=0.8,
    shuffle=False, 
)

df = pd.DataFrame({'x':X_train, 'y':y_train})
df.to_csv('data.csv')

X_train = X_train.to_numpy().reshape(-1, 1)
y_train = y_train.values

# print(isinstance(y_train, pd.Series))

class my_data:
    def __init__(self, p_yeild) -> None:
        self.p_yeild = p_yeild

if __name__ == '__main__':
    print(X_train)
    print(y_train)