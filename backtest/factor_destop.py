''' 因子去除涨跌停股票'''

import pandas as pd
import numpy as np
from tqdm import tqdm

def destop(factor:pd.DataFrame):
    '''
    is_trading:涨跌停股票   
    factor在第i日的因子值作为第i+1日调仓的依据, 若第i+1日涨跌停, 则第i日因子值变成NaN  
    有显著缺点  
    '''
    
    is_trading = pd.read_csv('/mnt/datadisk2/aglv/aglv/lab_aglv/is_trading.csv')
    if 'date' in is_trading.columns:
        is_trading.set_index('date', inplace=True)
    is_trading.index = [str(i) for i in is_trading.index]

    if 'Date' in factor.columns:
        factor.set_index('Date', inplace=True)
    factor.index = [str(i) for i in factor.index]

    res = factor.shift(1) * is_trading.loc[factor.index, factor.columns]
    res.index.names = ['Date']
    return res

if __name__ == '__main__':
    input_root = ''
    output_root = input_root[:-4] + 'destop.csv'
    factor_destop = destop(input_root)
    factor_destop.to_csv(output_root)

    