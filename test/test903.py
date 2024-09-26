import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
tqdm.pandas()

sys.path.append('/mnt/datadisk2/aglv/aglv/lab_aglv/forintern')
# from forintern.DataDaily import DataDaily

def rsrs(price:pd.DataFrame, window=15):
    def _rsrs(price_stock):
        res = pd.Series(index=price_stock.index)
        for i in range(window, len(res)):
            # data = price_stock.iloc[i-window:i, :].fillna(1)
            # res.iloc[i] = np.polyfit(data.xs('adjlow', level='Type', axis=1).values.flatten(), data.xs('adjhigh', level='Type', axis=1).values.flatten(), 1)[0]
            res.iloc[i] = np.polyfit(price_stock.iloc[i-window:i, 0].fillna(1), price_stock.iloc[i-window:i, 1].fillna(1), 1)[0]
        
        return res
            
    return price.groupby('Stock', axis=1).progress_apply(_rsrs)

price_daily = pd.read_hdf('/mnt/datadisk2/aglv/aglv/lab_aglv/datadaily_aglv.h5')
high_low = price_daily[['adjlow', 'adjhigh']].stack().unstack(level='InstrumentID').unstack()
high_low.columns.names = ['Stock', 'Type']

rsrs1 = rsrs(high_low.loc[20240704:])