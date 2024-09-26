## 优化函数  九月

import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import multiprocessing

import sys
sys.path.append('/mnt/datadisk2/aglv/aglv/lab_aglv/forintern')
from DataDaily import DataDaily


def rsrs(price_stock, window = 15):
    '''
    尝试在rolling过程中使用两列元素, 运行速度有点慢....
    '''
    main_index = price_stock.index[window:]
    sub_index = np.arange(window)
    mul_index = pd.MultiIndex.from_product(
        [main_index, sub_index],
        names=['main_index', 'sub_index']
    )

    res = pd.DataFrame(
        [price_stock.loc[:date].iloc[-window:].values[i] for date in main_index for i in sub_index],
        index=mul_index,
        columns=price_stock.columns
    )

    def utils(data):
        if (data == 1).sum().sum() == 2*window:
            return np.nan
        else:
            return np.polyfit(data.iloc[:, 0], data.iloc[:, 1], 1)[0]
        
    return res.groupby(level='main_index').apply(utils)


def rsrs_v2(price:pd.DataFrame, window=15):
    '''
    直接用for循环计算  速度也有点小慢
    '''
    def _rsrs(price_stock):
        res = pd.Series(index=price_stock.index[window:])
        for i in range(window, len(res)):
            # data = price_stock.iloc[i-window:i, :].fillna(1)
            # res.iloc[i] = np.polyfit(data.xs('adjlow', level='Type', axis=1).values.flatten(), data.xs('adjhigh', level='Type', axis=1).values.flatten(), 1)[0]
            res.iloc[i] = np.polyfit(price_stock.iloc[i-window:i, 0].fillna(1), price_stock.iloc[i-window:i, 1].fillna(1), 1)[0]
        
        return res
            
    return price.groupby('Stock', axis=1).progress_apply(_rsrs)


def rsrs_v3(data, window=15):
    '''
    data是一个每个元素为元组的Series对象
    结果无法运行, apply函数对元素非数值的对象使用
    '''
    data = np.array(data.values.tolist())
    data = np.nan_to_num(data, nan=1)
    if np.sum([data == 1]) == 2 * window:
        return np.nan
    else:
        data += np.random.rand(window, 2) * 1e-6
        return np.polyfit(data[:, 0], data[:, 1], 1)[0]
    
def destop(factor:pd.DataFrame):
    '''
    去掉一字板涨跌停因子
    '''
    is_trading = pd.read_csv('/mnt/datadisk2/aglv/aglv/lab_aglv/is_trading.csv')
    is_trading.index = [str(i) for i in is_trading.index]

    factor.index = [str(i) for i in factor.index]
    res = factor.shift(1) * is_trading.loc[factor.index, factor.columns]
    res.index.names = ['Date']
    return res


def turn_std(vol:pd.DataFrame):
    def _turn_std(vol_daily:pd.DataFrame):
        vol_daily_percent = vol_daily / vol_daily.sum()
        vol_std = vol_daily_percent.std()

        return vol_std
    
    return vol.groupby(level='Date').progress_apply(_turn_std)

def turn_std_v2(vol:pd.DataFrame, k=0.9):
    '''turn cvar'''
    def _turn_std(vol_daily:pd.DataFrame):
        vol_daily_percent = vol_daily / vol_daily.sum()
        return vol_daily_percent[vol_daily_percent > vol_daily_percent.quantile(k)].mean()
    
    return vol.groupby(level='Date').progress_apply(_turn_std)

def turn_std_v3(vol:pd.DataFrame):
    '''
    将分钟级别成交量在rolling5天前的总数据的占比作为分钟级别换手率, 
    目的为了平衡掉某日整体成交量非常高的情形
    '''

    vol_r5mean = vol.groupby('Date').apply(lambda x: x.sum()).rolling(5).mean()

    def _turn_std(vol_daily:pd.DataFrame):
        vol_daily_percent = vol_daily / vol_r5mean
        vol_std = vol_daily_percent.std()

        return vol_std
    
    return vol.groupby(level='Date').progress_apply(_turn_std)

def turn_std_v4(vol:pd.DataFrame, agg_num=10):
    '''
    聚合时间, 将换手率序列转化为agg min 级别的数据再计算std
    '''
    def _assist(vol_daily:pd.DataFrame):
        vol_daily /= vol_daily.sum()
        vol_daily.index = list(range(len(vol_daily)))
        vol_daily_agg = vol_daily.groupby(vol_daily.index // agg_num).sum()

        return vol_daily_agg.std()
    
    return vol.groupby(level='Date').progress_apply(_assist)

def turn_std_v5(vol:pd.DataFrame, agg_num=10):
    '''将判定换手波动的指标从标准层改为一阶差分'''
    
    def _assist(vol_daily:pd.DataFrame):
        vol_daily /= vol_daily.sum()
        vol_daily.index = list(range(len(vol_daily)))
        vol_daily_agg = vol_daily.groupby(vol_daily.index // agg_num).sum()

        return vol_daily_agg.diff().abs().sum()
    
    return vol.groupby(level='Date').progress_apply(_assist)

def turn_vol_v1(data:pd.DataFrame):
    '''
    捕捉日内价增量涨行情
    data:MultiIndex对象, 包含列Close和LastVolume, 分钟级数据
    '''
    def _turn_vol_v1(data_daily:pd.DataFrame):
        def handle_stockly(data_daily_stock:pd.DataFrame):
            '''单只股票单日的分钟级别数据'''
            ret = data_daily_stock['Close'].pct_change()
            turn = data_daily_stock['LastVolume'] / data_daily_stock['LastVolume'].sum()

            return ret.corr(turn)

        return data_daily.groupby(level='InstrumentID').apply(handle_stockly)

    return data.groupby(level='Date').progress_apply(_turn_vol_v1)

def turn_vol_v3(data:pd.DataFrame, agg_num=10):
    '''
    捕捉日内价增量涨行情:将日内数据聚合后求将相关系数
    data:MultiIndex对象, 包含列Close和LastVolume, 分钟级数据
    '''

    def agg_series(s:pd.Series, agg_num=agg_num):
        s.index = list(range(len(s)))
        s_agg = s.groupby(s.index // agg_num).sum()
        
        return s_agg

    def _turn_vol(data_daily:pd.DataFrame):
        def handle_stockly(data_daily_stock:pd.DataFrame):
            '''单只股票单日的分钟级别数据'''
            ret = data_daily_stock['Close'].pct_change()
            turn = data_daily_stock['LastVolume'] / data_daily_stock['LastVolume'].sum()

            ret = agg_series(ret)
            turn = agg_series(turn)

            return ret.corr(turn, method='spearman')

        return data_daily.groupby(level='InstrumentID').apply(handle_stockly)

    return data.groupby(level='Date').progress_apply(_turn_vol)

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


def price_vol_corr(data:pd.DataFrame, r_win=15):
    '''
    捕捉量增价涨行情, 计算rolling一段时间的相关系数
    data: MultiIndex对象, l1:InstrumentID, l2:data, 日频数据
    '''
    def _handle_stockly(data_stock:pd.DataFrame):
        return data_stock['close'].rolling(r_win).corr(data_stock['vol'])

    factor = data.groupby(level='InstrumentID').progress_apply(_handle_stockly)
    factor = factor.unstack(level='date')
    factor.index.names = ['InstrumentID', 's']
    factor = factor.droplevel('s')

    return factor.T
    
def ret_vol_idxmax(data:pd.DataFrame, r_win=15):
    '''
    捕捉量增价涨行情, 计算rolling一段时间的相关系数
    data: MultiIndex对象, l1:InstrumentID, l2:data, 日频数据
    '''
    def _handle_stockly(data_stock:pd.DataFrame):
        def _f(x):
            id = x.idxmax()
            int_id = data_stock.index.get_loc(id)
            return data_stock['vol'].iloc[int_id] / data_stock['vol'].iloc[int_id-3 : int_id].sum()

        return data_stock['close'].pct_change().rolling(r_win).apply(lambda x: _f(x))

    factor = data.groupby(level='InstrumentID').progress_apply(_handle_stockly)
    factor = factor.unstack(level='date')
    factor.index.names = ['InstrumentID', 's']
    factor = factor.droplevel('s')

    return factor.T

def close_vol_shift_corr(data:pd.DataFrame, corr_win=20, shift_win=5):
    '''
    计算close和vol向前shift一段时间后的corr的和
    '''
    data = data.sort_index()
    def corr_shift(i):
        def _handle_stockly(data_daily:pd.DataFrame):
            return data_daily['close'].rolling(corr_win).corr(data_daily['vol'].shift(i))

        return data.groupby(level='InstrumentID').progress_apply(_handle_stockly)
    
    for i in range(shift_win):
        if i == 0:
            factor = corr_shift(i)
        else:
            factor += corr_shift(i)

    factor = factor.unstack(level='date')
    factor.index.names = ['InstrumentID', 's']
    factor = factor.droplevel('s')

    return factor.T
    
def p_vol_weight_corr_v1(data):
    '''
    此函数封装后调用会报无法pickle错误, 使用时应该将此拆开使用

    实现带ret权的价量相关系数  
    使用multiprocessing.Pool 多进程池
    data: MultiIndex对象, l1:InstrumentID, l2:data, 日频数据
    '''
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def _handle_stockly(data_stock: pd.DataFrame, r_win=15):
        def _f(x: pd.Series):
            x = (x - x.mean()) / (x.std() + 1e-10)
            vol = data_stock['vol'].loc[x.index]
            vol = (vol - vol.mean()) / (vol.std() + 1e-10)
            ret = x.pct_change().fillna(0)
            
            return np.dot(x * ret, vol)

        return data_stock['close'].rolling(r_win).apply(_f)

    def process_group(group):
        instrument_id, group_data = group
        return (instrument_id, _handle_stockly(group_data))

    def _price_vol_weight_corr_v1(data: pd.DataFrame, r_win=15, n_processes=4):
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

        return factor
    
    return _price_vol_weight_corr_v1(data)


def p_vol_weight_corr_v2(data, shift_day=0, ret_threshold=0, r_win=15, n_processing=4):
    '''
    设置return的阈值来筛选上涨行情, 然后计算价量相关系数, 
    data: MultiIndex对象, l1:InstrumentID, l2:data, 日频数据
    shift_day: corr = price.corr(vol.shift(shift_day))
    ret_threshold: 控制时的ret的阈值, 大于阈值权重为1, 否则为0
    '''
    def _handle_stockly(data_stock: pd.DataFrame, r_win=r_win):
        def _f(x: pd.Series):
            vol = data_stock['vol'].loc[x.index].shift(shift_day)
            vol = (vol - vol.mean()) / (vol.std() + 1e-10)
            ret = x.pct_change().fillna(0)
            weight = np.where(ret > ret_threshold, 1, 0)
            x = (x - x.mean()) / (x.std() + 1e-10)

            return np.dot(x * vol, weight)

        return data_stock['close'].rolling(r_win).apply(_f)

    def process_group(group, r_win=15):
        instrument_id, group_data = group
        return (instrument_id, _handle_stockly(group_data, r_win))

    def _price_vol_weight_corr_v1(data: pd.DataFrame, n_processes=n_processing):
        grouped = list(data.groupby(level='InstrumentID'))
        with multiprocessing.Pool(processes=n_processes) as pool:
            results = list(tqdm(pool.imap(process_group, grouped), total=len(grouped)))

        factor_list = []
        for instrument_id, factor in results:
            factor_df = pd.DataFrame(factor.values, index=factor.index.get_level_values('date'), columns=[factor.index[0][0]])
            factor_list.append(factor_df)

        factor = pd.concat(factor_list, axis=1)

        return factor.T
    
    return _price_vol_weight_corr_v1(data)




def main():
    datadaily = DataDaily()
    data = get_price_vol(datadaily)
    p_vol_weight_corr_v1(data)

if __name__ == '__main__':
    main()