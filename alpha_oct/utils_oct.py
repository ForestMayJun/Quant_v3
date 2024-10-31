
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import multiprocessing
import matplotlib.pyplot as plt

def split_ret_inday(p_inday:pd.DataFrame, time_split):
    '''
    Paras:
    p_inday: MultiIndex对象, 分钟级别价格
    time_split: 日内收益分隔的时间点 
    '''

    left_p = p_inday[:, time_split]
    right_p = p_inday[time_split:]

    def p_to_ret(p, shift_min=0):
        afno_ret = p.groupby(level='Date').apply(lambda x: x.iloc[:, -1] / x.iloc[:, shift_min] -1)
        afno_ret.index.names = ['a', 'Date', 'InsrumentID']
        afno_ret.index = afno_ret.index.droplevel('a')
        afno_ret = afno_ret.unstack()
        
        return afno_ret
    
    left_ret = p_to_ret(left_p, shift_min=1) # 有些股票第一分钟没数据
    right_ret = p_to_ret(right_p)


def inverse(price:pd.DataFrame, l_min=30):
    l_ret = price.groupby('Date', axis=0).apply(lambda x: x.iloc[l_min, :] / x.iloc[0, :] - 1)
    r_ret = price.groupby('Date', axis=0).apply(lambda x: x.iloc[-1, :] / x.iloc[l_min, :] -1)

    sym = l_ret * r_ret
    sym = sym / abs(sym)

    num = sym.shape[1] - sym.isna().sum(axis=1)
    inv_per = sym.sum(axis=1) / num
    
    return inv_per, sym, num

def multiporcessing_generate_factor(data):
    '''
    传入multiindex对象, level1-InstrumentID, level2-日频data, columns为日频价量数据, 可以多列
    '''
    shift_day = 0
    r_win = 20
    n_processing = 64

    def _handle_stockly(data_stock: pd.DataFrame, r_win=r_win):
        def _f(x: pd.Series):
            pass

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


def draw_price_vol_unique(data:pd.DataFrame):
    '''
    data:MultiIndex对象, 一列价格一列成交量数据
    '''
    p_unique = data['Close'].unique()
    vol_unique = pd.Series(index=p_unique)
    for p in vol_unique.index:
        vol_unique.loc[p] = data['LastVolume'][data['Close'] == p].sum()

    plt.scatter(p_unique, vol_unique)
        
def drop_level1_index(data:pd.DataFrame):
    res = data.copy()
    res.index.names = ['a', 'InstrumentID', 'Date']
    id = res.index.droplevel('a')
    res.index = id
    del data

    return res.unstack(level='InstrumentID')

def ic_plot(factor:pd.DataFrame, ret):
    '''还没写完'''
    def get_ic_series(factor, ret):
        u = list(set(factor.columns).intersection(set(ret.columns)))
        ret = ret.loc[factor.index, u]
        icall = pd.DataFrame()
        fall = pd.merge(factor.stack(), ret.stack(), left_on=['date', 'InstrumentID'], right_on=['date', 'InstrumentID'])
        icall = fall.groupby('date').apply(lambda x : x.corr()['ret']).reset_index()
        icall = icall.dropna().drop(['ret'], axis=1).set_index('date')

        return icall
    
    ic_f = get_ic_series(factor, ret)
    ic_f.index = [str(i) for i in ic_f.index]
    f_name = ic_f.columns[0]
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes()
    xtick = np.arange(0, ic_f.shape[0], 20)
    xtick_label = pd.Series(ic_f.index[xtick])
    plt.bar(np.arange(ic_f.shape[0]), ic_f[f_name], color='darkred')
    
    ax1 = plt.twinx()
    ax1.plot(np.arange(ic_f.shape[0], ic_f.cumsum(), color='orange'))

    ax.set_xticks(xtick)
    ax.set_yticks(xtick_label)

    plt.show()


def factor_plot(factor:pd.DataFrame, gap=1, win=20):
    '''看因子的日期图像'''
    if len(factor.shape) < 1:
        if 'date' in factor.columns:
            factor = factor.set_index('date')
        elif 'Date' in factor.columns:
            factor = factor.set_index('Date')
    
    factor.index = [str(i) for i in factor.index]

    plt.figure(figsize=(12, 6))
    plt.plot(factor, color='#FF9999')
    plt.tight_layout()
    plt.xticks(ticks=range(0, len(factor), max(1, len(factor)//10)), 
               labels=factor.index[::max(1, len(factor)//10)], rotation=45)
    plt.title(f'Factor Plot')
    plt.show()

def factor_self_corr(factor:pd.DataFrame, gap=1, win=20):
    '''因子自相关性'''
    if len(factor.shape) < 1:
        if 'date' in factor.columns:
            factor = factor.set_index('date')
        elif 'Date' in factor.columns:
            factor = factor.set_index('Date')
    
    factor.index = [str(i) for i in factor.index]

    factor_corr = factor.rolling(win).apply(lambda x: x.corr(x.shift(-1)))

    plt.figure(figsize=(12, 6))
    plt.plot(factor_corr, color='#FF9999')
    plt.tight_layout()
    plt.xticks(ticks=range(0, len(factor_corr), max(1, len(factor_corr)//10)), 
               labels=factor_corr.index[::max(1, len(factor_corr)//10)], rotation=45)
    plt.title(f'Factor Self-rolling{win}-shift{gap}-Corr')
    plt.show()

def twist_factor(factor:pd.DataFrame):
    '''扭转因子 按照中位数做对称化'''
    factor = ((factor.T - factor.median(axis=1)) / (factor.std(axis=1) + 1e-10)).T
    factor = factor.applymap(lambda x: x if x > 0 else -x)
    factor = ((factor.T - factor.mean(axis=1)) / (factor.std(axis=1) + 1e-10)).T
    return factor

def factor_distribution_plot(data):
    '''
    将因子值二维数组转化为分布图形式
    '''
    import seaborn as sns

    if len(data.shape) > 1:
        data = data.stack()

    plt.figure(figsize=(10, 6))

    plt.hist(data, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Histogram')
    sns.kdeplot(data, color='red', label='KDE')

    plt.title('Data Distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()

    plt.show()