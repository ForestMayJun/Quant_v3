'''
因子回测系统
'''
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('/mnt/datadisk2/aglv/aglv/lab_aglv/forintern/')
from DataDaily import DataDaily

def factor_group(factor, group_number=10):
    '''按照因子值分组,grp1最小'''
    fg = factor.apply(
        lambda x: pd.qcut(x.rank(method='first'), q=group_number, labels=False) + 1, 
        axis=1)
        
    def _factor_group(factor_daily:pd.Series):
        dic = {f'group{str(g_value)}':factor_daily[factor_daily == g_value].index.tolist() for g_value in range(1, 1+group_number)}
        return pd.Series(dic)
    
    return {'group_stock':fg.apply(_factor_group, axis=1), 'group_index':fg}

def group_backtest(price:pd.DataFrame, factor:pd.DataFrame, group_number=10):
    '''
    分组回测,日度调仓,所有股票等权买入(买入各个股票的价值相同)  

    Paras:
    price:日频价格数据,index:Date,
    factor:日度因子值,index:Date,
    group_number:分组数,默认值为10, 组数大的因子值大

    Return:
    index:日期, col:组别, value: 指定日期下指定组数的收益率
    '''

    if 'Date' in factor.columns:
        factor = factor.set_index('Date')
    elif 'date' in factor.columns:
        factor = factor.set_index('date')

    factor = factor.shift(1).iloc[1:] # 第i日的因子作为第i+1日对应股票的调仓依据
    
    ret = price / price.shift(1)
    ret = ret.loc[factor.index, factor.columns].fillna(1)

    fg = factor_group(factor, group_number=group_number)['group_index']
    
    dic = {}
    for g_value in tqdm(range(1, 1+group_number)):
        is_in_group = fg.applymap(lambda x: 1 if x == g_value else 0)
        group_ret_daily = (ret * is_in_group).sum(axis=1) / is_in_group.sum(axis=1)
        dic[f'group{g_value}'] = group_ret_daily

    g_ret = pd.DataFrame(dic)
    # g_ret['g_mean'] = g_ret.mean(axis=1)
    g_ret['benchmark'] = ret.mean(axis=1)

    return g_ret

def position_backtest(price:pd.DataFrame, position:pd.DataFrame):
        '''
        有待完成
        用仓位数据来进行回测,
        price:日度价格数据
        position:对应的日度股票持仓数量
        '''
        holding = price * position
        holding_all = holding.sum(axis=1)
        ret = holding_all / holding_all - 1

        return ret

def group_backtest_plot(price:pd.DataFrame, factor:pd.DataFrame, group_number=5, save_path=None, is_zz1000=False, gap=1):
    '''
    分组回测
    Para:
    price:日频价格数据,index:Date,
    factor:日度因子值,index:Date,
    group_number:分组数,默认值为10, 组数大的因子值大
    '''

    if gap == 1:
        g_ret = group_backtest(price=price, factor=factor, group_number=group_number)
    else:
        g_ret = group_backtest_gapday(price=price, factor=factor, group_num=group_number, gap=gap) 

    g_ret.index = [str(i) for i in g_ret.index]
    g_ret_cumsum = g_ret.cumprod()

    if is_zz1000:
        index_data = pd.read_csv('/mnt/datadisk2/aglv/aglv/lab_aglv/index_data/close.csv')
        index_data.set_index('Unnamed: 0', inplace=True)
        zz1000 = index_data.loc[852, :]
        zz1000 = zz1000.loc[g_ret.index]
        zz1000_bt = zz1000 / zz1000.iloc[0]
        g_ret_cumsum['zz1000'] = zz1000_bt
    
    plt.style.use('seaborn')
    g_ret_cumsum.plot(figsize=(12, 6), linewidth=1)

    if is_zz1000:
        plt.plot(zz1000_bt, linewidth=2)

    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, facecolor='white', dpi=300)
    plt.show()


def group_backtest_gapday(price:pd.DataFrame, factor:pd.DataFrame, group_num=5, gap=10):
    '''
    换仓期为gap日的回测系统
    '''

    def handle_factor(factor=factor, gap=gap):
        '''将因子值按gap日求平均得到新的间隔gap日的因子序列'''
        if 'Date' in factor.columns:
            factor.set_index('Date', inplace=True)
        elif 'date' in factor.columns:
            factor = factor.set_index('date')
        
        new_index = [factor.index[i] for i in range(0, len(factor.index), gap)]
        new_factor = pd.DataFrame(
            [factor.loc[new_index[i]:new_index[i+1]].mean() for i in range(len(new_index) - 1)], 
            index=new_index[:-1]
        )
        
        return new_factor, new_index
    
    new_factor, new_index = handle_factor(factor=factor)

    # 做axis=1的因子中性化 方便fillna(0)
    new_factor = ((new_factor.T - new_factor.T.mean()) / (new_factor.T.std() + 1e-10)).T.fillna(0)
    
    new_factor = new_factor.shift(1).iloc[1:] # shift意为 日期i作为日期i+1的分组指标
    
    ret = pd.DataFrame(
        [price.loc[new_index[i]] / (price.loc[new_index[i-1]] + 1e-10) for i in range(1, len(new_index))],
        index=new_index[1:]
    )
    ret = ret.loc[new_factor.index, new_factor.columns].fillna(1)

    fg = factor_group(new_factor, group_number=group_num)['group_index']
    
    dic = {}
    for g_value in tqdm(range(1, 1+group_num)):
        is_in_group = fg.applymap(lambda x: 1 if x == g_value else 0)
        group_ret_daily = (ret * is_in_group).sum(axis=1) / is_in_group.sum(axis=1)
        dic[f'group{g_value}'] = group_ret_daily

    g_ret = pd.DataFrame(dic)
    g_ret['benchmark'] = ret.mean(axis=1)

    return g_ret


def check_extrme_factor(factor:pd.DataFrame, date, rolling_day=10, is_large=True, check_num=10, check_period=5):
    '''
    检测某天的极端因子值的股票表现
    因子值按照向前rolling 10天的平均值进行排序 绘图显示前后5天的股票表现, 在jupyter中检测
    需要用到datadaily类获取日频价格数据
    '''

    if 'date' in factor.columns:
        factor = factor.set_index('date')
    elif 'Date' in factor.columns:
        factor = factor.set_index('Date')

    if date not in factor.index:
        raise ValueError('日期为非交易日')

    factor = factor.rolling(10).mean()
    factor_date = factor.loc[date]
    factor_date_sorted = factor_date.sort_values().dropna()

    if is_large:
        check_stock = factor_date_sorted.iloc[-check_num:].index
    else:
        check_stock = factor_date_sorted.iloc[:check_num].index
    
    close = datadaily.adjclose
    for stock in check_stock:
        date_id = close.index.get_loc(date)
        p_date = close.iloc[date_id-check_period:date_id+check_period+1, :].loc[:, stock]
        p_date.index = [str(i) for i in p_date.index]

        plt.plot(p_date)
        plt.title(stock)
        plt.show()


def factors_corr(factor_folder):
    '''检测因子之间相关性'''
    import glob, os
    import seaborn as sns

    csv_path = glob.glob(os.path.join(factor_folder, '*.csv'))
    new_path = []
    for p in csv_path:
        f_name = p.split('/')[-1][:-4]
        if 'relu' in f_name:
            new_path.append(p)
    
    new_path = new_path[:8]

    dfs = [pd.read_csv(f).set_index('date') for f in new_path]

    num = len(dfs)
    print(num)

    corrs = pd.DataFrame(index=range(num), columns=range(num))
    for i in range(num):
        print(i, new_path[i].split('/')[-1][:-4])
        for j in range(num):
            if j < i:
                corrs.iloc[i, j] = corrs.iloc[j ,i]
            elif j == i:
                corrs.iloc[i, j] = 1
            else:
                corrs.iloc[i, j] = dfs[i].corrwith(dfs[j]).mean()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corrs.astype(float), annot=True, cmap='Greens', fmt=".2f")
    plt.title('Correlation with Means Heatmap')
    plt.show()


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

def factor_self_corr(factor:pd.DataFrame, gap=1, win=20):
    '''因子自相关性检测'''
    if len(factor.shape) < 1:
        if 'date' in factor.columns:
            factor = factor.set_index('date')
        elif 'Date' in factor.columns:
            factor = factor.set_index('Date')
    
    factor.index = [str(i) for i in factor.index]

    factor_corr = factor.rolling(win).apply(lambda x: x.corr(x.shift(-1)))

    plt.figure(figsize=(12, 6))
    plt.plot(factor_corr)
    plt.tight_layout()
    plt.xticks(ticks=range(0, len(factor_corr), max(1, len(factor_corr)//10)), 
               labels=factor_corr.index[::max(1, len(factor_corr)//10)], rotation=45)
    plt.show()

def factor_plot(factor:pd.DataFrame, gap=1, win=20):
    '''因子值绘图检测'''
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

def ic_plot(factor, ret):
    '''绘制月度ic和累计ic情况'''
    def get_ic_series(factor, ret):
        icall = pd.DataFrame()
        fall = pd.merge(factor, ret, left_on=['date', 'stock'], right_on=['date', 'stock'])
        icall = fall.groupby('date').apply(lambda x : x.corr()['ret']).reset_index()
        icall = icall.dropna().drop(['ret'], axis=1).set_index('date')

        return icall
    
    ic_f = get_ic_series(factor, ret)
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




def main():
    pass

if __name__ == '__main__':
    datadaily = DataDaily()
    close = datadaily.adjclose

    f1 = pd.read_csv('/mnt/datadisk2/aglv/aglv/lab_aglv/aglv_factor/factor_829/inaday_cvar_neg_0.9_21_23_min.csv')
    group_backtest_gapday(close, f1, gap=10)