'''
计算p_volmax以及其他相关的数据  
终端运行第二个参数为储存文件名  
第三个参数为储存的文件文件夹地址  
'''

import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import multiprocessing
import sys

class PVolMax:
    def __init__(self, data, method, n_processing=64):
        self.data = data
        self.n_processing = n_processing
        self.method = method

    def _handle_stockily_pvolmax(self, data_stock:pd.DataFrame):
        def handle_stockly_daily(data_stock_daily: pd.Series):
            """
            对单只股票进行日频数据处理，计算成交量按价格的分布
            :param data_stock_daily: 单只股票的日频数据
            :return: 成交量最大的价格区间的中心值
            """
            g_num = 10
            min_s, max_s = data_stock_daily['Close'].min(), data_stock_daily['Close'].max()

            if min_s == max_s:
                return np.nan
            
            p_range = np.linspace(min_s, max_s, g_num)
            data_stock_daily['cate'] = pd.cut(data_stock_daily['Close'], p_range)
            g_value_vol = data_stock_daily.groupby('cate').sum()['LastVolume']
            p_volmax_iv = g_value_vol.idxmax()
            p_volmax = (p_volmax_iv.left + p_volmax_iv.right) / 2

            return p_volmax
        
        return data_stock.groupby(level='Date').apply(handle_stockly_daily)
    
    def _handle_stockly_pvolstd(self, daata_stock:pd.DataFrame):
        def handle_stockly_daily(data_stock_daily:pd.DataFrame):
            data_stock_daily = data_stock_daily.set_index('Close')['LastVolume']
            data_stock_daily = data_stock_daily.sort_index()
            data_stock_daily = np.log(data_stock_daily.replace(0, np.nan))
            return data_stock_daily.std()
        
        return daata_stock.groupby(level='Date').apply(handle_stockly_daily)
    
    def _handle_stockly(self, data_stock):
        if self.method == 'p_volmax':
            return self._handle_stockily_pvolmax(data_stock)
        elif self.method == 'p_logvol_dis_std':
            return self._handle_stockly_pvolstd(data_stock)
        

    def _process_group(self, group):
        """
        处理每个股票分组
        :param group: 股票的一个分组
        :return: 股票的 InstrumentID 和对应的处理结果
        """
        instrument_id, group_data = group
        return instrument_id, self._handle_stockly(group_data)
    
    def _multiprocessing_factor(self):
        """
        使用多进程计算因子
        :return: 计算得到的因子 DataFrame
        """
        grouped = list(self.data.groupby(level='InstrumentID'))
        with multiprocessing.Pool(processes=self.n_processing) as pool:
            results = list(tqdm(
                pool.imap(self._process_group, grouped), total=len(grouped)
            ))

        factor_list = []
        date_name = 'date' if 'date' in self.data.index.names else 'Date'
        for instrument_id, factor in results:
            factor_df = pd.DataFrame(
                factor.values,
                index=factor.index.get_level_values(date_name), 
                columns=[instrument_id]
            )
            factor_list.append(factor_df)

        factor = pd.concat(factor_list, axis=1)

        return factor
    
    def calculate(self):
        """
        外部调用方法, 返回计算结果
        """
        return self._multiprocessing_factor()
    

def main():
    args = sys.argv[1:]
    if len(args) == 0:
        path_factor = '/mnt/datadisk2/aglv/aglv/aglv_factor/new_pvol/'
    else:
        path_factor = args[0]
    

    print('loading data')
    data2123 = pd.read_hdf('/mnt/datadisk2/aglv/aglv/lab_aglv/min_clo_vol_2123.h5')
    print('loadding done')

    methods_list = [
        'p_volmax',
        'p_logvol_dis_std',
    ]

    model = PVolMax(data2123, methods_list[1])
    factor = model.calculate()

    file_name = 'p_vol_dis_std'
    factor.to_csv(path_factor + file_name + '.csv')


if __name__ == '__main__':
    main()