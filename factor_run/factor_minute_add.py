from multiprocessing import Pool
import pandas as pd
import numpy as np
import datetime
import os
import shutil
import sys
from utils import get_trade_date, median_fill, my_skew, my_kurt
import yaml
from factor_minute import generate_onetime, factor_standarize, update_to_newdate, out, generate_history

config_path = sys.argv[0].replace(".py", ".yaml")
config = yaml.load(open(config_path),Loader=yaml.SafeLoader)

class Config(object):
    def __init__(self, timepoint, t):
        self.minute_dir = config["minute_dir"]
        self.factor_minue_dir = config["factor_minue_dir"]
        self.factor_dir = config["factor_dir"]
        self.stock_name_list = pd.read_csv(
            config["stock_name_list"], header=None, sep=' ').iloc[:, 0].to_list()
        self.timepoint = timepoint
        self.timepoint_now = str(timepoint)[:2] + ":" + str(timepoint)[2:]
        if "factor_names_normal" in config.keys():
            self.factor_names = [config["factor_names_normal"].replace('0', str(tmp)) for tmp in range(int(config["factor_names"][0]), int(config["factor_names"][1])+1)]
        else:
            self.factor_names = config["factor_names"]
        if "factor_names_stand_normal" in config.keys():
            self.factor_names_stand = [config["factor_names_stand_normal"].replace('0', str(tmp)) for tmp in range(int(config["factor_names_stand"][0]), int(config["factor_names_stand"][1])+1)]
        else:
            self.factor_names_stand = config["factor_names_stand"]
        
        self.start_date = config["start_date"]
        self.end_date = int(t)
        self.factor_test = self.factor_minue_dir + \
            self.factor_names[0] + f'_{timepoint}.csv'
    
    def get_kindle_factor_oneday(self, cfg, t):
        df = pd.read_hdf(f"{cfg.minute_dir}/{t}.h5")
        tradedate = pd.read_csv("./trade_date_long.csv")
        p = pd.Series(list(range(len(tradedate))))[
            tradedate.iloc[:, 0] == t].values[0]
        t_shift = tradedate.iloc[p-1, 0]
        epsilon = 10**-10
        df_2 = pd.read_hdf(f"{cfg.minute_dir}/{t_shift}.h5")
        """factor"""
        df_2['EndTime'] = df_2['EndTime']
        df['EndTime'] = df['EndTime'] + 'today'
        u1 = set(cfg.stock_name_list).intersection(set(df.index.unique()))
        u2 = set(cfg.stock_name_list).intersection(set(df_2.index.unique()))
        u = u1.intersection(u2)
        null_stock_list = set(cfg.stock_name_list) - set(u)
        if cfg.timepoint != 1500:
            df = df[df['EndTime'] <= cfg.timepoint_now+'today']
        df = df.reset_index().set_index('InstrumentID').loc[u, :].reset_index(
        ).set_index(['EndTime', 'InstrumentID']).unstack()
        df_2 = df_2.reset_index().set_index('InstrumentID').loc[u, :].reset_index(
        ).set_index(['EndTime', 'InstrumentID']).unstack()
        df_3 = df_2.append(df)
        Time_point_all_today = df.index.values
        Time_point_all_last = df_2.index.values
        Time_point_all_all = df_3.index.values
        print(df_3)
        open_ = df_3['Open'].copy()
        open_.loc['09:31today', open_.loc['09:31today', :]
                == 0] = open_.loc['09:32today', :]
        close = df_3['Close']
        high = df_3['High']
        low = df_3['Low']
        ret = close / (open_ + epsilon) - 1
        ret_hl = high / (low + epsilon) - 1
        ret_sign = np.sign(ret)
        ret_sign_1000 = ret_sign.loc['10:01today':, :]
        ret_sign_today = ret_sign.loc['09:31today':, :]

        last_turnover = df_3['LastTurnOver']
        last_volume = df_3['LastVolume']
        last_volume_today = last_volume.loc['09:31today':, :]
        last_turnover_today = last_turnover.loc['09:31today':, :]
        last_volume_1000 = last_volume_today.loc['10:01today':, :]
        last_turnover_1000 = last_turnover_today.loc['10:01today':, :]

        tmp = close.sub(high.loc['09:31today':, :].max(axis=0), axis=1).div(close.mean(axis=0), axis=1)
        tmp2 = close.sub(high.loc['09:31today':, :].max(
            axis=0), axis=1).sub(close.std(axis=0), axis=1).div(close.mean(axis=0), axis=1)
        factor1 = (tmp > 0).sum(axis=0) / 240
        factor2 = (tmp > 0.005).sum(axis=0) / 240
        factor3 = (tmp > 0.01).sum(axis=0) / 240
        factor4 = (tmp2 > 0).sum(axis=0) / 240
    #use min not max
        tmp = close.sub(low.loc['09:31today':, :].min(axis=0), axis=1).div(close.mean(axis=0), axis=1)
        tmp2 = close.sub(low.loc['09:31today':, :].min(
            axis=0), axis=1).sub(-close.std(axis=0), axis=1).div(close.mean(axis=0), axis=1)
        factor5 = (tmp < 0).sum(axis=0) / 240
        factor6 = (tmp < -0.005).sum(axis=0) / 240
        factor7 = (tmp < -0.01).sum(axis=0) / 240
        factor8 = (tmp2 < 0).sum(axis=0) / 240

        factor9 = high.apply(lambda x: x.argmax(axis=0)) / 480
        factor10 = high.loc['10:01today':, :].apply(
            lambda x: x.argmax(axis=0)) / 240
        factor11 = low.apply(lambda x: x.argmin(axis=0)) / 480
        factor12 = low.loc['10:01today':, :].apply(
            lambda x: x.argmin(axis=0)) / 240

        factor13 = last_volume.append(ret).apply(lambda x: np.corrcoef(
            x.values[:int(len(x)/2)], x.values[int(len(x)/2):])[0, 1], axis=0)
        factor14 = last_volume.loc['09:31today':, :].append(ret.loc['09:31today':, :]).apply(
            lambda x: np.corrcoef(x.values[:int(len(x)/2)], x.values[int(len(x)/2):])[0, 1], axis=0)

        factor_df = pd.DataFrame()
        for i in range(len(cfg.factor_names)):
            name = cfg.factor_names[i]
            f = pd.DataFrame({name: eval(f'{name}')})
            f[last_volume.sum() == 0] = np.nan
            f[close.sum() == 0] = np.nan
            factor_df = pd.concat([factor_df, f], axis=1)
        for x in null_stock_list:
            factor_df = factor_df.append(pd.Series(name=x, dtype='float'))
        factor_df = factor_df.reindex(cfg.stock_name_list)
        factor_df['date'] = int(t)
        factor_df = factor_df.reset_index().set_index(['date', 'InstrumentID'])
        return factor_df

def main(time, t="1500"):
    print(t, time)
    cfg = Config(t, time)
    generate_onetime(cfg)
    return


if __name__ == '__main__':
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2])
    else:
        main(sys.argv[1])