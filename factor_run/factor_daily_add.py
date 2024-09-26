import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from forintern.DataDaily import DataDaily
# from forintern.utils import median_fill

from multiprocessing import Pool
import yaml
from tqdm import tqdm
from factor_daily import generate_onetime, out, factor_standarize
from asymmetric_alpha.utils import skewness, e_phi_series, s_phi_series, asym_p_series, cVaR_series, s_phi_series_version2, skew_scale
from asymmetric_alpha.utils import l_moments, asym_p_version2
from factor_utils import log_return

config_path = sys.argv[0].replace(".py", ".yaml")
config = yaml.load(open(config_path),Loader=yaml.SafeLoader)

class Config(object):
    def __init__(self, t):
        self.factor_minue_dir = config["factor_minue_dir"]
        self.factor_dir = config["factor_dir"]
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

    def update_to_newdate(self, cfg):
        data = DataDaily(check_data=False)
        u = data.universe_all
        u = list(set(u).intersection(set(data.adjclose.columns)))
        epsilon = 10**-10
        start_date = cfg.start_date
        end_date = cfg.end_date
        # open_ = data.adjopen.loc[start_date:end_date, u]
        close = data.adjclose.loc[start_date:end_date, u]
        # close_log = log_return(close)
        close_pct = close.pct_change().fillna(0)
        
        # high = data.adjhigh.loc[start_date:end_date, u]
        # low = data.adjlow.loc[start_date:end_date, u]
        # volume = data.volume.loc[start_date:end_date, u]
        # mkt_cap = data.mkt_cap.loc[start_date:end_date, u]
        # amount = data.amount.loc[start_date:end_date, u]
        # avgprice = data.adjavgprice.loc[start_date:end_date, u]
        # open_amount = data.OpenAmount.loc[start_date:end_date, u]
        # turnover_rate = amount / mkt_cap
        # ret_oc = open_ / (close + epsilon) - 1
        # ret_oc[ret_oc>0.3]=0
        # ret_oc[ret_oc<-0.3]=0
        # ret_oc_r = ret_oc.sub(ret_oc.mean(axis=1), axis=0) 
        # ret_hc = high / (close + epsilon) - 1
        # ret_ol = open_ / (low + epsilon) - 1
        # factor1 = (ret_oc>0).rolling(20).sum().fillna(0)
        # factor2 = (ret_oc>0).rolling(5).sum().fillna(0)
        # factor3 = (ret_oc>0.01).rolling(20).sum().fillna(0)
        # factor4 = (ret_oc>0.01).rolling(5).sum().fillna(0)
        # factor5 = (ret_oc_r>0).rolling(20).sum().fillna(0)
        # factor6 = (ret_oc_r>0).rolling(5).sum().fillna(0)
        # factor7 = (ret_oc_r>0.01).rolling(20).sum().fillna(0)
        # factor8 = (ret_oc_r>0.01).rolling(5).sum().fillna(0)
        # factor9 = ((ret_oc>0)&(ret_hc<0.01)).rolling(20).sum().fillna(0)
        # factor10 = ((ret_oc>0)&(ret_hc<0.01)).rolling(5).sum().fillna(0)
        # factor11 = ((ret_oc>0.01)&(ret_hc<0.03)).rolling(20).sum().fillna(0)
        # factor12 = ((ret_oc>0.01)&(ret_hc<0.03)).rolling(5).sum().fillna(0)

        # obj = Asymmetric(close_pct)
        # def rolling_series(series, window=60):
        #     res = pd.Series(index=series.index)
        #     for i in range(window, len(series)):
        #         obj = Asymmetric(data.iloc[i-window:i])
        #         res.iloc[i] = obj.skewness.f

        tqdm.pandas(desc='processing')
        
        factor1 = close_pct.rolling(20).progress_apply(cVaR_series)
        # factor2 = close_pct.rolling(30).progress_apply(cVaR_series)
        # factor3 = close_pct.rolling(20).progress_apply(lambda x:cVaR_series(x, c_level=0.8))

        p = Pool(min(64, len(cfg.factor_names)))
        for factor_name, name in zip(cfg.factor_names, cfg.factor_names_stand):
            p.apply_async(out, args=(
                eval(f'{factor_name}'), cfg, factor_name, name, '-quick', 5))
        p.close()
        p.join()

        p = Pool(min(64, len(cfg.factor_names)))
        for factor_name, name in zip(cfg.factor_names, cfg.factor_names_stand):
            p.apply_async(out, args=(
                eval(f'{factor_name}'), cfg, factor_name, name))
        p.close()
        p.join()
        print("update success")


def main():
    if len(sys.argv) == 2:
        print('update factor...')
        cfg = Config(sys.argv[1])
        generate_onetime(cfg)
        print('update success')


if __name__ == '__main__':

    main()
