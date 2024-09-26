
import sys
import yaml
import pandas as pd, numpy as np
from typing import Union
import matplotlib.pyplot as plt
from utils import get_trade_date
from lil_tools import shift_trade_date
from DataDaily import DataDaily

# config_path = sys.argv[0].replace(".py", ".yaml")
config_path = "/home/hzli/data_home/code/my_code/tools_factors/factor_tester_1_2.yaml"
config = yaml.load(open(config_path),Loader=yaml.SafeLoader)

class FactorTester1:
    def __init__(self, config, start_date:int=None, end_date:int=None):
        self.start_date = start_date if start_date else config['start_date']
        self.end_date = end_date if end_date else config['end_date']
        self.path_factor = config['path_factor']
        self.names_factor = config['names_factor']
        self.thres_pct_positive = 0.333
        self.if_ffill = config["if_ffill"]
        print(self.if_ffill)
        if self.if_ffill:
            self.ffill_method = config['ffill_method']
            self.is_ew = 'ew' in self.ffill_method
            self.ff_lim = config['ffill_lim']
            self.halflife = config['halflife']
        self.path_index = "./data/index_data"
        self.dic_index = {300: 'hs300',
                          905: 'zz500',
                          852: 'zz1000'}
        self.rm_limit = True
        # self.rm_limit = False
        self.path_save = config['path_save']
        # self.out_pos_ret = False
        self.out_pos_ret = True
        self.index_default = 852

    def to_fact_format(self, df_ser_, is_stk=True):
        df_ser = df_ser_.copy()
        df_ser.index = [pd.to_datetime(str(dt_)) for dt_ in df_ser.index]
        df_ser.index.name = df_ser_.index.name
        if is_stk and isinstance(df_ser_, pd.DataFrame):
            df_ser.columns = [c_[2:]+'.'+c_[:2] for c_ in df_ser.columns]
            df_ser.columns.name = df_ser_.columns.name
        return df_ser


    def to_str_index(self, df_ser:Union[pd.Series, pd.DataFrame], inplace=False):
        if inplace:
            df_ser.index = df_ser.index.astype(str)
            return df_ser
        else:
            df_ser_ = df_ser.copy()
            df_ser_.index = df_ser_.index.astype(str)
            return df_ser_

    def read_factors(self):
        dic_df_fact = {}
        for name_ in self.names_factor:
            df_ = pd.read_csv(f"{self.path_factor}/{name_}.csv", index_col=0)
            # df_ = df_.sort_index()
            print(df_)
            dic_df_fact[name_] = df_.copy()
            print(f"read factor {name_} OK ")
        return dic_df_fact


    def prepare_datadaily(self):
        datadaily = DataDaily()
        close = datadaily.adjclose
        univ = datadaily.universe_all
        univ = list(set(univ).intersection(set(close.columns)))
        close = close.loc[:, univ]
        lim_dn = (datadaily.LimitBoard).reindex(index=close.index, columns=close.columns)
        lim_up = (datadaily.StockBoard).reindex(index=close.index, columns=close.columns)
        lim_ud = lim_up - lim_dn
        self.univ = univ
        self.close = close
        self.lim_ud = lim_ud


    def get_returns(self):
        date_start_ret = shift_trade_date(self.start_date, -25)
        date_end_ret =  shift_trade_date(self.end_date, 25)
        # stock return

        ret_cc1 = np.log((self.close.shift(-1)) / self.close)
        ret_cc1 = ret_cc1.loc[date_start_ret:date_end_ret]

        # index return
        index_close = pd.read_csv(f"{self.path_index}/close.csv", index_col=0).T
        index_close.index = index_close.index.astype(int)
        index_close = index_close.loc[20150101:]
        ret_index_cc1 = np.log(index_close.shift(-1) / index_close)
        ret_index_cc1 = ret_index_cc1.loc[date_start_ret:date_end_ret]
        print('get returns OK ')

        return ret_cc1, ret_index_cc1


    def extend_df_factor(self, df_fact:pd.DataFrame, from_config:bool=False, ff_lim:int=60, 
                         is_ew:bool=True, halflife:int=30, is_event:bool=True):
        if from_config:
            print('parameters for extend from config! ')
            ff_lim = self.ff_lim
            is_ew = self.is_ew
            halflife = self.halflife
        date_start_ff = shift_trade_date(self.start_date, -ff_lim)
        df_fact_ = df_fact.loc[date_start_ff: self.end_date]
        df_fact_ff = df_fact_.ffill(limit=ff_lim)
        if is_ew and (not is_event):
            ff_isna = df_fact_ff.isna()
            df_fact_ew = df_fact.fillna(0)
            for i in range(1, halflife+1):
                df_fact_ew = df_fact_ew + df_fact.shift(i).fillna(0) * np.exp((np.log(0.5) * i / halflife))
            df_fact_ew[ff_isna] = np.nan
            df_fact_e = df_fact_ew
        else:
            df_fact_e = df_fact_ff
        if self.rm_limit:
            print('remove factor values for limit up/down stocks x dates ')
            lim_ud_rdx = self.lim_ud.reindex(index=df_fact.index, columns=df_fact.columns).fillna(0)
            is_lim_ud = (lim_ud_rdx == 1) | (lim_ud_rdx == -1)
            df_fact_e.mask(is_lim_ud, inplace=True)
            # print('df_fact_r', df_fact_r)
            # print('diff_lim', (df_fact_r.fillna(0) - df_fact).abs().sum(axis=1).sum())
        return df_fact_e


    def get_position_from_df_factor(self, df_fact:pd.DataFrame, is_event:bool=False, is_rank:bool=True):
        ## index: date1
        df_fact_r = df_fact.copy()
        if is_event:
            return df_fact_r
        if is_rank:
            df_fact_r = df_fact_r.rank(pct=True, axis=1) * 2 - 1
        df_fact_rpos = (df_fact_r >= 0) * df_fact_r
        df_fact_rneg = (df_fact_r < 0) * df_fact_r
        df_fact_rpos = df_fact_rpos.div(df_fact_rpos.sum(axis=1), axis=0)
        df_fact_rneg = df_fact_rneg.div(df_fact_rneg.sum(axis=1), axis=0)
        df_fact_position = df_fact_rpos - df_fact_rneg
        print('fact position OK ')
        return df_fact_position
    

    def prepare_position(self):
        ## read and extend factors
        dic_df_fact = self.read_factors()
        dic_fact_e = {}
        dic_fact_pos = {}
        dic_is_event = {}
        for name_ in self.names_factor:
            ## check if it is event type (number of different values <= threshold(=10))
            thres_n_fact = 10
            vals_u = pd.unique(dic_df_fact[name_].values.ravel())
            is_event = (len(vals_u) <= thres_n_fact)
            dic_is_event[name_] = is_event
            if is_event:
                print(f'event factor: {name_} ')
            if self.if_ffill:
                df_fact_e = self.extend_df_factor(dic_df_fact[name_], from_config=True, is_event=is_event)
            else:
                df_fact_e = dic_df_fact[name_].copy()
            df_fact_e = df_fact_e.sort_index().loc[self.start_date:self.end_date]
            dic_fact_e[name_] = df_fact_e.copy()
            df_fact_pos = self.get_position_from_df_factor(df_fact_e, is_event=is_event)
            dic_fact_pos[name_] = df_fact_pos.copy()
        return dic_fact_e, dic_fact_pos, dic_is_event


    def align_pos_rets(self, df_fact_pos:pd.DataFrame, ret_stk:pd.DataFrame, ret_idx:pd.DataFrame):
        chg_fmt = False
        if chg_fmt:
            ret_stk_s1dt = self.to_fact_format(ret_stk).shift(-1)
            ret_idx_s1dt = self.to_fact_format(ret_idx, is_stk=False).shift(-1)
        else:
            ret_stk_s1dt = ret_stk.shift(-1)
            ret_idx_s1dt = ret_idx.shift(-1)
        ret_stk_s1dt = ret_stk_s1dt.reindex(index=df_fact_pos.index, columns=df_fact_pos.columns)
        ret_idx_s1dt = ret_idx_s1dt.reindex(index=df_fact_pos.index)
        return ret_stk_s1dt, ret_idx_s1dt
    

    def get_ic_rank_ic(self, df_fact_e:pd.DataFrame, ret_stk:pd.DataFrame, ret_all:pd.DataFrame=None):
        ### is_event = False
        ser_ic = df_fact_e.corrwith(ret_stk, axis=1)
        ser_rank_ic = df_fact_e.corrwith(ret_stk, axis=1, method='spearman')
        df_ic = pd.concat([ser_ic.to_frame('ic'), ser_rank_ic.to_frame('rank_ic')], axis=1)
        if ret_all is not None:
            ret_all_dt = ret_all.reindex(index=df_fact_e.index)
            df_fact_e_r = df_fact_e.reindex(columns=ret_all.columns)
            ser_ic_all = df_fact_e_r.corrwith(ret_all_dt, axis=1)
            ser_rank_ic_all = df_fact_e_r.corrwith(ret_all_dt, axis=1, method='spearman')
            df_ic = pd.concat([df_ic, ser_ic_all.to_frame('ic_all'), 
                               ser_rank_ic_all.to_frame('rank_ic_all')], axis=1)
        return df_ic
    

    def get_group_ret_means(self, df_fact_: pd.DataFrame, ret_stk:pd.DataFrame, ret_idx:pd.DataFrame=None, n_grp:int=5):
        ### if index return not provided, use de-meaned return
        if ret_idx is None:
            ret_stk_e = ret_stk.sub(ret_stk.mean(axis=1), axis=0)
            print('subtract by return mean')
        else:
            ret_stk_e = ret_stk.sub(ret_idx[self.index_default], axis=0)
            print(f'subtract by index {self.index_default} return')
        df_fact = df_fact_.copy()
        df_fact.to_csv(f"{self.path_save}/test_df_fact_g{n_grp}_01.csv")
        df_fact[df_fact.abs() < 1e-4] = np.nan
        df_fact_qc = df_fact.apply(lambda x: pd.qcut(x, n_grp, labels=False, duplicates='drop'), axis=1)
        # df_fact_qc = df_fact.apply(lambda x: pd.qcut(x, n_grp, labels=False), axis=1)
        df_fact_qc = df_fact_qc + 1
        df_fact_qc.to_csv(f"{self.path_save}/test_df_fact_qc_g{n_grp}_01.csv")
        df_ret_grp = pd.concat([df_fact_qc.stack(dropna=False).to_frame('q_group'),
                                ret_stk_e.stack(dropna=False).to_frame('ret_stk_e')], axis=1)
        df_ret_grp.index.names = ['date', 'symbol']
        df_ret_gm = df_ret_grp.groupby(['date', 'q_group'])['ret_stk_e'].mean().unstack()
        df_ret_gm.columns = [int(str(c_).split('.')[0]) for c_ in df_ret_gm.columns]
        print('return group mean: ', df_ret_gm)
        df_ret_gm.to_csv(f"{self.path_save}/test_df_ret_dm_g{n_grp}_01.csv")
        cs_ret_gm = df_ret_gm.fillna(0).cumsum()
        print('return group mean cumsum: ', cs_ret_gm)
        cs_ret_gm.to_csv(f"{self.path_save}/test_df_cs_ret_dm_g{n_grp}_01.csv")
        return df_ret_gm, cs_ret_gm

    
    def get_position_return(self, df_fact_pos:pd.DataFrame, ret_stk:pd.DataFrame, ret_idx:pd.DataFrame, is_event:bool=False):
        if is_event:
            ### only handle cases with factor values -1,0,1
            vals_u = pd.unique(df_fact_pos.values.ravel())
            ret_ptf_long = ret_stk[df_fact_pos > 0]
            ret_ptf_0 = ret_stk[df_fact_pos == 0]
            has_short = (len([v_ for v_ in vals_u if v_ < 0]) > 0)
            if has_short:
                ret_ptf_short = ret_stk[df_fact_pos < 0]
                ret_ptf = ret_ptf_long.mean(axis=1) + (-1) * ret_ptf_short.mean(axis=1)
            else:
                ret_ptf = ret_ptf_long.mean(axis=1)
        else:
            df_ret_pos = df_fact_pos * ret_stk
            ret_ptf = df_ret_pos.sum(axis=1)

        pct_positive = ((ret_ptf > 0).sum()) / (ret_ptf.shape[0])
        
        if pct_positive < self.thres_pct_positive:
            print(f"pct positive {pct_positive} < {self.thres_pct_positive}, change factor sign ")
            chg_sgn = True
            df_fact_pos *= -1
            df_ret_pos *= -1
            ret_ptf *= -1
        else:
            print(f"pct positive {pct_positive} >= {self.thres_pct_positive}, factor sign unchanged ")
            chg_sgn = False
        df_ptf_ret = pd.DataFrame(dtype=float)    
        df_ptf_ret.loc[:, 'ret_abs'] = ret_ptf.copy()

        for i_ in self.dic_index.keys():
            df_ptf_ret.loc[:, self.dic_index[i_]] = ret_ptf - ret_idx[i_]

        df_fact_pos_long = df_fact_pos.copy()
        if is_event and chg_sgn:
            print("negative event, print negative return ")
            df_fact_pos_long[df_fact_pos_long > 0] = 0
        else:
            df_fact_pos_long[df_fact_pos_long < 0] = 0
        df_ret_pos_long = df_fact_pos_long * ret_stk
        ret_ptf_long = df_ret_pos_long.sum(axis=1)
        df_ptf_ret_long = pd.DataFrame(dtype=float)    
        df_ptf_ret_long.loc[:, 'ret_abs'] = ret_ptf_long.copy()

        for i_ in self.dic_index.keys():
            df_ptf_ret_long.loc[:, self.dic_index[i_]] = ret_ptf_long - ret_idx[i_]

        if (not is_event) and self.out_pos_ret:
            return df_ptf_ret, df_ptf_ret_long, df_ret_pos, df_ret_pos_long
        else:
            return df_ptf_ret, df_ptf_ret_long
    

    def map_stk_index(self, df_fact:pd.Series, as_of_date:int=None, return_count:bool=True):
        if as_of_date is None:
            as_of_date = self.start_date
        ## use universe from index_data
        map_idx = []
        for i_ in self.dic_index.values():
            list_stk_ = pd.read_csv(f"{self.path_index}/universe_{i_}.csv", header=None).iloc[:, 0].to_list()
            map_stk_ = pd.Series(index=list_stk_, data=i_)
            map_idx.append(map_stk_)
        map_idx = pd.concat(map_idx)

        ## 
        ser_map_idx = pd.Series(index=df_fact.columns, data=df_fact.columns.map(map_idx).fillna('else'))
        df_map_idx = ser_map_idx.to_frame(df_fact.index[0]).reindex(columns=df_fact.index).T.ffill()
        df_map_idx = df_map_idx.mask(df_fact.notna())
        count_idx = {}
        for i_ in list(self.dic_index.values()) + ['else']:
            count_idx[i_] = (df_map_idx == i_).sum(axis=1)
        count_idx = pd.concat(count_idx, axis=1)
        count_idx['n_stk'] = count_idx.sum(axis=1)
        
        if return_count:
            return count_idx
        else:
            return df_map_idx
    

    def run_pos_ret(self):
        ## prepare daily data 
        self.prepare_datadaily()
        ## get all fact pos
        dic_fact_e, dic_fact_pos, dic_is_event = self.prepare_position()
        ## get returns
        ret_cc1, ret_index_cc1 = self.get_returns()
        ## 
        dic_ptf_ret = {}
        dic_ptf_ret_long = {}
        dic_ct_idx = {}
        dic_ic = {}
        for name in self.names_factor:
            print(name)
            is_event = dic_is_event[name]
            if is_event:
                name_save = name + '_evt_ff' if self.if_ffill else name + '_evt_no_ff' 
            else:
                name_save = name + '_' + self.ffill_method if self.if_ffill else name + '_no_ff' 
            if self.rm_limit:
                name_save += '_rml'
            dic_fact_pos[name].to_csv(f"{self.path_save}/fact_pos_{name_save}.csv")
            ret_stk, ret_idx = self.align_pos_rets(dic_fact_pos[name], ret_cc1, ret_index_cc1)
            ### temporary: save
            ret_stk.to_csv(f"{self.path_save}/ret_stk_s1dt_{self.start_date}_{self.end_date}.csv")
            ret_idx.to_csv(f"{self.path_save}/ret_idx_s1dt_{self.start_date}_{self.end_date}.csv")
            
            ### position return
            position_return = self.get_position_return(dic_fact_pos[name], ret_stk, ret_idx, is_event=is_event)
            # print(len(position_return))
            df_ptf_ret, df_ptf_ret_long = position_return[0], position_return[1]
            dic_ptf_ret[name] = df_ptf_ret.copy()
            dic_ptf_ret[name].to_csv(f"{self.path_save}/df_ptf_ret_{name_save}.csv")
            print(dic_ptf_ret[name])
            dic_ptf_ret_long[name] = df_ptf_ret_long.copy()
            if not is_event:
                dic_ptf_ret_long[name].to_csv(f"{self.path_save}/df_ptf_ret_long_{name_save}.csv")
                print('long: ', dic_ptf_ret_long[name])
            if len(position_return) == 4:
                df_ret_pos, df_ret_pos_long = position_return[2], position_return[3]
                df_ret_pos.to_csv(f"{self.path_save}/ret_pos_{name_save}.csv")
                df_ret_pos_long.to_csv(f"{self.path_save}/ret_pos_long_{name_save}.csv")
            
            ### count stocks in indices 300,500,1000
            ct_idx = self.map_stk_index(dic_fact_e[name])
            dic_ct_idx[name] = ct_idx.copy()
            dic_ct_idx[name].to_csv(f"{self.path_save}/count_index_{name_save}.csv")
            print('count index: ', dic_ct_idx[name])

            ### ic for non-event factor
            if not is_event:
                # df_ic = self.get_ic_rank_ic(dic_fact_e[name], ret_stk)
                df_ic = self.get_ic_rank_ic(dic_fact_e[name], ret_stk, ret_cc1)
                dic_ic[name] = df_ic.copy()
                dic_ic[name].to_csv(f"{self.path_save}/df_ptf_ic_{name_save}.csv")
                print('ic: ', dic_ic[name])
                # print('ret all: ', ret_cc1)
                # print(dic_fact_e[name])
                
                # self.get_group_ret_means(dic_fact_pos[name], ret_stk, n_grp=10)
                # self.get_group_ret_means(dic_fact_e[name], ret_stk, n_grp=10)
                self.get_group_ret_means(dic_fact_e[name], ret_stk)

        return dic_ptf_ret, dic_ptf_ret_long, dic_ct_idx, dic_ic

    
    def save_plot_from_df_ser(self, df_ser_:Union[pd.DataFrame, pd.Series], 
                              name_plot:str, method_agg:str='', figsize:tuple=(16, 9)):
        df_ser = self.to_str_index(df_ser_)
        if ('cumsum' in method_agg) or ('cs' in method_agg):
            df_ser = df_ser.cumsum()
        elif ('cumprod' in method_agg) or ('cp' in method_agg):
            df_ser = (1 + df_ser).cumprod()
        title_plot = f"{name_plot.replace('_', ' ')} {method_agg}"
        ax = df_ser.plot(figsize=figsize, title=title_plot)
        fig = ax.get_figure()
        fig.savefig(f"{self.path_save}/plot_{method_agg}_{name_plot}.png")
        return df_ser


    def run_and_plot(self):
        dic_ptf_ret, dic_ptf_ret_long, dic_ct_idx, dic_ic = self.run_pos_ret()
        for name in self.names_factor:
            self.save_plot_from_df_ser(dic_ptf_ret[name], name_plot='ptf_ret_'+name, method_agg='cumprod')
            self.save_plot_from_df_ser(dic_ptf_ret_long[name], name_plot='ptf_ret_long_'+name, method_agg='cumprod')
            self.save_plot_from_df_ser(dic_ct_idx[name], name_plot='ct_idx_'+name)
            if len(dic_ic) > 0:
                self.save_plot_from_df_ser(dic_ic[name], name_plot='ic_'+name, method_agg='cumsum')


if __name__ == '__main__':
    if len(sys.argv) > 2:
        FT1 = FactorTester1(config, int(sys.argv[1]), int(sys.argv[2]))
    elif len(sys.argv) == 2:
        FT1 = FactorTester1(config, int(sys.argv[1]), int(sys.argv[1]))
    else:
        FT1 = FactorTester1(config, 20210101, 20230930)

    FT1.run_pos_ret()
    # FT1.run_and_plot()

