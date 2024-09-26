from multiprocessing import Pool
import pandas as pd
import numpy as np
import datetime
import os
import shutil
import sys
from utils import get_trade_date, median_fill
import warnings
warnings.filterwarnings("ignore")

def generate_onetime(cfg):
    if not os.path.exists(cfg.factor_test):
        generate_history(cfg)
    else:
        update_to_newdate(cfg)


def factor_standarize(df, cfg, name1, name2, label=''):
    x = median_fill(df)
    x_5 = (x.sort_index().rolling(
        5, min_periods=1).mean()).reindex(index=x.index)
    x_10 = (x.sort_index().rolling(
        10, min_periods=1).mean()).reindex(index=x.index)
    x_5_10 = (x_5-x_10).reindex(index=x.index)
    x_1_5 = (x-x_10).reindex(index=x.index)

    if int(cfg.timepoint) == 1500:
        x.to_csv(f"{cfg.factor_dir}/{name2}.csv"+label)
        x_5.to_csv(f"{cfg.factor_dir}/{name2}_5.csv"+label)
        x_10.to_csv(f"{cfg.factor_dir}/{name2}_10.csv"+label)
        x_5_10.to_csv(f"{cfg.factor_dir}/{name2}_5_10.csv"+label)
        x_1_5.to_csv(f"{cfg.factor_dir}/{name2}_1_5.csv"+label)
    else:
        x.to_csv(f"{cfg.factor_dir}/{name2}_{cfg.timepoint}.csv"+label)
        x_5.to_csv(f"{cfg.factor_dir}/{name2}_5_{cfg.timepoint}.csv"+label)
        x_10.to_csv(f"{cfg.factor_dir}/{name2}_10_{cfg.timepoint}.csv"+label)
        x_5_10.to_csv(
            f"{cfg.factor_dir}/{name2}_5_10_{cfg.timepoint}.csv"+label)
        x_1_5.to_csv(f"{cfg.factor_dir}/{name2}_1_5_{cfg.timepoint}.csv"+label)
    print(f"factor {name1}, {name2} standerdize success")


def update_to_newdate(cfg):
    dates = get_trade_date(cfg.start_date, cfg.end_date)
    l = pd.read_csv(cfg.factor_test, usecols=[0]).iloc[:, 0].values
    l = [int(tmp) for tmp in l]
    if_old = False
    for factor_name in cfg.factor_names:
        try:
            factor = pd.read_csv(
                f"{cfg.factor_minue_dir}/{factor_name}_{cfg.timepoint}.csv", usecols=[0]).iloc[:, 0].values
            factor = [int(tmp) for tmp in factor]
        except:
            print("file error read old file")
            factor = pd.read_csv(
                f"{cfg.factor_minue_dir}/{factor_name}_{cfg.timepoint}.csv.old", usecols=[0]).iloc[:, 0].values
            factor = [int(tmp) for tmp in factor]
            if_old = True
        l = set(l).intersection(set(factor))
        print(factor_name)
        print(len(l))
    old_dates = l
    add_dates = sorted(set(dates)-set(old_dates))
    add_dates = sorted(set(add_dates + [int(cfg.end_date)]))
    print(f"total trading dates to add: {len(add_dates)}")
    if len(add_dates) == 0:
        print("no need to update")
        return
    df_all = pd.DataFrame()
    for d in add_dates:
        try:
            df_one = cfg.get_kindle_factor_oneday(cfg, d)
            print(df_all)
            print(df_one)
            df_all = df_all.append(df_one)
            print(f"{d} add success")
        except Exception as e:
            print(f"{d} add error {e}")

    p = Pool(min(64, len(cfg.factor_names)))
    for factor_name, name in zip(cfg.factor_names, cfg.factor_names_stand):
        p.apply_async(out, args=(df_all[factor_name].unstack(
        ), cfg, factor_name, name, if_old, '-quick', 12))
    p.close()
    p.join()

    p = Pool(min(64, len(cfg.factor_names)))
    for factor_name, name in zip(cfg.factor_names, cfg.factor_names_stand):
        p.apply_async(out, args=(
            df_all[factor_name].unstack(), cfg, factor_name, name, if_old))
    p.close()
    p.join()
    print("update success")


def out(f, cfg, factor_name, name, if_old, label='', n=100000):
    if if_old:
        old_f = pd.read_csv(
            f"{cfg.factor_minue_dir}/{factor_name}_{cfg.timepoint}.csv.old", index_col=0)
    else:
        old_f = pd.read_csv(
            f"{cfg.factor_minue_dir}/{factor_name}_{cfg.timepoint}.csv", index_col=0)
    print(factor_name)
    f = pd.concat([f, old_f]).sort_index(ascending=False)
    f = f.loc[~f.index.duplicated(
        keep='first')].sort_index(ascending=False)
    f = f.loc[[tmp for tmp in f.index if tmp <= cfg.end_date]] 
    if not if_old:
        shutil.copy(f"{cfg.factor_minue_dir}/{factor_name}_{cfg.timepoint}.csv",
                    f"{cfg.factor_minue_dir}/{factor_name}_{cfg.timepoint}.csv.old")
    if n > 20:
        f.to_csv(f"{cfg.factor_minue_dir}/{factor_name}_{cfg.timepoint}.csv")
    factor_standarize(f.iloc[:n, :], cfg, factor_name, name, label)


def generate_history(cfg):
    add_dates = get_trade_date(cfg.start_date, cfg.end_date)
    df_all = pd.DataFrame()
    for d in add_dates:
        try:
            df_one = cfg.get_kindle_factor_oneday(cfg, d)
            print(df_all)
            print(df_one)
            df_all = df_all.append(df_one)
            print(f"{d} add success")
        except Exception as e:
            print(f"{d} add error {e}")
    for factor_name in df_all.columns:
        f = df_all[factor_name].unstack()
        f.to_csv(f"{cfg.factor_minue_dir}/{factor_name}_{cfg.timepoint}.csv")
    print("update success")
