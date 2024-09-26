import numpy as np
import pandas as pd
from Data_origin.utils import median_fill
import warnings
warnings.filterwarnings("ignore")

def generate_onetime(cfg):
    cfg.update_to_newdate(cfg)

def out(f, cfg, factor_name, name, label='', n=100000):
    f = f.sort_index(ascending=False)
    if n > 20:
        f.to_csv(f"{cfg.factor_minue_dir}/{factor_name}.csv")
    factor_standarize(f.iloc[:n, :], cfg, factor_name, name, label)
    print(f"{name} success")

def factor_standarize(df, cfg, name1, name2, label=''):
    x = median_fill(df)
    x.to_csv(f"{cfg.factor_dir}/{name2}.csv"+label)
    print(f"factor {name1}, {name2} standerdize success")

def factor_evaluate(sig, ret, start_date, end_date, period=252):
    sign = sig.reindex(index=ret.index, columns=ret.columns).fillna(0).values
    retn = ret.fillna(0).values
    td = (ret.index >= start_date)*(ret.index <= end_date)
    print(np.sum(td))
    facret = []
    selfcorr = []
    for i in range(len(sign)):
        X = sign[i]
        y = retn[i]
        if X.max() > X.min():
            p = np.corrcoef(X, y)[0, 1]
            if not np.isnan(p):
                facret = np.append(facret, p)
                if i > 0:
                    p2 = np.corrcoef(sign[i], sign[i-1])[0, 1]
                    if not np.isnan(p2):
                        selfcorr = np.append(selfcorr, p2)
            else:
                facret = np.append(facret, 0)
        else:
            facret = np.append(facret, 0)
    IR = facret[td].mean() / facret[td].std()*period**0.5
    print("IC {:.3f}, IR {:.2f}, self_corr {:.2f}, last_week IC:{:.3f}, last_month IC:{:.3f}, last_season IC:{:.3f}".format(
        facret[td].mean(), IR, selfcorr.mean(), facret[td][-5:].mean(), facret[td][-20:].mean(), facret[td][-60:].mean()))
    a = pd.Series(facret, index=ret.index)
    #print(a)
    return a 


def toweight(m, T, u):
    tmp = m.copy()
    tmp2 = (tmp.fillna(0).rolling(T).mean(
    ) / tmp.fillna(0).rolling(T).std()*(250**0.5))  # .apply(lambda around(a,2),
    tmp3 = (tmp.fillna(0).rolling(20).mean(
    ) / tmp.fillna(0).rolling(T).std()*(250**0.5))  # .apply(lambda around(a,2))
    w = (0.3 * tmp3+tmp2).values.reshape(len(m), 1)*np.ones([1, len(u)])
    x = pd.DataFrame(w, index=m.index, columns=u).ffill()
    #print(x)
    x[x > 4] = 4
    x[x < -4] = -4
    return xs

def toweight2(m, T, u):
    tmp = m.copy()
    tmp2 = (tmp.fillna(0).rolling(T).mean(
    ) / tmp.fillna(0).rolling(T).std()*(250**0.5))  # .apply(lambda around(a,2),
    tmp3 = (tmp.fillna(0).rolling(5).mean(
    ) / tmp.fillna(0).rolling(T).std()*(250**0.5))  # .apply(lambda around(a,2))
    w = (0.3 * tmp3+tmp2).values.reshape(len(m), 1)*np.ones([1, len(u)])
    x = pd.DataFrame(w, index=m.index, columns=u).ffill()
    x[x > 4] = 4
    x[x < -4] = -4
    return x
