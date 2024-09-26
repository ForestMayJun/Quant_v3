import pandas as pd
import numpy as np
import pywt
from scipy import signal
from scipy import fftpack
from scipy.stats import kurtosis, skew
from statsmodels.tsa.seasonal import seasonal_decompose

# 截面标准化
def cs_standarize(index_name):
    def operator(x):
        if len(x)!=len(index_name):
            return np.zeros(len(x))
        x0 = pd.Series(x)
        x0.index = index_name
        x0 = x0.unstack()
        return (x0.sub(x0.mean(axis=1), axis=0).div(x0.std(axis=1), axis=0)).T.unstack().reindex(index=index_name).values
    return operator

# unstack之后横轴和纵轴对不上

# 截面rank
def cs_rank(index_name):
    def operator(x):
        if len(x)!=len(index_name):
            return np.zeros(len(x))
        x0 = pd.Series(x)
        x0.index = index_name
        x0 = x0.unstack()
        x0 = x0.rank(axis=1).div(x0.notnull().sum(axis=1), axis=0)
        return x0.rank(axis=1).T.unstack().reindex(index=index_name).values
    return operator


# 截面幂次
def cs_power(n):
    def operator(x):
        if n < 1:
            return np.sign(x) * (np.abs(x) ** n)
        else:
            return x**n
    return operator

# 截面求和
def cs_add(x, y):
    return x + y

# 截面相减
def cs_sub(x, y):
    return x - y

# 截面相除
def cs_div(x, y):
    return x / ( y + 10**-10)

# 截面相乘
def cs_mul(x, y):
    return x * y


def cs_cap(index_name, max_value):
    def operator(x):
        if len(x)!=len(index_name):
            return np.zeros(len(x))
        x0 = pd.Series(x)
        x0.index = index_name
        x0 = x0.unstack()
        x0 = x0.sub(x0.mean(axis=1), axis=0).div(x0.std(axis=1), axis=0)
        x0[x0>max_value] = max_value
        x0[x0<-max_value] = -max_value
        return x0.T.unstack().reindex(index=index_name).values
    return operator

def ts_ma(index_name, n):
    def operator(x):
        if len(x)!=len(index_name):
            return x
        x0 = pd.Series(x)
        x0.index = index_name
        x0 = x0.unstack().sort_index()
        return x0.apply(lambda x:x.ewm(n).mean(), axis=0).T.unstack().reindex(index=index_name).values
    return operator

def ts_ema(index_name, n):
    def operator(x):
        if len(x)!=len(index_name):
            return x
        x0 = pd.Series(x)
        x0.index = index_name
        x0 = x0.unstack().sort_index()
        return x0.rolling(n, min_periods=1).mean().T.unstack().reindex(index=index_name).values
    return operator

def ts_delay(index_name, n):
    def operator(x):
        if len(x)!=len(index_name):
            return np.zeros(len(x))
        x0 = pd.Series(x)
        x0.index = index_name
        x0 = x0.unstack().sort_index().fillna(0)
        return x0.shift(n).bfill().T.unstack().reindex(index=index_name).values
    return operator

def ts_max(index_name, n):
    def operator(x):
        if len(x)!=len(index_name):
            return np.zeros(len(x))
        x0 = pd.Series(x)
        x0.index = index_name
        x0 = x0.unstack().sort_index()
        return x0.rolling(n, min_periods=1).max().T.unstack().reindex(index=index_name).values
    return operator

def ts_min(index_name, n):
    def operator(x):
        if len(x)!=len(index_name):
            return np.zeros(len(x))
        x0 = pd.Series(x)
        x0.index = index_name
        x0 = x0.unstack().sort_index()
        return x0.rolling(n, min_periods=1).min().T.unstack().reindex(index=index_name).values
    return operator

def ts_std(index_name, n):
    def operator(x):
        if len(x)!=len(index_name):
            return np.zeros(len(x))
        x0 = pd.Series(x)
        x0.index = index_name
        x0 = x0.unstack().sort_index()
        return x0.rolling(n, min_periods=1).std().T.unstack().reindex(index=index_name).values
    return operator

def ts_corr(index_name, n):
    def operator(x, y):
        if len(x)!=len(index_name):
            return np.zeros(len(x))
        x0 = pd.Series(x)
        x0.index = index_name
        x0 = x0.unstack().sort_index()
        y0 = pd.Series(y)
        y0.index = index_name
        y0 = y0.unstack().sort_index()
        corr0 = x0.corrwith(y0).rolling(n, min_periods=1).mean()
        x0[x0!=1] = 1
        x0 = x0.mul(corr0, axis=0)
        return x0.T.unstack().reindex(index=index_name).values
    return operator

# -----------------------------------------------------------------------*****************--------------------------------------------------------------
# -----------------------------------------------------------------------modified operator--------------------------------------------------------------

# shift(1)都删去
# 分清楚到底是df还是series

# 时序变化率 check
def ts_change_rate(index_name, n):
    def operator(x):
        if len(x) != len(index_name):
            return np.zeros(len(x))
        x0 = pd.Series(x)
        x0.index = index_name
        x0 = x0.unstack().sort_index.fillna(0)
        change_rate = (x0 - x0.shift(n)) / (x0.shift(n) + 0.000001)
        return change_rate.fillna(0).T.unstack().reindex(index=index_name).values #pct为一天的change，shift(n)为n天之前的数据
    return operator

# 分桶 check
def cs_quantile_bin(index_name, q):
    def operator(x):
        if len(x) != len(index_name):
            return np.zeros(len(index_name))
        x0 = pd.Series(x)
        x0.index = index_name
        x0 = x0.unstack().sort_index()
        quantile_bins = x0.apply(lambda s: pd.qcut(s, q, labels=False, duplicates='drop').fillna(-1), raw=False, axis=1) #-1表示无效的分箱
        return quantile_bins.T.unstack().reindex(index=index_name).values
    return operator

# 截面对数变换 check
def cs_log_transform(index_name):
    def operator(x):
        if len(x)!=len(index_name):
            return np.zeros(len(x))
        return np.log(x+1)
    return operator
 
# 截面二元化 check
def cs_binarize(threshold):
    def operator(x):
        return pd.Series(np.where(x > threshold, 1, 0))
    return operator

# 截面高斯平滑 check
def cs_gaussian_transform(index_name):
    def operator(x):
        if len(x) != len(index_name):
            return np.zeros(len(index_name))
        x0 = pd.Series(x)
        x0.index = index_name
        x0 = x0.unstack().sort_index()
        mean_vals = x0.expanding(min_periods=1).mean() # 可以换成n，也可以换成天数
        std_vals = x0.expanding(min_periods=1).std()  # rolling
        gaussian_transformed = np.exp(-((x0 - mean_vals) ** 2) / (2 * std_vals ** 2))
        return gaussian_transformed.fillna(0).T.unstack().reindex(index=index_name).values
    return operator

# 时序峰度 check
# def ts_kurtosis(index_name, n):
#     def operator(x):
#         if len(x) != len(index_name):
#             return np.zeros(len(index_name))
#         x0 = pd.Series(x)
#         x0.index = index_name
#         x0 = x0.unstack().sort_index()
#         kurt_values = x0.rolling(window=n, min_periods=1).apply(lambda s: kurtosis(s, fisher=True, bias=False), raw=False)
#         return kurt_values.fillna(0).T.unstack().reindex(index=index_name).values
#     return operator

# 时序偏度 check
# def ts_skewness(index_name, n):
#     def operator(x):
#         if len(x) != len(index_name):
#             return np.zeros(len(index_name))
#         x0 = pd.Series(x)
#         x0.index = index_name
#         x0 = x0.unstack().sort_index()
#         skew_values = x0.rolling(window=n, min_periods=1).apply(lambda s: skew(s, bias=False), raw=False)
#         return skew_values.fillna(0).T.unstack().reindex(index=index_name).values
#     return operator

# 截面峰度 check
def cs_kurtosis(index_name, n):
    def operator(x):
        if len(x) != len(index_name):
            return np.zeros(len(index_name))
        x0 = pd.Series(x)
        x0.index = index_name
        x0 = x0.unstack().sort_index()
        # 对截面赋予同一个峰度值
        row_kurts = x0.apply(lambda s: kurtosis(s.dropna(), fisher=True, bias=False), raw=False, axis=1)
        kurt_values = pd.DataFrame(np.tile(row_kurts, (x0.shape[1], 1)).T, index=x0.index, columns=x0.columns)
        # kurt_values = x0.rolling(window=n, min_periods=1).apply(lambda s: kurtosis(s, fisher=True, bias=False), raw=False)
        return kurt_values.fillna(0).T.unstack().reindex(index=index_name).values
    return operator

# 截面偏度 check
def cs_skewness(index_name, n):
    def operator(x):
        if len(x) != len(index_name):
            return np.zeros(len(index_name))
        x0 = pd.Series(x)
        x0.index = index_name
        x0 = x0.unstack().sort_index()
        # 对截面赋予同一个偏度值
        row_skews = x0.apply(lambda row: skew(row.dropna(), bias=False), axis=1)
        skew_values = pd.DataFrame(np.tile(row_skews, (x0.shape[1], 1)).T, index=x0.index, columns=x0.columns)
        # skew_values = x0.rolling(window=n, min_periods=1).apply(lambda s: skew(s, bias=False), raw=False)
        return skew_values.fillna(0).T.unstack().reindex(index=index_name).values
    return operator

# 截面多项式拟合 
def cs_polynomial_features(index_name, degree=2):
    def operator(x):
        if len(x) != len(index_name):
            return np.zeros(len(index_name))

        x0 = pd.Series(x)
        x0.index = index_name
        x0 = x0.unstack().sort_index()
        poly_vals = []
        # 对截面拟合
        for idx in x0.index:
            y = x0.loc[idx, :]
            nan_pos = y.isna()
            y.dropna(inplace=True)
            x = np.arange(len(y))
            poly = np.polynomial.Polynomial.fit(x, y, deg=min(degree, len(y)-1))
            fitted_val = poly(x)
            full_fitted_val = np.full(x0.shape[1], np.nan)
            full_fitted_val[~nan_pos] = fitted_val
            poly_vals.append(full_fitted_val)

        poly_val_df = pd.DataFrame(poly_vals, index=x0.index, columns=x0.columns)
        return poly_val_df.fillna(0).T.unstack().reindex(index=index_name).values
    return operator

# 时序卷积 check
def ts_convolution(index_name, kernel_size=3, kernel_type='average'):
    def create_kernel(kernel_size, kernel_type):
        if kernel_type == 'average':
            return np.ones(kernel_size) / kernel_size
        elif kernel_type == 'gaussian':
            return np.exp(-np.linspace(-(kernel_size - 1)/2, (kernel_size - 1)/2, kernel_size)**2 / 2) / (np.sqrt(2 * np.pi))
        else:
            raise ValueError("unknown kernel type")
        
    def operator(x):
        if len(x) != len(index_name):
            return np.zeros(len(index_name))
        kernel = create_kernel(kernel_size, kernel_type)
        x0 = pd.Series(x)
        x0.index = index_name
        x0 = x0.unstack().sort_index()
        convolved = np.array([np.convolve(x0.iloc[:, i], kernel, mode='same') for i in range(x0.shape[1])])
        convolved_df = pd.DataFrame(convolved.T, index=x0.index, columns=x0.columns)
        return convolved_df.T.unstack().reindex(index=index_name).values
    return operator

# 时序傅里叶变换 check
def ts_fourier_transform(index_name):
    def operator(x):
        if len(x) != len(index_name):
            return np.zeros(len(index_name))
        x0 = pd.Series(x)
        x0.index = index_name
        x0 = x0.unstack().sort_index()
        fourier_transformed = x0.apply(lambda col: np.fft.fft(col).real, axis=0)
        return fourier_transformed.fillna(0).T.unstack().reindex(index=index_name).values
    return operator

# 截面傅里叶变换 check
def cs_fourier_transform(index_name):
    def operator(x):
        if len(x) != len(index_name):
            return np.zeros(len(index_name))
        x0 = pd.Series(x)
        x0.index = index_name
        x0 = x0.unstack().sort_index()
        fourier_transformed = x0.fillna(0).apply(lambda row: np.fft.fft(row).real, axis=1)
        dfs = []
        for col in fourier_transformed.index:
            df = pd.DataFrame(fourier_transformed[col])
            dfs.append(df)

        fourier_transformed_df = pd.concat(dfs, axis=1)

        res_df = fourier_transformed_df.T
        res_df.index = x0.index 
        res_df.columns = x0.columns
        return res_df.fillna(0).T.unstack().reindex(index=index_name).values
    return operator

