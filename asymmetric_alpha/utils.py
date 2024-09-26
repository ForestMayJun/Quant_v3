'''
function:[]
'''

import pandas as pd
import numpy as np
import scipy.integrate as integrate
import scipy.stats as stats
from tqdm import tqdm
from scipy.stats import gaussian_kde
from scipy.special import comb
from scipy.special import roots_chebyt
    
def skewness(series):
    '''self.X序列的偏度: 标准化后的三阶中心矩'''
    z_socre = (series - series.mean()) / (series.std() + 1e-8)
    return (z_socre ** 3).mean()

def _gauss_kernel(x):
    '''Gauss核函数 标准正态分布的密度函数'''
    return 1 / np.power(2*np.pi, 1/2) * np.exp(-1/2*np.power(x, 2))

def kernel_density(y, kernel=_gauss_kernel):
    '''
    核密度估计法,基于数据y的分布密度函数 Parameters:  
    y:计算区间内的的超额收益,真实统计值
    -> function(float)
    '''
    h = 1.06 * y.std() * np.power(len(y), -1/5) # 基于Silverman的经验法则的带宽估计

    def f(x):
        new_series = _gauss_kernel(1/h * (y - x))
        return 1/(len(y)*h) * new_series.sum()

    return f


def e_phi_series(series):
    k = series.quantile(0.9)
    density_func = kernel_density(series)
    left_inte, _ = integrate.quad(density_func, -np.inf, -k)
    right_inte, _ = integrate.quad(density_func, k, np.inf)

    return right_inte - left_inte

def s_phi_series(series, k=0.9):
    k_value = series.quantile(k)
    sign = np.sign(e_phi_series(series))

    def diff_func(f_1, f_2):
        '''积分辅助函数'''
        def f(x):
            return np.power(np.power(f_1(x), 1/2) - np.power(f_2(x), 1/2), 2)
        return f

    f_1 = gaussian_kde(series)
    f_2 = gaussian_kde(-series + 2*series.mean())
    diff_f = diff_func(f_1, f_2)

    return sign * 1/2 * (integrate.quad(diff_f, -np.inf, -k)[0] + integrate.quad(diff_f, k, np.inf)[0])

def s_phi_series_version2(series, k=0.6):

    sign = np.sign(series.skew())

    def diff_func(f_1, f_2):
        '''积分辅助函数'''
        def f(x):
            return np.power(np.power(f_1(x), 1/2) - np.power(f_2(x), 1/2), 2)
        return f
    
    series = np.array(series)
    epsilon = 1e-10
    series += epsilon * np.random.randn(len(series))

    f_1 = gaussian_kde(series)
    f_2 = gaussian_kde(-series + 2*series.mean())
    diff_f = diff_func(f_1, f_2)
    diff_series = diff_f(series)
    diff_series = pd.Series(diff_series)
    k_value = diff_series.quantile(k)

    return sign * diff_series[diff_series > k_value].mean()


def asym_p_series(series):
    # series = np.array(series)
    # density_func = clt_density(series)
    # acc_func = clt_cdf(series)
    # density_series = density_func(series) # 收益率序列的密度函数 f(r)
    # acc_series = acc_func(series)
    # density_series = stats.norm.pdf(series, loc=np.mean(series), scale=np.std(series))
    # acc_series = stats.norm.cdf(series, loc=np.mean(series), scale=np.std(series))
    series = np.array(series)
    epsilon = 1e-10
    series += epsilon * np.random.randn(len(series))

    kde = stats.gaussian_kde(series)
    density_series = kde(series)
    acc_series = stats.rankdata(series) / len(series)
    # acc_series = np.array([integrate.romberg(kde, -10, i, tol=1e-2)[0] for i in series])

    return stats.spearmanr(density_series, acc_series)[0]

def cVaR_series(series, c_level=0.9, is_low=True):
    '''
    CVaR:分布的尾部积分, Parameters:
    c_level:p值,默认0.9
    is_low:返回的是否是左侧CVaR,=False时返回右侧,默认True
    '''
    right_VaR = series.quantile(c_level)
    left_VaR = series.quantile(1- c_level)

    if is_low:
        X_low = pd.Series(series[series < left_VaR])
        low_mean = X_low.mean() if len(X_low) > 0 else 0
        return low_mean
    else:
        X_high = pd.Series(series[series > right_VaR])
        high_mean = X_high.mean() if len(X_high) > 0 else 0
        return -high_mean
    

def rolling(data, f, window=60):
    '''
    pd.series -> pd.series,
    每个元素是rolling前window窗口 -> 
    '''
    res = pd.Series(index=data.index) # 储存数据 iloc[i]为i-window - i时间段的f的值，f:pd.series -> float
    for i in range(window, len(data)):
        res.iloc[i] = f(data.iloc[i-window:i])

    return res.fillna(0)

def clt_density(y):
    '''中心极限定理下的估计密度函数'''
    def f(x):
        return stats.norm.pdf(x, loc=np.mean(y), scale=np.std(y))
    return f

def clt_cdf(y):
    '''中心极限定理下的估计分布函数'''
    def f(x):
        return stats.norm.cdf(x, loc=np.mean(y), scale=np.std(y))
    return f

def skewness_power(x, window=30, decay_rate=0.95):
    '''带加权值的skew'''
    y = (x - x.mean()) / (x.std() + 1e-10)
    weight = np.array([np.power(decay_rate, window - i) for i in range(window)])
    return np.dot(weight, y.values) / np.sum(weight)

def skew_scale(series:pd.Series, scale_por=0.9):
    '''去掉极值后的skew, scale_por:剩下数据数量/原始数据数量'''
    low =  series.quantile((1-scale_por)/2)
    high = series.quantile(0.5 + scale_por/2)

    series = series[(series >= low) & (series <= high)]

    return series.skew()

def l_moments(series:pd.Series):
    sorted_series = series.sort_values()
    n = len(series)
    weight2 = np.array([2*i - n - 1 for i in range(1, n+1)])
    weight3= np.array([comb(i-1, 2) - 2*(i-1)*(n-1) + comb(n-i, 2) for i in range(1, n+1)])

    l_2 = 1 / (n*(n-1)) * np.dot(weight2, sorted_series)
    l_3 = 1 / (3*comb(n, 3)) * np.dot(weight3, sorted_series)

    return l_3 / l_2

def asym_p_version2(series):
    low_bound = -5
    n_samples = 40
    left, right = -0.15, 0.15
    samples = np.arange(left, right, (right - left)/n_samples)

    epslion = 1e-10
    series += epslion * np.random.randn(len(series))
    f = gaussian_kde(series)
    f_array = f(samples)

    def g(f, a, b):
        def lambda_func(x):
            return (b-a) / 2 * f((b-a)*x/2 + (b+a)/2)
        
        return lambda_func
    
    nodes, weights = roots_chebyt(n_samples)
    F_array = np.array([np.sum(np.dot(weights, g(f, low_bound, sample)(nodes))) for sample in samples])

    return np.corrcoef(f_array, F_array)[0][1]
    

    
if __name__ == '__main__':
    X = pd.DataFrame()
    X['b'] = pd.Series(np.exp(2 * np.random.rand(100) - 1) -1)
    X['a'] = pd.Series(0.2*np.random.randn(100) - 0.1)
    # X['c'] = pd.Series([integrate.quad(stats.gaussian_kde(X['a']), -10, i, epsabs=1e-2)[0] for i in X['a'].values])
    # X['d'] = pd.Series([integrate.quad(stats.gaussian_kde(X['a']), -100, i, epsabs=1e-2)[0] for i in X['a'].values])
    # X['e'] = pd.Series(stats.rankdata(X['a']) / len(X['a']))
    # X['f'] = pd.Series(stats.gaussian_kde(X['a'])(X['a'].values))
    # asymetric = Asymmetric(X)
    print(X)
    # print('skewness:')
    # print(asymetric.skewness())
    # print('e_phi:')
    # print(X.rolling(30).apply(e_phi_series))
    # print('s_phi')
    # print(asymetric.s_phi())
    # print('asym_p')
    print(X.rolling(30).apply(skew_scale))
    # print(X.rolling(30).apply(s_phi_series_version2))
    # print(X.rolling(30).apply(l_moments))
    print(X.rolling(30).apply(asym_p_version2))
    
    # print('cVaR:')
    # print(asymetric.cVaR())
    # print(X.apply(lambda x: rolling(x, e_phi_series)))