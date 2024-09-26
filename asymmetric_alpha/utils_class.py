


import pandas as pd
import numpy as np
import scipy.integrate as integrate
import scipy.stats as stats
from tqdm import tqdm

class Asymmetric:
    '''
    蕴含计算收益率序列非对称的一些方法 Parameter:
    X:pd.Series 收益率数据
    Methods:
    skewness;e_phi;s_phi;asym_p;cVaR
    '''

    def __init__(self, X) -> None:
        self.X = X

    def skewness(self):
        '''self.X序列的偏度: 标准化后的三阶中心矩'''
        z_socre = (self.X - self.X.mean()) / (self.X.std() + 1e-8)
        return (z_socre ** 3).mean()

    def _gauss_kernel(self, x):
        '''Gauss核函数 标准正态分布的密度函数'''
        return 1 / np.power(2*np.pi, 1/2) * np.exp(-1/2*np.power(x, 2))

    def kernel_density(self, y, kernel=_gauss_kernel):
        '''
        核密度估计法,基于数据y的分布密度函数 Parameters:  
        y:计算区间内的的超额收益,真实统计值
        -> function(float)
        '''
        h = 1.06 * y.std() * np.power(len(y), -1/5) # 基于Silverman的经验法则的带宽估计

        def f(x):
            new_series = self._gauss_kernel(1/h * (y - x))
            return 1/(len(y)*h) * new_series.sum()

        return f

    def e_phi(self, k=1):
        '''
        从异尾概率出发，从左右概率作差体现分别的非对称性  Parameters:
        self.X : 收益率序列
        k: 左右尾的阈值
        '''
        def e_phi_series(series):
            k = series.quantile(0.9)
            density_func = self.kernel_density(series)
            left_inte, _ = integrate.quad(density_func, -np.inf, -k)
            right_inte, _ = integrate.quad(density_func, k, np.inf)

            return right_inte - left_inte
        
        return self.X.apply(e_phi_series)
    

    def s_phi(self, k=1):
        '''
        相较于e_phi改进
        '''
        def s_phi_series(series):
            k = series.quantile(0.9)
            sign = np.sign(self.e_phi(k))

            def diff_func(f_1, f_2):
                '''积分辅助函数'''
                def f(x):
                    return np.power(np.power(f_1(x), 1/2) - np.power(f_2(x), 1/2), 2)
                return f

            f_1 = self.kernel_density(series)
            f_2 = self.kernel_density(-series + 2*series.mean())
            diff_f = diff_func(f_1, f_2)

            return sign * 1/2 * (integrate.quad(diff_f, -np.inf, -k)[0] + integrate.quad(diff_f, k, np.inf)[0])
        
        return self.X.apply(s_phi_series).iloc[0]

    def asym_p(self):
        '''Asym_p因子 反映非对称性'''
        def asym_p_series(series):
            density_func = self.kernel_density(series)
            density_series = pd.Series([density_func(series.values[i]) for i in range(len(series))], index=series.index) # 收益率序列的密度函数 f(r)
            acc_series = pd.Series([integrate.quad(density_func, -np.inf, series.iloc[i])[0] for i in range(len(series))], index=series.index)

            return - density_series.corr(acc_series) if density_series.std() > 0 else 0
        
        return self.X.apply(asym_p_series)
    
    def cVaR(self,c_level = 0.95):
        '''
        CVaR因子,计算给定置信水平下的超出部分的均值收益/损失,置信水平默认0.95
        '''
        def cVaR_series(series):
            right_VaR = series.quantile(c_level)
            left_VaR = series.quantile(1- c_level)

            X_low = pd.Series(series[series < left_VaR])
            X_high = pd.Series(series[series > right_VaR])

            low_mean = X_low.mean() if len(X_low) > 0 else 0
            high_mean = X_high.mean() if len(X_high) > 0 else 0

            return low_mean, high_mean
        
        return self.X.apply(cVaR_series)