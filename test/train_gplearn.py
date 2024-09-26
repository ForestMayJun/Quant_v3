import pandas as pd
from utils import get_trade_date
from glob import glob
import sys
import os
from DataDaily import DataDaily
from DataMinute_vwap import DataMinute
import numpy as np
from tool_new import *
from gplearn.functions import make_function
from gplearn.fitness import make_fitness
from gplearn.genetic import SymbolicTransformer
from gplearn.fitness import _Fitness
import pyarrow.parquet as pq
import pyarrow as pa
import warnings
warnings.filterwarnings('ignore')


# script_dic = os.path.dirname(os.path.abspath(__file__))
# date_path = os.path.join(script_dic, 'trade_date_long.csv')

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
# df = pd.read_csv(date_path)
#------------------------------------------------------------------------------------------

#read data
model_path = './model_gplearn/'
# trade_month = get_trade_date(20180101, 20200701)[:-1] 
trade_month = get_trade_date(20180101, 20190101)[:-1]
trade_month = list(set([str(tmp)[:6] for tmp in trade_month]))
trade_month = list(set([tmp[:5]+"1-6" if tmp[-1]<'7' else tmp[:5]+"7-12" for tmp in trade_month]))
trade_month.sort()
print("start reading data")
print(trade_month)
file_all = []
for m in trade_month:
    file_all.extend(glob("/mnt/datadisk2/data_train/data_train/data_merge_v2/"+m+"/*.parquet"))
print(len(file_all))
data = [pd.read_parquet(tmp) for tmp in file_all[:50]] # 先用少量的数据
# data = [pd.read_parquet(tmp) for tmp in file_all[:2]]


# data = [pd.read_parquet(file_all[0]), pd.read_parquet(file_all[1])]
data = pd.concat(data, ignore_index=True)
data = data.astype({c: np.float32 for c, t in data.dtypes.items() if t == np.float64})
data = data.set_index(['TradingDay', 'InstrumentID'], drop=False)
data.index.names = ['', '']
print("start preprocessing data")
data_minute = DataMinute(check_data=False)
data_daily = DataDaily(check_data=False)
timepoint_2='09:30'
mask = (((data_minute.get_price(timepoint_2)*data_daily.adjfactor).shift(-1)/(data_daily.limit_price *
        data_daily.adjfactor).shift(-1) - 1).abs() > 0.001).applymap(lambda x: 0 if x else 1)
mask_2 = (((data_minute.get_price(timepoint_2)*data_daily.adjfactor).shift(-1)/(data_daily.stop_price *
            data_daily.adjfactor).shift(-1) - 1).abs() > 0.001).applymap(lambda x: 0 if x else 1)
mask += mask_2
mask_3 = (((data_minute.get_price(timepoint_2)*data_daily.adjfactor).shift(-1) /
            (data_daily.close*data_daily.adjfactor) - 1).abs() <= 0.098).applymap(lambda x: 0 if x else 1)
mask_4 = ((data_daily.close > 0).rolling(15).sum()
            > 5).applymap(lambda x: 0 if x else 1)
mask_3 = mask_3.unstack().reset_index()
mask_4 = mask_4.unstack().reset_index()
mask = mask.unstack().reset_index()
mask.columns = ['InstrumentID', 'TradingDay', 'mask']
mask_3.columns = ['InstrumentID', 'TradingDay', 'mask2']
mask_4.columns = ['InstrumentID', 'TradingDay', 'mask3']
data = pd.merge(mask, data, on=[
                'InstrumentID', 'TradingDay'], right_index=False, left_index=False, how='right')
data = pd.merge(mask_3, data, on=[
                'InstrumentID', 'TradingDay'], right_index=False, left_index=False, how='right')
data = pd.merge(mask_4, data, on=[
                'InstrumentID', 'TradingDay'], right_index=False, left_index=False, how='right')
print(data.memory_usage().sum() / 1024**2)
data.replace(-np.inf, 0, inplace=True)
data.replace(np.inf, 0, inplace=True)
label_name = 'label_1_5d_max_30mt'
data_train = data.loc[data[label_name].notnull()]
data_train = data_train[(data_train["mask3"]+data_train["mask2"]+data_train["mask"])==0]
data_train = data_train.set_index(['TradingDay', 'InstrumentID'])
data_tmp = data_train.iloc[:10000,:].dropna()
data_tmp = data_tmp.sub(data_tmp.mean()).div(data_tmp.std()+0.0001)
duplicates = (data_tmp.T.duplicated())|((data_tmp==0).sum()==len(data_tmp))
duplicates[['factor' not in tmp for tmp in duplicates.index]] = False
print("drop duplicates factor")
print(data_train.columns[duplicates.values])
data_train = data_train.loc[:, ~duplicates.values]

# 创建 gplearn 可用的函数 arity是输入参数个数
index_name = data_train.index
gp_operator1 = make_function(function=cs_standarize(index_name), name='cs_standardize', arity=1)
gp_operator2 = make_function(function=cs_rank(index_name), name='cs_rank', arity=1)
gp_operator3 = make_function(function=cs_power(0.5), name='cs_power_05', arity=1)
gp_operator4 = make_function(function=cs_power(2), name='cs_power_2', arity=1)
gp_operator5 = make_function(function=cs_power(3), name='cs_power_3', arity=1)
gp_operator6 = make_function(function=cs_power(4), name='cs_power_4', arity=1)
gp_operator7 = make_function(function=cs_add, name='cs_add', arity=2)
gp_operator8 = make_function(function=cs_sub, name='cs_sub', arity=2)
gp_operator9 = make_function(function=cs_div, name='cs_div', arity=2)
gp_operator10 = make_function(function=cs_mul, name='cs_mul', arity=2)
# gp_operator11 = make_function(function=cs_cap(index_name, 5), name='cs_cap_5', arity=1)
# gp_operator12 = make_function(function=cs_cap(index_name, 2), name='cs_cap_2', arity=1)
# gp_operator13 = make_function(function=cs_cap(index_name, 3), name='cs_cap_3', arity=1)
# gp_operator14 = make_function(function=ts_ma(index_name, 2), name='ts_ma2', arity=1)
# gp_operator15 = make_function(function=ts_ma(index_name, 5), name='ts_ma5', arity=1)
# gp_operator16 = make_function(function=ts_ma(index_name, 20), name='ts_ma20', arity=1)
# gp_operator17 = make_function(function=ts_ma(index_name, 60), name='ts_ma60', arity=1)
# gp_operator18 = make_function(function=ts_ma(index_name, 120), name='ts_ma120', arity=1)
# gp_operator19 = make_function(function=ts_ema(index_name, 2), name='ts_ema2', arity=1)
# gp_operator20 = make_function(function=ts_ema(index_name, 5), name='ts_ema5', arity=1)
# gp_operator21 = make_function(function=ts_ema(index_name, 20), name='ts_ema20', arity=1)
# gp_operator22 = make_function(function=ts_ema(index_name, 60), name='ts_ema60', arity=1)
# gp_operator23 = make_function(function=ts_ema(index_name, 120), name='ts_ema120', arity=1)
# gp_operator24 = make_function(function=ts_delay(index_name, 2), name='ts_delay2', arity=1)
# gp_operator25 = make_function(function=ts_delay(index_name, 5), name='ts_delay5', arity=1)
# gp_operator26 = make_function(function=ts_delay(index_name, 20), name='ts_delay20', arity=1)
# gp_operator27 = make_function(function=ts_max(index_name, 2), name='ts_max2', arity=1)
# gp_operator28 = make_function(function=ts_max(index_name, 5), name='ts_max5', arity=1)
# gp_operator29 = make_function(function=ts_max(index_name, 20), name='ts_max20', arity=1)
# gp_operator30 = make_function(function=ts_max(index_name, 60), name='ts_max60', arity=1)
# gp_operator31 = make_function(function=ts_min(index_name, 2), name='ts_min2', arity=1)
# gp_operator32 = make_function(function=ts_min(index_name, 5), name='ts_min5', arity=1)
# gp_operator33 = make_function(function=ts_min(index_name, 20), name='ts_min20', arity=1)
# gp_operator34 = make_function(function=ts_min(index_name, 60), name='ts_min60', arity=1)
# gp_operator35 = make_function(function=ts_std(index_name, 5), name='ts_std5', arity=1)
# gp_operator36 = make_function(function=ts_std(index_name, 20), name='ts_std20', arity=1)
# gp_operator37 = make_function(function=ts_std(index_name, 60), name='ts_std60', arity=1)
# gp_operator38 = make_function(function=ts_corr(index_name, 5), name='ts_corr5', arity=2)
# gp_operator39 = make_function(function=ts_corr(index_name, 20), name='ts_corr20', arity=2)
# gp_operator40 = make_function(function=ts_corr(index_name, 60), name='ts_corr60', arity=2)
# gp_operator41 = make_function(function=ts_change_rate(index_name, 5), name='ts_chr5', arity=1)
# gp_operator42 = make_function(function=ts_change_rate(index_name, 20), name='ts_chr20', arity=1)
# gp_operator43 = make_function(function=ts_change_rate(index_name, 60), name='ts_chr60', arity=1)
# gp_operator44 = make_function(function=cs_quantile_bin(index_name, 5), name='cs_qcut5', arity=1)
# gp_operator45 = make_function(function=cs_quantile_bin(index_name, 10), name='cs_qcut10', arity=1)
# gp_operator46 = make_function(function=cs_quantile_bin(index_name, 20), name='cs_qcut20', arity=1)
# gp_operator47 = make_function(function=cs_log_transform(index_name), name='cs_log', arity=1)
# gp_operator48 = make_function(function=cs_binarize(0.5), name='cs_binary', arity=1)
# gp_operator49 = make_function(function=cs_polynomial_features(index_name, 2), name='cs_poly2', arity=1)
# gp_operator50 = make_function(function=cs_polynomial_features(index_name, 3), name='cs_poly3', arity=1)
# gp_operator51 = make_function(function=cs_polynomial_features(index_name, 4), name='cs_poly4', arity=1)
# gp_operator52 = make_function(function=cs_polynomial_features(index_name, 5), name='cs_poly5', arity=1)
# gp_operator49 = make_function(function=cs_kurtosis(index_name), name='cs_kurt', arity=1)
# gp_operator50 = make_function(function=cs_skewness(index_name), name='cs_skew', arity=1)
# gp_operator55 = make_function(function=ts_convolution(index_name, 3, 'average'), name='ts_conv3average', arity=1)
# gp_operator56 = make_function(function=ts_convolution(index_name, 5, 'average'), name='ts_conv5average', arity=1)
# gp_operator57 = make_function(function=ts_convolution(index_name, 3, 'average'), name='ts_conv3gauss', arity=1)
# gp_operator58 = make_function(function=ts_convolution(index_name, 5, 'average'), name='ts_conv5gauss', arity=1)
# gp_operator59 = make_function(function=ts_fourier_transform(index_name), name='ts_ft', arity=1)
# gp_operator60 = make_function(function=cs_fourier_transform(index_name), name='cs_ft', arity=1)

def raw_fitness(y, y_pred):
    """计算适应度
    例如，使用ir
    """
    y0 = pd.Series(y)
    y0.index = index_name
    y0 = y0.unstack()
    y_pred0 = pd.Series(y_pred)
    y_pred0.index = index_name
    y_pred0 = y_pred0.unstack()
    ic = y0.corrwith(y_pred0, axis=1)
    # print(ic)
    # 保存一下ic和ir
    return ic.mean()/ic.std()*252**0.5, ic.mean() # scale到年

print("Start training")
#prepare data
# name_list = [[1,76], [76,110], [110, 122], [122,195], [195, 207], [207, 219], [219, 319], [1001,1054], [1054, 1111], [1111,1183], [1183, 1197]]
# for g in range(len(name_list)):
#     feature_list = [f'factor{i}' for i in range(name_list[g][0], name_list[g][1])]
#     feature_list = [tmp for tmp in feature_list if tmp in data_train.columns.values]
#     print(name_list[g])

cluster_feature_lists = []  # 用于存储所有聚类的特征列表

for cluster_label in range(1, 21):  # 从1.txt到20.txt
    with open(f'../train/cluster_res/{cluster_label}.txt', 'r') as file:
        features = [line.strip() for line in file.readlines()]
        cluster_feature_lists.append(features)

# 对每一类分别采用gplearn
trans_feature_count = 0
for feature_list, cnt in zip(cluster_feature_lists, range(1, len(cluster_feature_lists) + 1)):
    feature_list = [tmp for tmp in feature_list if tmp in data_train.columns.values]
    print(len(feature_list))
    X = data_train[feature_list].values
    y = data_train[label_name].values

    # 创建 SymbolicTransformer 或 SymbolicRegressor 实例
    gp = SymbolicTransformer(
        generations=4,           # 遗传编程的代数，即遗传算法迭代的次数
        population_size=50,     # 种群大小，即每代中个体（程序）的数量
        hall_of_fame=20,         # 精英策略中保留的顶级程序的数量
        n_components=min(len(feature_list), 10),          # 生成的新特征数量
        function_set= [eval(f"gp_operator{i+1}") for i in range(10)],  # 使用的函数集合，即算子集
        # function_set= [eval(f"gp_operator{i+1}") for i in range(5)],
        parsimony_coefficient=0.0001,  # 简约系数，用于控制程序复杂度对适应度的影响
        max_samples=1.0,          # 用于每个程序的子样本的最大比例如果要对截面处理 必须是1
        verbose=4,                # 日志输出的详细程度：0为无输出，1为每代输出
        # random_state=0,           # 随机种子
        n_jobs=16                 #cpu num
    )
    gp.fit(X, y)
    print("First fit")
    # programs = gp._programs
    # top_programs = programs[-1][:gp.n_components]
    transformed_X1 = gp.transform(X)
    best_programs1 = gp._best_programs
    print(transformed_X1.shape)
    trans_df1 = pd.DataFrame(transformed_X1)
    unique_columns1 = ~trans_df1.T.duplicated()
    trans_df1 = trans_df1.loc[:,~trans_df1.T.duplicated()]
    best_programs1 = [prog for prog, is_unique in zip(best_programs1, unique_columns1) if is_unique]
    trans_X1 = trans_df1.values
    print(trans_X1.shape)
    X = np.concatenate([X, trans_X1], axis=1)

    gp.fit(X, y)
    print("Second fit")
    transformed_X2 = gp.transform(X)
    best_programs2 = gp._best_programs
    trans_df2 = pd.DataFrame(transformed_X2)
    unique_columns2 = ~trans_df2.T.duplicated()
    trans_df2 = trans_df2.loc[:,~trans_df2.T.duplicated()]
    best_programs2 = [prog for prog, is_unique in zip(best_programs2, unique_columns2) if is_unique]
    trans_X2 = trans_df2.values 
    X = np.concatenate([X, trans_X2], axis=1)

    gp.fit(X, y)
    print('Third fit')
    transformed_X3 = gp.transform(X)
    best_programs3 = gp._best_programs
    trans_df3 = pd.DataFrame(transformed_X3)
    unique_columns3 = ~trans_df3.T.duplicated()
    trans_df3 = trans_df3.loc[:,~trans_df3.T.duplicated()]
    best_programs3 = [prog for prog, is_unique in zip(best_programs3, unique_columns3) if is_unique]
    trans_X3 = trans_df3.values
    X = np.concatenate([X, trans_X3], axis=1)

    new_feature_count = trans_X1.shape[1] + trans_X2.shape[1] + trans_X3.shape[1]
    fianl_feature_names = feature_list + [f'trans_{i}' for i in range(trans_feature_count + 1, trans_feature_count + new_feature_count + 1)]
    trans_feature_count += new_feature_count
    final_df = pd.DataFrame(X, columns=fianl_feature_names, index=data_train.index)
    final_df['label_1_5d_max_30mt'] = y
    final_df.reset_index(inplace=True)

    table = pa.Table.from_pandas(final_df)
    file_name = f'trans/group{cnt}.parquet'
    pq.write_table(table, file_name)
    

    # print('This is transformed X:')
    # print(transformed_X)
    #[[0.44054614 0.44054614 0.44054614 ... 0.1940809  0.1940809  0.1940809 ]
    # [0.43287666 0.43287666 0.43287666 ... 0.18738221 0.18738221 0.18738221]
    # [0.4374805  0.4374805  0.4374805  ... 0.19138919 0.19138919 0.19138919]
    # ...
    # [0.438919   0.438919   0.438919   ... 0.19264989 0.19264989 0.19264989]
    # [0.4413573  0.4413573  0.4413573  ... 0.19479626 0.19479626 0.19479626]
    # [0.43996151 0.43996151 0.43996151 ... 0.19356613 0.19356613 0.19356613]]

    trans_X = np.concatenate([trans_X1, trans_X2, trans_X3], axis=1)

    # print(f'trans_X:{trans_X.shape[1]}') # 20
    best_programs = best_programs1 + best_programs2 + best_programs3 
    # print(f'len_program:{len(best_programs)}')

    ir = []
    ic = []
    for i in range(trans_X.shape[1]):
        ir.append(raw_fitness(trans_X[:, i], y))
        ic.append(np.corrcoef(trans_X[:, i], y))
        # print(ir[-1]) # (-2.768814449233023, -0.171875)
    # print(f'ic:{len(ic)}') # 20
    # print(f'ir:{len(ir)}') # 20
    # print(f'ir shape:{len(ir[0])}')

    
    # np.corrcoef(trans_X.T)

    # for i, program in enumerate(best_programs):
    #     if program is not None:    
    #         print(f'fitneess:{program.fitness_}')
    #         print(f'Corr:{ic[i]}')
    #         with open(file_name, 'w') as f:
    #             f.write(str(program) + '\n')
    #             f.write(f'Sharpe: {ir[i][0]}, IC Mean: {ir[i][1]}\n')
    #             f.write('\n')

    mapping = {f'X{i}': name for i, name in enumerate(feature_list)}
    sorted_mapping = dict(sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True))

    file_name = f'expression17/{cnt}.txt'

    cnt += 1

    # 保留三部分相加

    with open(file_name, 'w') as f: # 要改成每个类生成的feature分开保存
        for i, program in enumerate(best_programs):
            if program is not None:
                program_str = str(program)

                # 替换所有的 X 变量名
                for old, new in sorted_mapping.items():
                    program_str = program_str.replace(old, new)
                # transformed_X = gp.transform([program])
                # sharpe, ic_mean = raw_fitness(transformed_X, y)
                f.write(program_str + '\n')
                # print(i)
                f.write(f'Sharpe: {ir[i][0]}, IC Mean: {ir[i][1]}\n') # 越界
                f.write(f'Fitness:{program.fitness_}, Corr:{ic[i][0][1]}\n')
                f.write('\n')
    # with open(f'{model_path}model_{name_list[g][0]}_{name_list[g][1]}.pkl-test', 'wb') as file:
    #     # pickle.dump(gp, file)

