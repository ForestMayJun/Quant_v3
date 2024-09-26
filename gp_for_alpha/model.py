'''
主模型框架
'''

from data_forgp import X_train, X_test, y_train, y_test
from operator_forgp import init_func, user_func, my_func
from oprator2 import my_func
from oprators3 import function_set, my_metric
# from fitness import my_metric
from gplearn.genetic import SymbolicTransformer
import os
import pandas as pd
import pickle

X_train, y_train = pd.read_csv('gp_for_alpha/data.csv')['x'].to_numpy().reshape(-1,1), pd.read_csv('gp_for_alpha/data.csv')['y'].to_numpy()

# file_path = 'gp_alpha.pkl'
# if not os.path.isfile(file_path):
#     with open(file_path, 'w') as f:
#         pass


def main():

    # creat the gplearn model
    est = SymbolicTransformer(
    function_set=function_set,
    population_size=100,
    generations=20,
    # metric=my_metric,
    hall_of_fame=10,
    # stopping_criteria=0.01,
    p_crossover=0.4,
    p_subtree_mutation=0.01,
    p_hoist_mutation=0,
    p_point_mutation=0.01,
    p_point_replace=0.4,
    parsimony_coefficient=0.01,
    verbose=2,
    random_state=5,
    )

    est.fit(X_train, y_train)

    # show the alpha
    best_programs = est._best_programs
    # best_programs = est.hall_of_fame
    best_programs_dic = {}

    for p in best_programs:
        if str(p) not in best_programs_dic['expression']:
            factor_name = 'alpha_' + str(best_programs.index(p) + 1)
            best_programs_dic[factor_name] = {
                'fitness':p.fitness_,
                'expression':str(p),
                'depth':p.depth_,
                'length':p.length_,
            }
    
    best_programs_dic = pd.DataFrame(best_programs_dic).T
    best_programs_dic = best_programs_dic.sort_values(by='fitness')
    print(best_programs_dic)

    # # save the model
    # with open(file_path, 'wb') as f:
    #     pickle.dump(est, f)


if __name__ == '__main__':

    main()