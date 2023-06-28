# Fitting sklearn regression tree models
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import _tree
from sklearn import tree, ensemble

# Fitting sklearn polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# general
import numpy as np
import pandas as pd
import pickle

# misc
from smlp.smlp_plot import *
from smlp.formula_sklearn import (trees_to_rules, poly_model_to_formula)
#from smlp.train_common import report_prediction_results

# local names for model are 'dt', 'rf', ..., while global names are 'dt_sklearn'
# 'rf_sklearn', to distinguish dt, rf, ... implementation in different packages
def algo_name_local2global(algo):
    return algo+'_sklearn'

SKLEARN_MODELS = ['dt', 'et', 'rf', 'poly']
SMLP_SKLEARN_MODELS = [algo_name_local2global(m) for m in SKLEARN_MODELS]

DEF_POLY_DEGREE = 2

def get_sklearn_hparam_deafult_dict():
    sklearn_hparam_deafult_dict = { # TODO !!! do we need metavar feild?
        'poly_degree': {'abbr':'poly_degree', 'default': DEF_POLY_DEGREE, 'type':int,
            'help': 'Degree of the polynomial to train [default: ' + str(DEF_POLY_DEGREE) + ']'}, 
    }
    return sklearn_hparam_deafult_dict

'''
# generates domain file in current format for running the solvers.
# TODO: this is for temporary usage, amd also does not belong to this file
def training_data_to_domain_spec(X_poly, input_names):
    X_poly_df = pd.DataFrame(X_poly)
    print('X_poly_df\n', X_poly_df)
    X_test_scaled_df = X_poly_df[range(1,len(input_names)+1)]
    X_test_scaled_df.columns = input_names
    print('X_test_scaled_df\n', X_test_scaled_df)
    domain_spec = ''
    for col in X_test_scaled_df.columns.tolist():
        print(col, '\n', X_test_scaled_df[col].sort_values())
        domain_spec = domain_spec + col + ' -- ' + '[' + str(X_test_scaled_df[col].min()) + ',' + str(X_test_scaled_df[col].max()) + ']\n'
    print('domain_spec\n', domain_spec)
    return domain_spec
'''

# train decision tree regression model with sklearn
def dt_regr_train(feature_names, resp_names, algo,
        X_train, X_test, y_train, y_test, seed, weights, save_model):
    # Fit the regressor, set max_depth = 3
    regr = DecisionTreeRegressor(max_depth=15, random_state=seed)
    model = regr.fit(X_train, y_train, sample_weight=weights)
    assert(regr == model)
    
    # print text representation of the tree model
    text_representation = tree.export_text(model)
    print(text_representation)

    '''
    # visualaize tree TODO !!!!!!!!!!!! does not work 
    fig = plt.figure(figsize=(25,20))
    _ = tree.plot_tree(regr)
    plt.show()
    plt.clf()
    '''  
    return model

# train random forest regression model with sklearn
def rf_regr_train(feature_names, resp_names, algo,
        X_train, X_test, y_train, y_test, seed, weights, save_model):
    # Fit the regressor, set max_depth = 3
    model = ensemble.RandomForestRegressor(max_depth=15, random_state=seed)
    model = model.fit(X_train, y_train, sample_weight=weights)
    #assert regr == model

    return model

# train extra trees regression model with sklearn
def et_regr_train(feature_names, resp_names, algo,
        X_train, X_test, y_train, y_test, seed, weights, save_model):
    # Fit the regressor, set max_depth = 3
    model = ensemble.ExtraTreesRegressor(max_depth=15, random_state=seed)
    model = model.fit(X_train, y_train, sample_weight=weights)
    #assert regr == model
    #assert regr.estimators_[0].tree_.value.shape[1] == 1 # no support for multi-output
    #trees_to_rules(inst, regr.estimators_, feature_names, resp_names, None, True, True)

    return model

# prints basic statistics of each column of df.
# similar to pd.descrie() but this functions only prints a few columns when df has many columns
def df_cols_summary(df):
    for col in df.columns.tolist():
        print(col, 
              'min', df[col].min(), 
              'max', df[col].max(),
              'mean', df[col].mean(),
              'std', df[col].std(), 
              'z_min', (df[col].min() - df[col].mean())/df[col].std(), 
              'z_max', (df[col].max() - df[col].mean())/df[col].std())  
        
# train polynomial regression model with sklearn
def poly_train(input_names, resp_names, degree,
        X_train, X_test, y_train, y_test, seed, weights):
    print('poly_degree', degree)    
    poly_reg = PolynomialFeatures(degree)
    #dummy = df_cols_summary(X_train) 
    X_train = poly_reg.fit_transform(X_train)
    X_test = poly_reg.transform(X_test)
    #print('X_train', X_train.shape, '\n', X_train)
    #print('transformed X_train data')
    #print(pd.DataFrame(X_train).describe())
    #dummy = df_cols_summary(pd.DataFrame(X_train))
    #raise Exception('tmp')
    pol_reg = LinearRegression()
    model = pol_reg.fit(X_train, y_train, sample_weight=weights)

    ''' # writing spec file
    model_domain = training_data_to_domain_spec(X_train, input_names)
    domain_file = open(inst._filename_prefix + "_poly_domain.txt"), "w")
    domain_file.write(model_domain)
    domain_file.close()
    '''
    
    assert(model == pol_reg)
    return model, poly_reg, X_train, X_test

def sklearn_main(inst, input_names, resp_names, algo,
        X_train, X_test, y_train, y_test, hparam_dict, interactive_plots, 
        seed, weights, save_model):

    print('sklearn_main, hparam_dict', hparam_dict)
    degree = hparam_dict['poly_degree']
    
    # Compute sample weights. Looks like currently even if there are multiple responses
    # sklearn package only supports one vector/list of sample weights for all reponses  
    # (one cannot specify different vectors of weights per response)
    sw = y_train.mean(axis='columns').values; print('before applying weights', sw[0: min(5, len(sw))]);
    print('weights', weights)
    sw = np.power(sw, abs(weights)); print('after applying weights', sw[0: min(5, len(sw))]);
    if weights < 0:
        sw = 1 - sw
    assert any(sw <= 1) ; assert any(sw >= 0) ; print('sw', sw[0: min(5, len(sw))]);

    if algo == 'dt':
        model = dt_regr_train(input_names, resp_names, algo,
            X_train, X_test, y_train, y_test, seed, sw, save_model)
    elif algo == 'rf':
        model = rf_regr_train(input_names, resp_names, algo,
            X_train, X_test, y_train, y_test, seed, sw, save_model)
    elif algo == 'et':
        model = et_regr_train(input_names, resp_names, algo,
            X_train, X_test, y_train, y_test, seed, sw, save_model)
    elif algo == 'poly':
        model, poly_reg, X_train, X_test = poly_train(input_names, resp_names, degree,
            X_train, X_test, y_train, y_test, seed, sw)
    else:
        raise Exception('Unsupported model type ' + str(algo) + ' in function tree_main')
    
    # export the model formula
    if algo in ['dt', 'rf', 'et']:
        # Print/save rule representation of the model (left-hand sides represent
        # branhes of the tree till the leaves, right-hand side are the prdicted 
        # values and they are the lables at the respective leaves of the tree
        rules_filename = inst._filename_prefix + '_' + str(algo_name_local2global(algo)) + '_tree_rules.txt'
        if algo == 'dt':
            tree_estimators = [model]
        elif algo in ['rf', 'et']:
            tree_estimators = model.estimators_
        trees_to_rules(tree_estimators, input_names, resp_names, None, True, rules_filename)
    elif algo == 'poly':
        #print('Polynomial model coef\n', model.coef_.shape, '\n', model.coef_)
        #print('Polynomial model terms\n', poly_reg.powers_.shape, '\n', poly_reg.powers_)
        for resp_id in range(len(resp_names)):
            formula_filename = inst._filename_prefix + '_' + str(algo) + '_' + resp_names[resp_id] + "_poly_formula.txt"
            model_formula = poly_model_to_formula(input_names, resp_names, model.coef_, poly_reg.powers_, resp_id,
                                                  True, formula_filename)
        
    # save model to enable re-running on new data sets
    if save_model:
        model_filename = inst._filename_prefix + '_' + algo_name_local2global(algo) + '.model'
        pickle.dump(model, open(model_filename, 'wb'))
 
        # load the model to see the content
        loaded_model = pickle.load(open(model_filename, 'rb'))
    
    if algo in ['dt', 'rf', 'et']: 
        return model
    elif algo == 'poly':
        return model, poly_reg, X_train, X_test
        
