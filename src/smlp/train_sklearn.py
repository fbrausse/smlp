# Fitting sklearn regression tree models
from sklearn.tree import DecisionTreeRegressor
#from sklearn.tree import _tree
from sklearn import tree, ensemble

# Fitting sklearn polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# general
import numpy as np
import pandas as pd
import pickle

# SMLP
from smlp.smlp_plot import *
from smlp.formula_sklearn import SklearnFormula
from logs_common import create_logger


# defaults for local parameters
# TODO !!!: should way be within the class?
DEF_POLY_DEGREE = 2

# hyper params dictionary for sklearn model training
sklearn_hparam_dict = {
    'poly_degree': {'abbr':'poly_degree', 'default': DEF_POLY_DEGREE, 'type':int,
        'help': 'Degree of the polynomial to train [default: ' + str(DEF_POLY_DEGREE) + ']'}}

# methods for training and predction with SKLEARN package   
# Currently 'rf', 'dt', 'et', 'poly' are supported
# When addig new models self._KERAS_MODELS = ['nn'] needs to be updated
class ModelSklearn:
    def __init__(self, log_file : str, log_level : str, log_mode : str, log_time : str):    
        #data_logger = logging.getLogger(__name__)
        self._sklearn_logger = create_logger('sklearn_logger', log_file, log_level, log_mode, log_time)
        self._SKLEARN_MODELS = ['dt', 'et', 'rf', 'poly']
        self.SMLP_SKLEARN_MODELS = [self._algo_name_local2global(m) for m in self._SKLEARN_MODELS]
        self._instFormula = SklearnFormula()
        
    # local names for model are 'dt', 'rf', ..., while global names are 'dt_sklearn'
    # 'rf_sklearn', to distinguish dt, rf, ... implementation in different packages
    def _algo_name_local2global(self, algo):
        return algo+'_sklearn'
    
    # local hyper params dictionary
    def get_sklearn_hparam_default_dict(self):
        sklearn_hparam_deafult_dict = { # TODO !!! do we need metavar feild?
            'poly_degree': {'abbr':'poly_degree', 'default': DEF_POLY_DEGREE, 'type':int,
                'help': 'Degree of the polynomial to train [default: ' + str(DEF_POLY_DEGREE) + ']'}, 
        }
        return sklearn_hparam_deafult_dict

    # train decision tree regression model with sklearn
    def dt_regr_train(self, feature_names, resp_names, algo,
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
    def rf_regr_train(self, feature_names, resp_names, algo,
            X_train, X_test, y_train, y_test, seed, weights, save_model):
        # Fit the regressor, set max_depth = 3
        model = ensemble.RandomForestRegressor(max_depth=15, random_state=seed)
        model = model.fit(X_train, y_train, sample_weight=weights)
        #assert regr == model

        return model

    # train extra trees regression model with sklearn
    def et_regr_train(self, feature_names, resp_names, algo,
            X_train, X_test, y_train, y_test, seed, weights, save_model):
        # Fit the regressor, set max_depth = 3
        model = ensemble.ExtraTreesRegressor(max_depth=15, random_state=seed)
        model = model.fit(X_train, y_train, sample_weight=weights)
        #assert regr == model
        #assert regr.estimators_[0].tree_.value.shape[1] == 1 # no support for multi-output
        #self._instFormula.trees_to_rules(inst, regr.estimators_, feature_names, resp_names, None, True, True)

        return model

    # prints basic statistics of each column of df.
    # similar to pd.descrie() but this functions only prints a few columns when df has many columns
    def df_cols_summary(self, df):
        for col in df.columns.tolist():
            print(col, 
                  'min', df[col].min(), 
                  'max', df[col].max(),
                  'mean', df[col].mean(),
                  'std', df[col].std(), 
                  'z_min', (df[col].min() - df[col].mean())/df[col].std(), 
                  'z_max', (df[col].max() - df[col].mean())/df[col].std())  

    # train polynomial regression model with sklearn
    def poly_train(self, input_names, resp_names, degree,
            X_train, X_test, y_train, y_test, seed, weights):
        #print('poly_degree', degree); print('weigts', weights);     
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

    def sklearn_main(self, inst, input_names, resp_names, algo,
            X_train, X_test, y_train, y_test, hparam_dict, interactive_plots, 
            seed, sample_weights_vect, save_model):

        #print('sklearn_main, hparam_dict', hparam_dict); print('sample_weights_vect', sample_weights_vect)
        degree = hparam_dict['poly_degree']

        if algo == 'dt':
            model = self.dt_regr_train(input_names, resp_names, algo,
                X_train, X_test, y_train, y_test, seed, sample_weights_vect, save_model)
        elif algo == 'rf':
            model = self.rf_regr_train(input_names, resp_names, algo,
                X_train, X_test, y_train, y_test, seed, sample_weights_vect, save_model)
        elif algo == 'et':
            model = self.et_regr_train(input_names, resp_names, algo,
                X_train, X_test, y_train, y_test, seed, sample_weights_vect, save_model)
        elif algo == 'poly':
            model, poly_reg, X_train, X_test = self.poly_train(input_names, resp_names, degree,
                X_train, X_test, y_train, y_test, seed, sample_weights_vect)
        else:
            raise Exception('Unsupported model type ' + str(algo) + ' in function tree_main')

        # export the model formula
        if algo in ['dt', 'rf', 'et']:
            # Print/save rule representation of the model (left-hand sides represent
            # branhes of the tree till the leaves, right-hand side are the prdicted 
            # values and they are the lables at the respective leaves of the tree
            rules_filename = inst._filename_prefix + '_' + str(self._algo_name_local2global(algo)) + '_tree_rules.txt'
            if algo == 'dt':
                tree_estimators = [model]
            elif algo in ['rf', 'et']:
                tree_estimators = model.estimators_
            self._instFormula.trees_to_rules(tree_estimators, input_names, resp_names, None, True, rules_filename)
        elif algo == 'poly':
            #print('Polynomial model coef\n', model.coef_.shape, '\n', model.coef_)
            #print('Polynomial model terms\n', poly_reg.powers_.shape, '\n', poly_reg.powers_)
            for resp_id in range(len(resp_names)):
                formula_filename = inst._filename_prefix + '_' + str(algo) + '_' + resp_names[resp_id] + "_poly_formula.txt"
                model_formula = self._instFormula.poly_model_to_formula(input_names, resp_names, model.coef_, 
                    poly_reg.powers_, resp_id, True, formula_filename)

        # save model to enable re-running on new data sets
        if save_model:
            model_filename = inst._filename_prefix + '_' + self._algo_name_local2global(algo) + '.model'
            pickle.dump(model, open(model_filename, 'wb'))

            # load the model to see the content
            loaded_model = pickle.load(open(model_filename, 'rb'))

        if algo in ['dt', 'rf', 'et']: 
            return model
        elif algo == 'poly':
            return model, poly_reg, X_train, X_test

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
