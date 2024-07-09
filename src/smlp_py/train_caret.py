#from pycaret.classification import * 
from pycaret.regression import * 
from pycaret.datasets import get_data
import pandas as pd
import numpy as np
import os

# SMLP
from smlp_py.smlp_plots import *
from smlp_py.smlp_terms import TreeTerms
from smlp_py.smlp_utils import str_to_bool


# TODO: couldn't manage to disable cross-validation, looks like at least two folds is a must.
# TODO: sample weights do not work with cross-validation. See https://github.com/pycaret/pycaret/issues/1350

# Methods for training and predction, results reporting with CARET package 
# Currently 'rf', 'dt', 'et', 'lightgbm', 'gbr', 'knn' are supported
# When addig new models self._CARET_MODELS = ['nn'] needs to be updated
class ModelCaret:
    def __init__(self):   
        #data_logger = logging.getLogger(__name__)
        self._caret_logger = None 
        self._CARET_MODELS = ['rf', 'dt', 'et', 'lightgbm', 'gbr', 'knn']
        self.SMLP_CARET_MODELS = [self._algo_name_local2global(m) for m in self._CARET_MODELS]
        self._instTreeTerms = TreeTerms()
        # params for setup() function: session_id, verbose, fold, data_split_shuffle (set to False to acheive reproducibility?)
        # setup() hyper parameter defaults
        self._DEF_SESSION_ID = None
        self._DEF_DATA_SPLIT_SHUFFLE = True # True
        self._DEF_CV_FOLDS = 3 #2
        self._DEF_TREE_MAX_DEPTH = 3 #3
        self._DEF_CROSS_VALIDATION = True # True
        self._DEF_VERBOSE = True
        self._DEF_RETURN_TRAIN_SCORE = False

        # tune_model() default values
        self._DEF_SEARCH_ALGO = 'random' # 'random'
        self._DEF_TUNER_VERBOSE = True
        
        #  params dictionary for setup() function
        self._setup_hparam_dict = {
            'session_id': {'abbr':'session_id', 'default': self._DEF_SESSION_ID, 'type':int,
                'help': 'Controls the randomness of experiment. It is equivalent to ‘random_state’ ' +
                        'in scikit-learn. When None, a pseudo random number is generated. ' +
                        'This can be used for later reproducibility of the entire experiment ' + 
                        '[default: ' + str(self._DEF_SESSION_ID) + ']'},
            'fold': {'abbr':'fold', 'default': self._DEF_CV_FOLDS, 'type':int,
                'help': 'Controls cross-validation. If None, the CV generator in the ' +
                        'fold_strategy parameter of the setup function is used. ' +
                        'When an integer is passed, it is interpreted as the ‘n_splits’ ' +
                        'parameter of the CV generator in the setup function. ' +
                        '[default: ' + str(self._DEF_CV_FOLDS) + ']'},
            'data_split_shuffle': {'abbr':'data_split_shuffle', 'default': self._DEF_DATA_SPLIT_SHUFFLE, 'type':str_to_bool,
                'help': 'When set to False, prevents shuffling of rows during ‘train_test_split’. ' + 
                        '[default: ' + str(self._DEF_DATA_SPLIT_SHUFFLE) + ']'},
            'verbose': {'abbr':'verbose', 'default': self._DEF_VERBOSE, 'type':str_to_bool,
                'help': 'When set to False, Information grid is not printed. ' +
                        '[default: ' + str(self._DEF_VERBOSE) + ']'},
            }

        # params dictionary for create_model() function
        self._model_hparam_dict = {
            'cross_validation': {'abbr':'cross_validation', 'default': self._DEF_CROSS_VALIDATION, 'type':str_to_bool,
                'help': 'When set to False, metrics are evaluated on holdout set. ' +
                        'fold param is ignored when cross_validation is set to False. ' +
                        '[default: ' + str(self._DEF_CROSS_VALIDATION) + ']'},
            'verbose': {'abbr':'verbose', 'default': self._DEF_VERBOSE, 'type':str_to_bool,
                'help': 'Score grid is not printed when verbose is set to False. ' +
                        '[default: ' + str(self._DEF_VERBOSE) + ']'},
            'return_train_score': {'abbr':'return_train_score', 'default': self._DEF_RETURN_TRAIN_SCORE, 'type':str_to_bool,
                'help': 'If False, returns the CV Validation scores only. If True, returns ' +
                        'the CV training scores along with the CV validation scores. ' +
                        'This is useful when the user wants to do bias-variance tradeoff. ' +
                        'A high CV training score with a low corresponding CV validation score ' +
                        'indicates overfitting. [default: ' + str(self._DEF_RETURN_TRAIN_SCORE) + ']'}
            }

        # params dictionary for tune_model() function
        self._tuner_hparam_dict = {
            'search_algorithm': {'abbr':'search_algo', 'default': self._DEF_SEARCH_ALGO, 'type':str,
                'help': 'The search algorithm depends on the search_library parameter. ' +
                        'If None, will use search library-specific default algorithm. ' +
                        'Other possible values are ‘random’ : random grid search (default) ' +
                        'and ‘grid’ : grid search [default: ' + str(self._DEF_SEARCH_ALGO) + ']'},
            'tuner_verbose': {'abbr':'tuner_verbose', 'default': self._DEF_TUNER_VERBOSE, 'type':str_to_bool,
                'help': 'If True or above 0, will print messages from the tuner. ' +
                        'Ignored when verbose param is False. [default: ' + str(self._DEF_TUNER_VERBOSE) + ']'}
            }

        self.caret_hparam_dict = self.get_caret_hparam_default_dict()
        
    # set logger from a caller script
    def set_logger(self, logger):
        self._caret_logger = logger
        self._instTreeTerms.set_logger(logger)
    
    # set report_file_prefix from a caller script
    def set_report_file_prefix(self, report_file_prefix):
        self.report_file_prefix = report_file_prefix
    
    # set model_file_prefix from a caller script
    def set_model_file_prefix(self, model_file_prefix):
        self.model_file_prefix = model_file_prefix
    
    # local names for model are 'dt', 'rf', ..., while global names are 'dt_caret'
    # 'rf_caret', to distinguish dt, rf, ... implementation in different packages
    def _algo_name_local2global(self, algo):
        return algo+'_caret'
    
    # local name of hyper parameter (as in sklearn package) to global name;
    # the global name is obtained from local name, say 'max_depth', by prefixing it
    # with the global name of the algorithm, which results in 'dt_sklearn_max_depth'
    def _hparam_name_local_to_global(self, hparam, algo):
        #print('hparam global name', hparam, algo)
        return self._algo_name_local2global(algo) + '_' + hparam
        
    # given training algo name like dt and the hyper parameter dictionary param_dict  
    # for that algo in the python package used in this class), this function returns  
    # a modified dictionary obtained from param_dict by adding algo name like dt_sklearn
    # (where sklearn is the name of the package used) to the parameter name and its
    # correponding abbriviated name in param_dict.
    def _param_dict_with_func_name(self, param_dict, func):
        #print('param_dict', param_dict)
        result_dict = {}
        for k, v in param_dict.items():
            v_updated = v.copy()
            v_updated['abbr'] = self._hparam_name_local_to_global(v['abbr'], func)
            #print('updated abbrv', v_updated['abbr'])
            #print('updated key', self._hparam_name_local_to_global(k, func))
            result_dict[self._hparam_name_local_to_global(k, func)] = v_updated
        #raise Exception('tmp')
        return result_dict
    
    # predictions for single response using models supported in caret package
    def _caret_train_single_response(self, get_model_file_prefix, feature_names, resp_name, algo,
            X_train, X_test, y_train, y_test, interactive_plots, seed, data_split_shuffle,
            folds=3, sample_weights_vect=None, models_compare=False):

        # prefix of output / report filename for this function -- includes _report_name_prefix,
        # the name of the algo/model, name of the response, and report-specific suffix
        resp_model_file_prefix = get_model_file_prefix([resp_name], self._algo_name_local2global(algo))
        
        # compute sample waights based on the value in the response y_train
        # Sample weights do not work with cross-validation (see the comment above)
        # cross-validation cannot be disabled, therefore currently sample weights are 
        # used in compare_models() but not in create_model() and tune_model()
        use_sample_weights = False
        perform_cv = folds > 1 # whether to perform cross-validation
        df_train = pd.concat([X_train, y_train], axis=1)

        #print('df_train\n', df_train); print(resp_name)
        exp_clf = setup(df_train, target=resp_name, session_id=seed, data_split_shuffle=data_split_shuffle)

        # create multiple models to find best (this step is optional, useful but time consuming)
        if models_compare:
            eslf._caret_logger.info('compare models')
            best_model = compare_models(cross_validation=perform_cv, fold=max(2,folds), 
                fit_kwargs={'sample_weight': sample_weights_vect}, n_select=1) ; #print('best model\n', best_model)
        
        # Uses the default hyperparameters to train the model; need to pass it the required algo
        # since otherwise the best model found by compare models will be used (if it was run)
        self._caret_logger.info('Creating {} model: start'.format(algo))    
        if use_sample_weights:
            model = create_model(algo, cross_validation=perform_cv, fold=max(2,folds), 
                                 fit_kwargs={'sample_weight': sw})
        else:
            model = create_model(algo, cross_validation=perform_cv, fold=max(2,folds))
        self._caret_logger.info('Creating {} model: end'.format(algo))
        #print('created model\n', model)

        # Tunes the hyperparameters using a default grid RandomGridSearch()
        # One can aslo pass a customized custom_grid to tune_model() function.
        # Tuning is by defeult done wrt R2; can changes using option optimize.
        # For example: tune_model(model, optimize = 'MAE')
        self._caret_logger.info('Tuning {} model: start'.format(algo))
        # TODO: couldn't find a way to desable cross-validation in tune_model().
        # fold=1 causes runtime error, fold=0 implies default fold=10, and 
        # cross_validation is not supported as argument to disable CV.
        # Also, cross_validation=False enables to skip CV in create_model()
        # but does not seem to have any effect in compare_models()
        if use_sample_weights:
            tuned_model = tune_model(model, max(2,folds), fit_kwargs={'sample_weight': sw})
        else:
            tuned_model = tune_model(model, max(2,folds)) 
        self._caret_logger.info('Tuning {} model: end'.format(algo))

        # display tuned_model, its error metrics, feature ranking; TODO: requires more work...
        if False:
            self._caret_logger.info('Residuals Plot:')
            #plot_model(tuned_model, plot='residuals_interactive')
            plot_model(tuned_model, save=True) # Residuals Plot
            os.rename('./Residuals.png', resp_model_file_prefix + '_Residuals.png')
            self._caret_logger.info('Errors Plot')
            plot_model(tuned_model, plot='error', save=True)
            os.rename('./Prediction Error.png', resp_model_file_prefix + '_PredictionError.png')

            # 'knn' does not support feature ranking, it does not have attribute 'feature_importances_'
            if hasattr(tuned_model, 'coef_') and hasattr(tuned_model, 'feature_importances_'): 
                self._caret_logger.info('Features Ranking')
                plot_model(tuned_model, plot = 'feature', save=True)
                os.rename('./Residuals.png', resp_model_file_prefix + '_Residuals.png')

        # train model on entire input data -- training and test sets together
        self._caret_logger.info('Finalizing {} model: start'.format(algo))
        final_model = finalize_model(tuned_model)
        # TODO: plot_model() fails
        # plot_model(final_model, plot='tree', save=True)
        self._caret_logger.info('Finalizing {} model: end'.format(algo))

        # export trees into stdout and into file
        rules_filename = resp_model_file_prefix + '_tree_rules.txt'
        if algo == 'dt':
            tree_estimators = [final_model]
        elif algo in ['rf', 'et']:
            tree_estimators = final_model.estimators_
        elif algo == 'lightgbm':
            final_model.booster_.dump_model()
            lgbm_trees = final_model.booster_.trees_to_dataframe()
            #print(final_model._Booster.dump_model()["tree_info"])
            #print(lgbm_trees)
            raise Exception('Conversion of light GBM model to tree rules is not implemented yet')
        else:
            raise Exception('Algo ' + str(algo) + ' is not supported in model to formula conversion')
        self._instTreeTerms.trees_to_rules(tree_estimators, feature_names, [resp_name], None, False, rules_filename)

        return final_model
        
    # local hyper params dictionary
    def get_caret_hparam_default_dict(self):
        caret_setup_hparam_dict = self._param_dict_with_func_name(self._setup_hparam_dict, 'setup')
        caret_model_hparam_dict = self._param_dict_with_func_name(self._model_hparam_dict, 'model')
        caret_tuner_hparam_dict = self._param_dict_with_func_name(self._tuner_hparam_dict, 'tuner')
        caret_hparam_dict = caret_setup_hparam_dict | caret_model_hparam_dict | caret_tuner_hparam_dict
        return caret_hparam_dict

    # supports training multiple responses by iterating caret_train_single_response()
    # on each response (for now, sequentially);
    # the return value is a dictionary with the response names as keys and the repsctive 
    # models as values (this is true also if there is only one response in training data).
    # TODO !!!: couldn't find a way to build one model covering all responses -- likely not possible currently
    def caret_main(self, get_model_file_prefix, feature_names_dict, response_names, algo,
            X_train, X_test, y_train, y_test, hparam_dict, interactive_plots,
            seed, sample_weights_vect, models_compare=False):
        # supported model training algorithms
        if not algo in self._CARET_MODELS:
            raise Exception('Unsupported model ' + str(algo) + ' in caret_main')

        folds = hparam_dict[self._hparam_name_local_to_global('fold', 'setup')]
        data_split_shuffle = hparam_dict[self._hparam_name_local_to_global('data_split_shuffle', 'setup')]
        #max_depth = hparam_dict['caret_max_depth']

        models = {}
        for rn in response_names:
            models[rn] = self._caret_train_single_response(get_model_file_prefix, feature_names_dict[rn], rn, algo,
            X_train, X_test, y_train[[rn]], y_test[[rn]], interactive_plots, seed, data_split_shuffle,
            folds, sample_weights_vect=sample_weights_vect, models_compare=models_compare)
        
        return models
