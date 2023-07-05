#from pycaret.classification import * 
from pycaret.regression import * 
from pycaret.datasets import get_data
import pandas as pd
import numpy as np
import os
import pickle

# SMLP
from smlp.smlp_plot import *
from smlp.formula_sklearn import SklearnFormula
from logs_common import create_logger


# default hyper params for local models
# TODO !!!: should this be part of the class?
DEF_CV_FOLDS = 0

# hyper params dictionary for ceras model training
caret_hparam_dict = {
    'cv_folds': {'abbr':'cv_folds', 'default': DEF_CV_FOLDS, 'type':int,
        'help': 'cross-validation folds [default: ' + str(DEF_CV_FOLDS) + ']'}}


# TODO: couldn't manage to disable cross-validation, looks like at least two folds is a must.
# TODO: sample weights do not work with cross-validation. See https://github.com/pycaret/pycaret/issues/1350

# methods for training and predction with CARET package 
# Currently 'rf', 'dt', 'et', 'lightgbm', 'gbr', 'knn' are supported
# When addig new models self._CARET_MODELS = ['nn'] needs to be updated
class ModelCaret:
    def __init__(self, log_file : str, log_level : str, log_mode : str, log_time : str):    
        #data_logger = logging.getLogger(__name__)
        self._caret_logger = create_logger('caret_logger', log_file, log_level, log_mode, log_time)
        self._CARET_MODELS = ['rf', 'dt', 'et', 'lightgbm', 'gbr', 'knn']
        self.SMLP_CARET_MODELS = [self._algo_name_local2global(m) for m in self._CARET_MODELS]
        self._instFormula = SklearnFormula()
        
    # local names for model are 'dt', 'rf', ..., while global names are 'dt_caret'
    # 'rf_caret', to distinguish dt, rf, ... implementation in different packages
    def _algo_name_local2global(self, algo):
        return algo+'_caret'
    
    # local hyper params dictionary
    def get_caret_hparam_default_dict(self):
        caret_hparam_deafult_dict = { # TODO !!! do we need metavar feild? metavar='CV_FOLDS',
            'cv_folds': {'abbr':'cv_folds', 'default': DEF_CV_FOLDS, 'type':int,
                'help': 'cross-validation folds [default: ' + str(DEF_CV_FOLDS) + ']'}
        }
        return caret_hparam_deafult_dict

    # predictions for single response using models supported in caret package
    def _caret_train_single_response(self, inst, feature_names, resp_name, algo,
            X_train, X_test, y_train, y_test, interactive_plots,
            folds=3, sample_weights_vect=None, models_compare=False, 
            predict=False, predict_new=False, save_final_model=False):

        # prefix of output / report filename for this function -- includes _filename_prefix,
        # the name of the algo/model, name of the response, and report-specific suffix
        resp_model_report_filename_prefix = inst._filename_prefix + '_' + self._algo_name_local2global(algo) + '_' + resp_name

        # compute sample waights based on the value in the response y_train
        # Sample weights do not work with cross-validation (see the comment above)
        # cross-validation cannot be disabled, therefore currently sample weights are 
        # used in compare_models() but not in create_model() and tune_model()
        use_sample_weights = False
        perform_cv = folds > 1 # whether to perform cross-validation
        df_train = pd.concat([X_train, y_train], axis=1)
        #print('df_train\n', df_train); print(resp_name)
        exp_clf = setup(df_train, target=resp_name)

        # create multiple models to find best (this step is optional, useful but time consuming)
        if models_compare:
            self._caret_logger.info('compare models')
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
            os.rename('./Residuals.png', resp_model_report_filename_prefix + '_Residuals.png')
            self._caret_logger.info('Errors Plot')
            plot_model(tuned_model, plot='error', save=True)
            os.rename('./Prediction Error.png', resp_model_report_filename_prefix + '_PredictionError.png')

            # 'knn' does not support feature ranking, it does not have attribute 'feature_importances_'
            if hasattr(tuned_model, 'coef_') and hasattr(tuned_model, 'feature_importances_'): 
                self._caret_logger.info('Features Ranking')
                plot_model(tuned_model, plot = 'feature', save=True)
                os.rename('./Residuals.png', resp_model_report_filename_prefix + '_Residuals.png')

        # train model on entire input data -- training and test sets together
        self._caret_logger.info('Finalizing {} model: start'.format(algo))
        final_model = finalize_model(tuned_model)
        # TODO: plot_model() fails
        # plot_model(final_model, plot='tree', save=True)
        self._caret_logger.info('Finalizing {} model: end'.format(algo))

        # export trees into stdout and into file
        rules_filename = resp_model_report_filename_prefix + '_tree_rules.txt'
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
        self._instFormula.trees_to_rules(tree_estimators, feature_names, [resp_name], None, False, rules_filename)

        # Save the final model; to load the saved model later on use this function
        # Loaded pkl model is actually not so useful, it does not describe the model formula/trees
        model_filename = resp_model_report_filename_prefix + '_caret_model'
        if save_final_model:
            save_model(final_model, model_filename)
            # read model in two ways, see the content
            with open(model_filename + '.pkl', 'rb') as f:
                data = pickle.load(f)
            #print('pkl model\n', data)
            model_loaded = load_model(model_filename)
            #print('loaded model\n', model_loaded)

        return final_model

    # supports training multiple responses by iterating caret_train_single_response()
    # on each response (for now, sequentially);
    # the return value is a dictionary with the response names as keys and the repsctive 
    # models as values (this is true also if there is only one response in training data).
    # TODO !!!: couldn't figure out how to add a seed to ensure reproducibility
    def caret_main(self, inst, feature_names, response_names, algo,
            X_train, X_test, y_train, y_test, hparam_dict, interactive_plots,
            seed, sample_weights_vect, models_compare=False, save_final_model=False):

        # supported model training algorithms
        if not algo in self._CARET_MODELS:
            raise Exception('Unsupported model ' + str(algo) + ' in caret_main')

        folds = hparam_dict['cv_folds']

        models = {}
        for rn in response_names:
            models[rn] = self._caret_train_single_response(inst, feature_names, rn, algo,
            X_train, X_test, y_train[[rn]], y_test[[rn]], interactive_plots,
            folds, sample_weights_vect=sample_weights_vect, models_compare=models_compare, 
            save_final_model=save_final_model)
        return models
