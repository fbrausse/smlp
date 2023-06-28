#from pycaret.classification import * 
from pycaret.regression import * 
from pycaret.datasets import get_data
import pandas as pd
import numpy as np
import os
import pickle
from smlp.smlp_plot import *
from smlp.formula_sklearn import trees_to_rules
#from smlp.train_common import report_prediction_results

# local names for model are 'dt', 'rf', ..., while global names are 'dt_sklearn'
# 'rf_sklearn', to distinguish dt, rf, ... implementation in different packages
def algo_name_local2global(algo):
    return algo+'_caret'

CARET_MODELS = ['rf', 'dt', 'et', 'lightgbm', 'gbr', 'knn']
SMLP_CARET_MODELS = [algo_name_local2global(m) for m in CARET_MODELS]

DEF_CV_FOLDS = 0


def get_caret_hparam_deafult_dict():
    caret_hparam_deafult_dict = { # TODO !!! do we need metavar feild? metavar='CV_FOLDS',
        'cv_folds': {'abbr':'cv_folds', 'default': DEF_CV_FOLDS, 'type':int,
            'help': 'cross-validation folds [default: ' + str(DEF_CV_FOLDS) + ']'}
    }
    return caret_hparam_deafult_dict

# TODO: couldn't manage to disable cross-validation, looks like at least two folds is a must.
# TODO: sample weights do not work with cross-validation. See https://github.com/pycaret/pycaret/issues/1350

# predictions for single response using models supported in caret package
def caret_train_single_response(inst, feature_names, resp_name, algo,
        X_train, X_test, y_train, y_test, interactive_plots,
        folds=3, sample_weights=0, models_compare=False, 
        predict=False, predict_new=False, save_final_model=False):

    # prefix of output / report filename for this function -- includes _filename_prefix,
    # the name of the algo/model, name of the response, and report-specific suffix
    resp_model_report_filename_prefix = inst._filename_prefix + '_' + algo_name_local2global(algo) + '_' + resp_name

    # compute sample waights based on the value in the response y_train
    # Sample weights do not work with cross-validation (see the comment above)
    # cross-validation cannot be disabled, therefore currently sample weights are 
    # used in compare_models() but not in create_model() and tune_model()
    use_sample_weights = False
    sw = y_train.mean(axis='columns').values # applying mean is only needed for multiple responses
    sw = np.power(sw, sample_weights); #print('sw\n', sw)
    if sample_weights < 0:
        sw = 1 - sw
    assert any(sw <= 1) ; assert any(sw >= 0)

    perform_cv = folds > 1 # whether to perform cross-validation
    df_train = pd.concat([X_train, y_train], axis=1)
    #print('df_train\n', df_train); print(resp_name)
    exp_clf = setup(df_train, target=resp_name)
    
    # create multiple models to find best (this step is optional, useful but time consuming)
    if models_compare:
        print('compare models')
        best_model = compare_models(cross_validation=perform_cv, fold=max(2,folds), 
            fit_kwargs={'sample_weight': sw}, n_select=1) ; #print('best model\n', best_model)
    # Uses the default hyperparameters to train the model; need to pass it the required algo
    # since otherwise the best model found by compare models will be used (if it was run)
    print('Creating {} model: start'.format(algo))    
    if use_sample_weights:
        model = create_model(algo, cross_validation=perform_cv, fold=max(2,folds), 
                             fit_kwargs={'sample_weight': sw})
    else:
        #print('sample_weights', sample_weights, 'sw\n', sw)
        model = create_model(algo, cross_validation=perform_cv, fold=max(2,folds))
    print('Creating {} model: end'.format(algo))
    #print('created model\n', model)
    
    # Tunes the hyperparameters using a default grid RandomGridSearch()
    # One can aslo pass a customized custom_grid to tune_model() function.
    # Tuning is by defeult done wrt R2; can changes using option optimize.
    # For example: tune_model(model, optimize = 'MAE')
    print('Tuning {} model: start'.format(algo))
    # TODO: couldn't find a way to desable cross-validation in tune_model().
    # fold=1 causes runtime error, fold=0 implies default fold=10, and 
    # cross_validation is not supported as argument to disable CV.
    # Also, cross_validation=False enables to skip CV in create_model()
    # but does not seem to have any effect in compare_models()
    if use_sample_weights:
        tuned_model = tune_model(model, max(2,folds), fit_kwargs={'sample_weight': sw})
    else:
        tuned_model = tune_model(model, max(2,folds)) 
    print('Tuning {} model: end'.format(algo))
    
    # display tuned_model, its error metrics, feature ranking; TODO: requires more work...
    if False:
        print('Residuals Plot:')
        #plot_model(tuned_model, plot='residuals_interactive')
        plot_model(tuned_model, save=True) # Residuals Plot
        os.rename('./Residuals.png', resp_model_report_filename_prefix + '_Residuals.png')
        print('Errors Plot')
        plot_model(tuned_model, plot='error', save=True)
        os.rename('./Prediction Error.png', resp_model_report_filename_prefix + '_PredictionError.png')

        # 'knn' does not support feature ranking, it does not have attribute 'feature_importances_'
        if hasattr(tuned_model, 'coef_') and hasattr(tuned_model, 'feature_importances_'): 
            print('Features Ranking')
            plot_model(tuned_model, plot = 'feature', save=True)
            os.rename('./Residuals.png', resp_model_report_filename_prefix + '_Residuals.png')
        
    # train model on entire input data -- training and test sets together
    print('Finalizing {} model: start'.format(algo))
    final_model = finalize_model(tuned_model)
    # TODO: plot_model() fails
    # plot_model(final_model, plot='tree', save=True)
    print('Finalizing {} model: end'.format(algo))
    
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
    trees_to_rules(tree_estimators, feature_names, [resp_name], None, False, rules_filename)
    
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
def caret_main(inst, feature_names, response_names, algo,
        X_train, X_test, y_train, y_test, hparam_dict, interactive_plots,
        seed, sample_weights=0, models_compare=False, save_final_model=False):
    
    # supported model training algorithms
    if not algo in CARET_MODELS:
        raise Exception('Unsupported model ' + str(algo) + ' in caret_main')

    folds = hparam_dict['cv_folds']
    
    models = {}
    for rn in response_names:
        models[rn] = caret_train_single_response(inst, feature_names, rn, algo,
        X_train, X_test, y_train[[rn]], y_test[[rn]], interactive_plots,
        folds, sample_weights=sample_weights, models_compare=models_compare, 
        save_final_model=save_final_model)
    return models
