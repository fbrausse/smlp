#!/usr/bin/env python3
#
# This file is part of smlprover.
#
# Copyright 2019 Konstantin Korovin
# Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
# See the LICENSE file for terms of distribution.

# coding: utf-8

import os, sys, argparse
import pandas as pd

# imports from SMLP modules
from logs_common import *
from utils_common import np_JSONEncoder
from smlp.train_common import (model_train, model_predict, keras_dict, sklearn_dict, caret_dict, 
                              SMLP_KERAS_MODELS, SMLP_SKLEARN_MODELS, SMLP_CARET_MODELS, report_prediction_results)
from smlp.data_common import prepare_data_for_modeling


# for modular parsing -- parser from dict
from dictparse import DictionaryParser


# global defaults for SMLP

DEF_TRAIN_FIRST = 0 # subseting first_n rows from training data
DEF_TRAIN_RAND = 0  # sampling random_n rows from training data
DEF_TRAIN_UNIF = 0  # sampling from training data to acheive uniform distribution
DEF_SCALER = 'min-max'  # options: 'min-max', 'max-abs'
DEF_SPLIT_TEST = 0.2 # ratio to split training data into training and validation subsets
DEF_SAMPLE_WEIGHTS = 0 # degree/exponent of the power function that computes the
                       # sample weights based on response values on these samples


# default values of parameters related to dataset; used to generate args.
data_params_dict = {
    'response': {'abbr':'resp', 'default':None, 'type':str,
        'help':'Names of response variables, must be provided [default None]'}, 
    'features': {'abbr':'feat', 'default':None, 'type':str,
        'help':'Names of input features (can be computed from data) [default None]'}, 
    'split-test': {'abbr':'split', 'default':DEF_SPLIT_TEST, 'type':str,
        'help':'Fraction in (0,1) of data samples to split from training data' +
            ' for testing [default: {}]'.format(str(DEF_SPLIT_TEST))}, 
    'train_random_n': {'abbr':'train_rand', 'default':DEF_TRAIN_RAND, 'type':int,
        'help':'Subset random n rows from training data to use for training ' + 
            '[default: {}]'.format(str(DEF_TRAIN_RAND))}, 
    'train_first_n': {'abbr':'train_first', 'default':DEF_TRAIN_FIRST, 'type':int,
        'help':'Subset first n rows from training data to use for training ' + 
            '[default: {}]'.format(str(DEF_TRAIN_FIRST))},
    'train_uniform_n': {'abbr':'train_unif', 'default':DEF_TRAIN_UNIF, 'type':int,
        'help':'Subset random n rows from training data with close to uniform ' + 
            'distribution to use for training [default: {}]'.format(str(DEF_TRAIN_UNIF))}, 
    'sample_weights': {'abbr':'sw', 'default':DEF_SAMPLE_WEIGHTS, 'type':int,
        'help':'Sample weights to be used during model training; ' +
            'weights are defined as function of the response ' + 
            '(or average of the responses) in that sample; ' +
            'currently only power functions x^n are supported ' +
            'and the option value defines its exponent, n; when ' +
            'n is negative, the final weight is defined as 1 - x^n;' +
            ' [default: {}]'.format(DEF_SAMPLE_WEIGHTS)},
    'new_data': {'abbr':'new_data', 'default':None, 'type':str,
        'help':'Path excluding the .csv suffix to new data file [default: None]'}}

# args parser to which some of the arguments are added explicitly in a regular way
# and in addition it adds additional arguments from args_dict defined elsewhere;
# As of now args_dict includes model training hyperparameters from ML packages
# sklearm caret, keras --keras_dict, sklearn_dict, caret_dict, as well as dict
# introducing data related parameters / arguments data_params_dict defined abve.
def args_dict_parse(argv, args_dict):
    parser = argparse.ArgumentParser(prog=argv[0])
    def str_to_bool(value):
        if value.lower() in {'false', 'f', '0', 'no', 'n'}:
            return False
        elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
            return True
        raise ValueError(f'{value} is not a valid boolean value')
    parser.add_argument('-data', '--labeled_data', metavar='DATA', type=str,
                        help='Path excluding the .csv suffix to input training data ' +
                        'file containing labels')
    parser.add_argument('-model', '--model', metavar='MODEL', type=str,
                        help='Type of model to train (NN, Poly, ... ')
    parser.add_argument('-mode', '--analytics_mode', type=str, default=None,
                   help='What kind of analysis should be performed '+
                        'the supported modes are train and prediction ' +
                        '[default: train]')
    parser.add_argument('-scale', '--scale_data', type=str_to_bool, default=True,
                   help='Should features and reponses be scaled to [0,1] in input data?'+
                        '[default: True]')
    parser.add_argument('-plots', '--interactive_plots', type=str_to_bool, default=True,
                   help='Should plots be displayed interactively (or only saved)?'+
                        '[default: True]')
    # TODO !!!: make sure seed is recorded in persisteny -- for debugging
    parser.add_argument('-seed', '--seed', type=int, default=None,
                   help='Initial random seed')
    parser.add_argument('-pref', '--run_prefix', type=str, default=None,
                   help='String to be used as prefix for the output files '+
                        '[default: empty string]')
    parser.add_argument('-out_dir', '--output_directory', type=str, default=None,
                   help='Output directory where all reports and output files will be written '+
                        '[default: the same directory from which data is loaded]')
    for p, v in args_dict.items():
        #print(p, v)
        parser.add_argument('-'+v['abbr'], '--'+p, default=v['default'], 
                            type=v['type'], help=v['help'])
    args = parser.parse_args(argv[1:])
    return args

# Split input data into training and test, subset / resample from these
# training and test data that will actually be used for training and validation.
# execute training and prediction on training test, the entire input data
# and on new data (new_data) if that is also provided along with data for training.
def main(argv):
    args_dict = keras_dict | sklearn_dict | caret_dict | data_params_dict
    args = args_dict_parse(argv, args_dict)
    print('params', vars(args))
    # data related args
    inst = DataFileInstance(args.labeled_data, args.run_prefix, args.output_directory, args.new_data) 
    
    if args.response is None:
        raise Exception('Response names should be provided')
    resp_names = args.response.split(',')

    feat_names = args.features.split(',') if not args.features is None else None
        
    split_test = DEF_SPLIT_TEST if args.split_test is None else args.split_test

    X, y, X_train, y_train, X_test, y_test, mm_scaler_feat, mm_scaler_resp, levels_dict = prepare_data_for_modeling(
        inst.data_fname, True, split_test, feat_names, resp_names, inst._filename_prefix, 
        args.train_first_n, args.train_random_n, args.train_uniform_n, args.interactive_plots, 
        args.scale_data)
    
    feat_names = X_train.columns.tolist()
    
    X_train_proc = X_train #.copy()
    X_test_proc = X_test #.copy()
    # saving the column min/max info into json file to be able to scale model prediction
    # results back to the original scale of the responses. The information in this file
    # is essetially the same as that avilable within mm_scaler but is easier to consume.
    with open(inst.data_bounds_file, 'w') as f:
        json.dump({
            col: { 'min': mm_scaler_feat.data_min_[i], 'max': mm_scaler_feat.data_max_[i] }
            for i,col in enumerate(feat_names) } |
            {col: { 'min': mm_scaler_resp.data_min_[i], 'max': mm_scaler_resp.data_max_[i] }
            for i,col in enumerate(resp_names)
        }, f, indent='\t', cls=np_JSONEncoder)
        
    # run model training and prediction
    if args.model in SMLP_KERAS_MODELS:
        hparam_dict = dict((k, vars(args)[k]) for k in keras_dict.keys())
    elif args.model in SMLP_SKLEARN_MODELS:
        hparam_dict = dict((k, vars(args)[k]) for k in sklearn_dict.keys())
    elif args.model in SMLP_CARET_MODELS:
        hparam_dict = dict((k, vars(args)[k]) for k in caret_dict.keys())
    else:
        raise Exception('Unsupprted model training algp ' + str(args.model))
    #args.run_prefix
    model = model_train(inst, feat_names, resp_names, args.model, X_train, X_test, y_train, y_test,
        hparam_dict, args.interactive_plots, args.seed, args.sample_weights, True, data=None)
    if args.model == 'poly_sklearn':
        model, poly_reg, X_train, X_test = model

    print('\nPREDICT ON TRAINING DATA')
    y_train_pred = model_predict(inst, model, X_train, y_train, resp_names, args.model, 'train', args.interactive_plots)
    report_prediction_results(inst, args.model, resp_names, y_train, y_train_pred, True, mm_scaler_resp, #X_train_proc
        args.interactive_plots, 'train')
    
    print('\nPREDICT ON TEST DATA')
    y_test_pred = model_predict(inst, model, X_test, y_test, resp_names, args.model, 'test', args.interactive_plots)
    report_prediction_results(inst, args.model, resp_names, y_test, y_test_pred, True, mm_scaler_resp, #X_test_proc, 
        args.interactive_plots, 'test')
    
    print('\nPREDICT ON LABELED (TRAINING & TEST) DATA')
    # In case a polynomial model was run, polynomial features have been added to X_train and X_test,
    # therefore we need to reconstruct X before evaluating the model on all labeled features. 
    # once X has been updated, we need to update y as well in case X_train/y_train and/or X_test/y_test
    # was modified after generating them from X,y and before feeding to training.
    # use original X vs using X_train and X_test that were generated using sampling X_tran and/or X_test
    run_on_orig_X = True         
    if run_on_orig_X and args.model == 'poly_sklearn':
        X_proc = X
        X = poly_reg.transform(X)
    elif not run_on_orig_X:
        X_proc = np.concatenate([X_train_proc, X_test_proc], axis=1)
        X = np.concatenate((X_train, X_test)); 
        y = pd.concat([y_train, y_test])
        #print(X_train.shape, X_test.shape, X.shape, y.shape);
    else:
        X_proc = X
    y_pred = model_predict(inst, model, X, y, resp_names, args.model, 'labeled', args.interactive_plots)
    report_prediction_results(inst, args.model, resp_names, y, y_pred, True, mm_scaler_resp, #X_proc, 
        args.interactive_plots, 'labeled')
    
    if not args.new_data is None:
        print('\nPREDICT ON NEW DATA')
        X_new, y_new = prepare_data_for_modeling(
            inst.new_data_fname, False, None, feat_names, resp_names, inst._filename_prefix, 
            None, None, None, args.interactive_plots, args.scale_data, mm_scaler_feat, mm_scaler_resp, levels_dict)
        X_new_proc = X_new
        if args.model == 'poly_sklearn':
            X_new = poly_reg.transform(X_new)
        y_new_pred = model_predict(inst, model, X_new, y_new, resp_names, args.model, 'new', args.interactive_plots)
        report_prediction_results(inst, args.model, resp_names, y_new, y_new_pred, True, mm_scaler_resp, #X_new_proc, 
            args.interactive_plots, 'new')
        return
        try:
            X_new, y_new = prepare_data_for_modeling(
                inst.new_data_fname, False, None, feat_names, resp_names, inst._filename_prefix, 
                None, None, None, args.interactive_plots, args.scale_data, mm_scaler_feat, mm_scaler_res, levels_dict)
            X_new_proc = X_new
            if args.model == 'poly_sklearn':
                X_new = poly_reg.transform(X_new)
            y_new_pred = model_predict(inst, model, X_new, y_new, resp_names, args.model, 'new', args.interactive_plots)
            report_prediction_results(inst, args.model, resp_names, y_new, y_new_pred, True, mm_scaler_resp, #X_new_proc, 
                args.interactive_plots, 'new')
        except Exception as error:
            print('Error occured during prediction on new data:\n' + str(error))
            error_file = open(inst.error_file, "wt")
            error_file.write('Error occured during prediction on new data:\n' + str(error))
            error_file.close()
            
            
    
if __name__ == "__main__":
    main(sys.argv)
