#!/usr/bin/env python3
#
# This file is part of smlprover.
#
# Copyright 2019 Konstantin Korovin
# Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
# See the LICENSE file for terms of distribution.

# coding: utf-8

import os, sys, argparse
import tensorflow as tf
import pandas as pd
from smlp.train_keras import (keras_main, get_nn_hparam_default, SMLP_KERAS_MODELS)
                              #DEF_SPLIT_TEST, DEF_OPTIMIZER, DEF_EPOCHS,
                              #HID_ACTIVATION, OUT_ACTIVATION, DEF_BATCH_SIZE, DEF_LOSS, 
                              #DEF_METRICS, DEF_MONITOR, DEF_SCALER, )
from smlp.smlp_plot import *
from smlp.train_caret import (caret_main, SMLP_CARET_MODELS)
from smlp.train_sklearn import (sklearn_main, SMLP_SKLEARN_MODELS)
from smlp.train_common import (model_predict)
from pandas import read_csv, DataFrame
from sklearn.model_selection import train_test_split
from common import *
import json
import time

# global defaults for SMLP
DEF_CV_FOLDS = 0
DEF_TRAIN_FIRST = 0
DEF_TRAIN_RAND = 0
DEF_TRAIN_UNIF = 0
DEF_SCALER = 'min-max'  # options: 'min-max', 'max-abs'
DEF_SPLIT_TEST = 0.2


def parse_args(argv):
    def str_to_bool(value):
        if value.lower() in {'false', 'f', '0', 'no', 'n'}:
            return False
        elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
            return True
        raise ValueError(f'{value} is not a valid boolean value')
    
    p = argparse.ArgumentParser(prog=argv[0])
    p.add_argument('data', metavar='DATA', type=str,
                   help='Path excluding the .csv suffix to input training data ' +
                        'file containing labels')
    p.add_argument('model', metavar='MODEL', type=str,
                   help='Type of model to train (NN, Poly, ... ')    
    p.add_argument('-a', '--nn_activation', 
                   default=get_nn_hparam_default('nn_activation'), 
                   metavar='ACT', #HID_ACTIVATION
                   help='activation for NN [default: not exposed]')
    p.add_argument('-b', '--nn_batch_size', type=int, metavar='BATCH', 
                   default=get_nn_hparam_default('nn_batch_size'), #DEF_BATCH_SIZE
                   help='batch_size for NN' +
                        ' [default: not exposed]')
    p.add_argument('-B', '--bounds', type=str, default=None,
                   help='Path to pre-computed bounds.csv')
    p.add_argument('-c', '--chkpt', type=str, 
                   default=None, #get_nn_hparam_default('nn_batch_size'), 
                   nargs='?', const=(),
                   help='save model checkpoints after each epoch; ' +
                        'optionally use CHKPT as path, can contain named ' +
                        'formatting options "{ID:FMT}" where ID is one of: ' +
                        "'epoch', 'acc', 'loss', 'val_loss'; if these are " +
                        "missing only the best model will be saved" +
                        ' [default: no, otherwise if CHKPT is missing: ' +
                        'model_checkpoint_DATA.h5]')
    p.add_argument('-d', '--poly_degree', type=int, default=2,
                   metavar='DEGREE',
                   help='Degree of the polynomial to train [default: ' + str(2) + ']')
    p.add_argument('-data_new', '--data_new', default=None, metavar='DATA_NEW', type=str,
                   help='Path excluding the .csv suffix to new data file' +
                        ' [default: None]')
    p.add_argument('-e', '--nn_epochs', type=int, 
                   default=get_nn_hparam_default('nn_epochs'), #DEF_EPOCHS
                   metavar='EPOCHS',
                   help='epochs for NN [default: not exposed]')
    p.add_argument('-f', '--filter', type=float, default=0.0,
                   help='filter data set to rows satisfying RESPONSE >= quantile(FILTER) '+
                        '[default: no]')
    p.add_argument('-folds', '--cv_folds', type=int, default=DEF_CV_FOLDS,
                   metavar='CV_FOLDS',
                   help='cross-validation folds [default: ' + str(DEF_CV_FOLDS) + ']')
    p.add_argument('-l', '--nn_layers', 
                   default=get_nn_hparam_default('nn_layers'), metavar='LAYERS',
                   help='specify number and sizes of the hidden layers of the '+
                        'NN as non-empty comma-separated list of positive '+
                        'fractions in the number of input features in, e.g. '+
                        'second of half input size, third of quarter input size; '+
                        '[default: 1 (one hidden layer of size exactly '+
                        '#input-features)]')
    p.add_argument('-nn_hid', '--nn_hid_activation', 
                   default=get_nn_hparam_default('nn_hid_activation'), 
                   metavar='ACT', #HID_ACTIVATION
                   help='activation for NN [default: not exposed]')
    p.add_argument('-nn_out', '--nn_out_activation', 
                   default=get_nn_hparam_default('nn_out_activation'), 
                   metavar='ACT', #HID_ACTIVATION
                   help='activation for NN [default: not exposed]')
    p.add_argument('-o', '--nn_optimizer', 
                   default=get_nn_hparam_default('nn_optimizer'), 
                   metavar='OPT', #DEF_OPTIMIZER
                   help='optimizer for NN [default: not exposed]')
    p.add_argument('-O', '--objective', type=str, default=None,
                   help='Objective function in terms of labelled outputs '+
                        '[default: RESPONSE if it is a single variable]')
    p.add_argument('-plots', '--interactive_plots', type=str_to_bool, default=True,
                   help='Should plots be displayed interactively (or only saved)?'+
                        '[default: True]')
    p.add_argument('-pref', '--run_prefix', type=str, default=None,
                   help='String to be used as prefix for the output files '+
                        '[default: empty string]')
    p.add_argument('-p', '--preprocess', default=[DEF_SCALER], metavar='PP',
                   action='append',
                   help='preprocess data using "std, "min-max", "max-abs" or "none" '+
                        'scalers. PP can optionally contain a prefix "F=" where '+
                        'F denotes a feature of the input data by column index '+
                        '(0-based) or by column header. If the prefix is absent, '+
                        'the selected scaler will be applied to all features. '+
                        'This parameter can be given multiple times. '+
                        '[default: '+DEF_SCALER+']')
    p.add_argument('-r', '--response', type=str,
                   help='comma-separated names of the response variables '+
                        '[default: taken from SPEC, where "type" is "response"]')
    p.add_argument('-R', '--seed', type=int, default=None,
                   help='Initial random seed')
    p.add_argument('-s', '--spec', type=str, required=True,
                   help='.spec file')
    p.add_argument('-S', '--split-test', type=float, default=DEF_SPLIT_TEST,
                   metavar='SPLIT',
                   help='Fraction in (0,1) of data samples to split ' +
                        'from training data for testing' +
                        ' [default: ' + str(DEF_SPLIT_TEST) + ']')
    p.add_argument('-train_first', '--train_first_n', type=int, default=DEF_TRAIN_FIRST,
                   metavar='TRAIN_FIRST',
                   help='Subset first n rows from training data to use for training ' + 
                        '[default: ' + str(DEF_TRAIN_FIRST) + ']')
    p.add_argument('-train_rand', '--train_random_n', type=int, default=DEF_TRAIN_RAND,
                   metavar='TRAIN_RAND',
                   help='Subset random n rows from training data to use for training ' + 
                        '[default: ' + str(DEF_TRAIN_RAND) + ']')
    p.add_argument('-train_unif', '--train_uniform_n', type=int, default=DEF_TRAIN_UNIF,
                   metavar='TRAIN_UNIF',
                   help='Subset random n rows from training data with close ' + 
                         'to uniform distrebution to use for training ' + 
                        '[default: ' + str(DEF_TRAIN_RAND) + ']')
    p.add_argument('-w', '--sample_weights', type=int, default=0,
                   help='Sample weights to be used during model training; ' +
                        'weights are defined as function of the response ' + 
                        '(or average of the responses) in that sample; ' +
                        'currently only power functions x^n are suppoerted ' +
                        'and the option value defines its exponent, n; when ' +
                        'n is negative, the final weight is defined as 1 - x^n;' +
                        ' [default: ' + str(0) + ']')
    args = p.parse_args(argv[1:])
    if args.objective is None and len(args.response.split(',')) == 1:
        args.objective = args.response
    return args


def main(argv):
    args = parse_args(argv)
    print('params', vars(args))
    # data related args
    inst = DataFileInstance(args.data); print(inst._dir, inst._out_prefix); 
    
    '''
    chkpt = args.chkpt
    if chkpt == () and args.model == 'NN':
        chkpt = inst.model_checkpoint_pattern
    '''
    with open(args.spec, 'r') as f:
        spec = json.load(f)
    if args.response is None:
        resp_names = [s['label'] for s in spec if s['type'] == 'response']
    else:
        resp_names = args.response.split(',')
        for n in resp_names:
            if not any(n == s['label'] for s in spec):
                print("error: response '%s' is not a response in spec" % n,
                      file=sys.stderr)
                sys.exit(1)
        resp_names = [s['label'] for s in spec if s['label'] in resp_names]
    if not resp_names:
        print('error: no response names', file=sys.stderr)
        sys.exit(1)

    input_names = get_input_names(spec)

    bnds = None if args.bounds is None else read_csv(args.bounds, index_col=0)

    # Most of this block is required for all models; taken from train.py (not form train-nn.py)
    seed = args.seed
    objective = args.objective
    quantile_filter = args.filter 
    split_test = DEF_SPLIT_TEST if args.split_test is None else args.split_test

    #if data is None:
    print('loading data')
    data = read_csv(inst.data_fname)
    print(data.describe())
    #plot_data_columns(data)

    print(data)
    if bnds is None:
        mm = MinMaxScaler()
        mm.fit(data)
    else:
        mm = scaler_from_bounds(({'label': c} for c in data), bnds)

    # remember responses, objectives and bounds for reproducibility
    # COMMON 
    persist = {
        'response': resp_names,
        'objective': objective,
        'obj-bounds': (lambda s: {
            'min': s.data_min_[0],
            'max': s.data_max_[0],
        })(MinMaxScaler().fit(DataFrame(data.eval(objective)))) if objective is not None else None,
    }

    '''
    #add training parameters for all supported engines
    # SPECIFIC to algo
    if args.model == 'nn_keras':
        persist['train'] = {
            'nn_epochs': args.nn_epochs,
            'batch-size': args.nn_batch_size,
            'optimizer': args.nn_optimizer,
            'activation': args.nn_activation,
            'split-test': args.split_test,
            'seed': args.seed,
        }
    elif args.model in SMLP_SKLEARN_MODELS:
        persist['train'] = {
            'poly_degree': args.poly_degree,
        }
    elif args.model in SMLP_CARET_MODELS:
        persist['train'] = {
            'cv_folds': args.cv_folds,
        }
    elif not args.model in ['dt_sklearn', 'et_sklearn', 'rf_sklearn']:
        raise Exception('Unsupported model ' + str(args.model) + ' in SMLP')
    '''
    # COMMON -- used for scaling and later on by solver to reconstrct original values
    with open(inst.data_bounds_file, 'w') as f:
        json.dump({
            col: { 'min': mm.data_min_[i], 'max': mm.data_max_[i] }
            for i,col in enumerate(data.columns)
        }, f, indent='\t', cls=np_JSONEncoder)

    # DEVELOPER use only, likely will be dropped
    if quantile_filter > 0:
        if len(resp_names) > 1:
            assert objective is not None
        obj = data.eval(objective)
        data = data[obj >= obj.quantile(quantile_filter)]
        persist['filter'] = { 'quantile': quantile_filter }
        del obj

    # normalize data
    # Data normalization -- COMMON
    data = DataFrame(mm.transform(data), columns=data.columns)
    persist['pp'] = {
        'response': 'min-max',
        'features': 'min-max',
    }

    print('normalized data')
    print(data.describe())
    # COMMON
    with open(inst.model_gen_file, 'w') as f:
        json.dump(persist, f)

    # COMMON
    X, y = get_response_features(data, input_names, resp_names)
    print('normalized training data\n', X)

    print("y.shape", y.shape)
    #print("y.min:", y.min())
    #print("y.max", y.max())
    print('scaled y\n', y)
    #plot_data_columns(DataFrame(y, columns=resp_names))
    # COMMON
    distplot_dataframe(inst, y, resp_names, args.interactive_plots)

    #y_0_9=np.argwhere(y>0.9) # get indecies
    #print('y[y>0.9]', y[[n>0.9 for n ]])
    #plot_data_columns(y)

    #scaler_std = StandardScaler()
    #X=scaler_std.fit_transform(X)
    #print(X)
    # split data inti training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=split_test,
                                                        shuffle=True,
                                                        random_state=17)
    print(os.path.join(inst._dir, inst._out_prefix))
    #y_test = None
    # make the training data small for purpose of quicker code development, improve distribution 
    # (make it close to uniform with random and uniform sampling with replacement), control training data size
    assert not (args.train_first_n >= 1 and args.train_random_n >= 1)
    assert not (args.train_first_n >= 1 and args.train_uniform_n >= 1)
    assert not (args.train_random_n >= 1 and args.train_uniform_n >= 1)
    if args.train_first_n >= 1:
        to_subset = nm.min(X_train.shape[0], args.train_first_n)
        X_train = X_train.iloc[:to_subset]
        y_train = y_train.iloc[:to_subset]
    elif args.train_random_n >= 1:
        # select with replacement only if we want more samples than available in X_train
        selct_with_replacement = args.train_random_n > X_train.shape[0]
        X_train = X_train.sample(args.train_random_n, replace=selct_with_replacement)
        y_train = y_train[y_train.index.isin(X_train.index)]
        y_train = y_train.reindex(X_train.index)
        #print(X_train.iloc[44832]) ; print(y_train.iloc[44832])
        # reset index in case of selection with replacement in order to ensure uniquness of indices
        if selct_with_replacement:
            X_train.reset_index(inplace=True, drop=True)
            y_train.reset_index(inplace=True, drop=True)
            #print(X_train.iloc[44832]) ; print(y_train.iloc[44832])        
    elif args.train_uniform_n >= 1:
        # select rows from X_train and y_train with repitition to acheive uniform destribution of 
        # values of y_train in the resumpled training data.
        uniform_n_01 = np.random.uniform(low=0.0, high=1.0, size=args.train_uniform_n) 
        filter_train_samples = [(y_train[resp_names[0]] - v).abs().idxmin() for v in uniform_n_01]
        # takes nearly the same time: filter_train_samples = list(map(select_closest_row, np.array(uniform_n_01)))
        print('y_train', y_train.shape)
        # .loc[] is required to sample exactly len(filter_train_samples) with replacement
        # cannot use .iloc[] because the indices are not continuous from 0 to k -- [0:k].
        # cannot use .isin() because it will not perform selection with replacement.
        y_train = y_train.loc[filter_train_samples];  print('y_train', y_train.shape)
        X_train = X_train.loc[filter_train_samples];  print('X_train', X_train.shape)
        print('y_train after uniform sampling', y_train.shape)
        # reset index in case of selection with replacement in order to ensure uniquness of indices
        X_train.reset_index(inplace=True, drop=True); print('X_train', X_train.shape)
        y_train.reset_index(inplace=True, drop=True); print('y_train', y_train.shape)
        
        
    # for experirmentation: in case we want to see how model performs on a subrange 
    # of the the domain of y, say on samples where y > 0.9 (high values in y)
    if True and not y_test is None:
        #print(y_test.head()); print(X_test.head())
        filter_test_samples = y_test[resp_names[0]] > 0.9
        y_test = y_test[filter_test_samples]; #print('y_test with y_test > 0.9', y_test.shape); print(y_test.head())
        X_test = X_test[filter_test_samples]; #print('X_test with y_test > 0.9', X_test.shape); print(X_test.head())

    
    # run model training and prediction
    if args.model == 'nn_keras':
        keras_algo = args.model[:-len('_keras')]
        nn_keras_hparam_dict = {'chkpt': args.chkpt, 'nn_layers': args.nn_layers, 'seed': args.seed, 
            'nn_epochs': args.nn_epochs, 'nn_batch_size': args.nn_batch_size, 'nn_optimizer': args.nn_optimizer,
            'nn_out_activation': args.nn_out_activation, 'nn_hid_activation': args.nn_hid_activation,
            'nn_activation': args.nn_activation, 'sample_weights': args.sample_weights}
        model = keras_main(inst, input_names, resp_names, keras_algo,
            X_train, X_test, y_train, y_test, args.run_prefix, args.interactive_plots, nn_keras_hparam_dict,
            True, False, None)
        '''
            #args.chkpt, 
            args.nn_layers, #args.preprocess,
            #args.seed, #args.filter, args.objective, bnds,
            #args.nn_epochs, args.nn_batch_size, args.nn_optimizer,
            #args.nn_activation, None, args.sample_weights
        '''
            
    elif args.model in ['dt_sklearn', 'et_sklearn', 'rf_sklearn', 'poly_sklearn']:
        sklearn_algo = args.model[:-len('_sklearn')]
        if args.model in ['dt_sklearn', 'et_sklearn', 'rf_sklearn']:
            model = sklearn_main(inst, input_names, resp_names, sklearn_algo,
                X_train, X_test, y_train, y_test, args.run_prefix, args.interactive_plots, args.poly_degree, 
                args.sample_weights, True)
        else:
            model, poly_reg, X_train, X_test = sklearn_main(inst, input_names, resp_names, sklearn_algo,
                X_train, X_test, y_train, y_test, args.run_prefix, args.interactive_plots, args.poly_degree, 
                args.sample_weights, True)
    elif args.model in SMLP_CARET_MODELS:
        caret_algo = args.model[:-len('_caret')]
        model = caret_main(inst, input_names, resp_names, caret_algo,
            X_train, X_test, y_train, y_test, args.run_prefix, args.interactive_plots, args.cv_folds, 
            args.sample_weights, False, True)       
    else:
        raise Exception('Unsuported algorithm ' + str(args.model))        

    if True:
        print('\nPREDICT ON TRAINING DATA')
        model_predict(inst, model, X_train, y_train, resp_names, args.model, 'train', args.interactive_plots)
        
        print('\nPREDICT ON TEST DATA')
        model_predict(inst, model, X_test, y_test, resp_names, args.model, 'test', args.interactive_plots)
        
        print('\nPREDICT ON LABELED (TRAINING & TEST) DATA')
        # In case a polynomial model was run, polynomial features have been added to X_train and X_test,
        # therefore we need to reconstruct X before evaluating the model on all labeled features. 
        # once X has been updated, we need to update y as well in case X_train/y_train and/or X_test/y_test
        # was modified after generating them from X,y and before feeding to training.
        # use original X vs using X_train and X_test that were generated using sampling X_tran and/or X_test
        run_on_orig_X = True         
        if run_on_orig_X and args.model == 'poly_sklearn':
            X = poly_reg.transform(X)
        elif not run_on_orig_X:
            X = np.concatenate((X_train, X_test)); 
            y = pd.concat([y_train, y_test])
            print(X_train.shape, X_test.shape, X.shape, y.shape);
        model_predict(inst, model, X, y, resp_names, args.model, 'labeled', args.interactive_plots)
        
    if not args.data_new is None:
        print('\nPREDICT ON NEW DATA')
        data_new = read_csv(os.path.join(inst._dir, args.data_new+'.csv'))
        data_new = data_new[data.columns.tolist()]
        data_new = DataFrame(mm.transform(data_new), columns=data.columns)
        X_new, y_new = get_response_features(data_new, input_names, resp_names)
        if args.model == 'poly_sklearn':
            X_new = poly_reg.transform(X_new)
        model_predict(inst, model, X_new, y_new, resp_names, args.model, 'new', args.interactive_plots)
    
    
if __name__ == "__main__":
    main(sys.argv)
