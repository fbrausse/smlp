#!/usr/bin/env python3
#
# This file is part of smlprover.
#
# Copyright 2019 Konstantin Korovin
# Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
# See the LICENSE file for terms of distribution.

# coding: utf-8

import os, sys, json, argparse
import pandas as pd

# imports from SMLP modules
from logs_common import DataFileInstance, SmlpLogger
from utils_common import np_JSONEncoder, str_to_bool
from smlp.train_common import ModelCommon
from smlp.data_common import DataCommon


# args parser to which some of the arguments are added explicitly in a regular way
# and in addition it adds additional arguments from args_dict defined elsewhere;
# As of now args_dict includes model training hyperparameters from ML packages
# sklearm caret, keras -- model_params_dict = keras_dict | sklearn_dict | caret_dict, 
# as well as data and logger related parameters: data_params_dict and logger_params_dict
def args_dict_parse(argv, args_dict):
    #print('argv', argv)
    parser = argparse.ArgumentParser(prog=argv[0])
    #print('parser', parser)
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
                   help='Should features and responses be scaled to [0,1] in input data?'+
                        '[default: True]')
    parser.add_argument('-plots', '--interactive_plots', type=str_to_bool, default=True,
                   help='Should plots be displayed interactively (or only saved)?'+
                        '[default: True]')
    parser.add_argument('-seed', '--seed', type=int, default=None,
                   help='Initial random seed')
    parser.add_argument('-pref', '--run_prefix', type=str, default=None,
                   help='String to be used as prefix for the output files '+
                        '[default: empty string]')
    parser.add_argument('-out_dir', '--output_directory', type=str, default=None,
                   help='Output directory where all reports and output files will be written '+
                        '[default: the same directory from which data is loaded]')

    for p, v in args_dict.items():
        #print('p', p, 'v', v); print('type', v['type'])
        parser.add_argument('-'+v['abbr'], '--'+p, default=v['default'], 
                            type=v['type'], help=v['help'])
    
    args = parser.parse_args(argv[1:])
    return args


# Split input data into training and test, subset / resample from these
# training and test data that will actually be used for training and validation.
# execute training and prediction on training test, the entire input data
# and on new data (new_data) if that is also provided along with data for training.
def main(argv):
    # data and model class instances
    dataInst = DataCommon() # inst.log_file, args.log_level, 'a', args.log_time
    modelInst = ModelCommon() # inst.log_file, args.log_level, 'a', args.log_time
    loggerInst = SmlpLogger() 
    
    # get args
    args_dict = modelInst.model_params_dict | dataInst.data_params_dict | loggerInst.logger_params_dict
    args = args_dict_parse(argv, args_dict)

    # eport related class -- instantiation
    # TODO !!!: rename inst to reportInst, and class DataFileInstance to ReportsInstance?
    inst = DataFileInstance(args.labeled_data, args.run_prefix, args.output_directory, args.new_data) 
    
    # logs.
    # TODO !!!: for now logger is independent from class inst; could be made part of it (any advantages ???)
    logger = loggerInst.create_logger('smlp_logger', inst.log_file, args.log_level, args.log_mode, args.log_time)
    logger.info('Executing smlp_train.py script: Start')
    logger.info('Params\n: {}'.format(vars(args)))
    dataInst.set_logger(logger)
    modelInst.set_logger(logger)
    
    
    # extract response and feature names
    if args.response is None:
        raise Exception('Response names should be provided')
    resp_names = args.response.split(',')
    feat_names = args.features.split(',') if not args.features is None else None
        
    # prepare data for model training
    logger.info('PREPARE DATA FOR MODELING')
    split_test = DEF_SPLIT_TEST if args.split_test is None else args.split_test
    X, y, X_train, y_train, X_test, y_test, mm_scaler_feat, mm_scaler_resp, levels_dict = dataInst.prepare_data_for_modeling(
        inst.data_fname, True, split_test, feat_names, resp_names, inst._filename_prefix, 
        args.train_first_n, args.train_random_n, args.train_uniform_n, args.interactive_plots, 
        args.scale_data, None, None, None)
    #print('(1)'); print('y\n', y);  print('y_train\n', y_train); print('y_test\n', y_test); 
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
        
    # run model training
    logger.info('TRAIN MODEL')
    print('args', args); print('args_model', args.model)
    hparams_dict = modelInst.get_hyperparams_dict(args, args.model); 
    model = modelInst.model_train(inst, feat_names, resp_names, args.model, X_train, X_test, y_train, y_test,
        hparams_dict, args.interactive_plots, args.seed, args.sample_weights_coef, True)
    if args.model == 'poly_sklearn':
        model, poly_reg, X_train, X_test = model
     
    
    logger.info('PREDICT ON TRAINING DATA')
    #print('(2)'); print('y\n', y);  print('y_train\n', y_train); print('y_test\n', y_test);
    y_train_pred = modelInst.model_predict(inst, model, X_train, y_train, resp_names, args.model)
    modelInst.report_prediction_results(inst, args.model, resp_names, y_train, y_train_pred, True, mm_scaler_resp, #X_train_proc
        args.interactive_plots, 'training')
     
    
    logger.info('PREDICT ON TEST DATA')
    #print('(3)'); print('y\n', y);  print('y_train\n', y_train); print('y_test\n', y_test);
    y_test_pred = modelInst.model_predict(inst, model, X_test, y_test, resp_names, args.model)
    #print('(3b)'); print('y\n', y);  print('y_train\n', y_train); print('y_test\n', y_test); 
    modelInst.report_prediction_results(inst, args.model, resp_names, y_test, y_test_pred, True, mm_scaler_resp, #X_test_proc, 
        args.interactive_plots, 'test')
    
    
    logger.info('PREDICT ON LABELED DATA')
    #print('(4)'); print('y\n', y);  print('y_train\n', y_train); print('y_test\n', y_test); 
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

    y_pred = modelInst.model_predict(inst, model, X, y, resp_names, args.model)
    modelInst.report_prediction_results(inst, args.model, resp_names, y, y_pred, True, mm_scaler_resp,
        args.interactive_plots, 'labeled')
    
    if not args.new_data is None:
        logger.info('PREDICT ON NEW DATA')
        X_new, y_new = dataInst.prepare_data_for_modeling(
            inst.new_data_fname, False, None, feat_names, resp_names, inst._filename_prefix, 
            None, None, None, args.interactive_plots, args.scale_data, mm_scaler_feat, mm_scaler_resp, levels_dict)
        X_new_proc = X_new
        if args.model == 'poly_sklearn':
            X_new = poly_reg.transform(X_new)
        y_new_pred = modelInst.model_predict(inst, model, X_new, y_new, resp_names, args.model)
        modelInst.report_prediction_results(inst, args.model, resp_names, y_new, y_new_pred, True, mm_scaler_resp, #X_new_proc, 
            args.interactive_plots, 'new')
        return
        try:
            X_new, y_new = dataInst.prepare_data_for_modeling(
                inst.new_data_fname, False, None, feat_names, resp_names, inst._filename_prefix, 
                None, None, None, args.interactive_plots, args.scale_data, mm_scaler_feat, mm_scaler_res, levels_dict)
            X_new_proc = X_new
            if args.model == 'poly_sklearn':
                X_new = poly_reg.transform(X_new)
            y_new_pred = modelInst.model_predict(inst, model, X_new, y_new, resp_names, args.model)
            modelInst.report_prediction_results(inst, args.model, resp_names, y_new, y_new_pred, True, mm_scaler_resp, #X_new_proc, 
                args.interactive_plots, 'new')
        except Exception as error:
            logger.info('Error occured during prediction on new data:\n' + str(error))
            error_file = open(inst.error_file, "wt")
            error_file.write('Error occured during prediction on new data:\n' + str(error))
            error_file.close()

    logger.info('Executing smlp_train.py script: End')


if __name__ == "__main__":
    main(sys.argv)
