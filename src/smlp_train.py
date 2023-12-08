#!/usr/bin/env python3
#
# This file is part of smlprover.
#
# Copyright 2019 Konstantin Korovin
# Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
# See the LICENSE file for terms of distribution.

# coding: utf-8

import os, sys, argparse #json, 
import pandas as pd


# /usr/intel/pkgs/gcc/8.2.0/lib64
from local_paths import SMLP_BOOST_PATH, SMLP_PYTHONPATH
print('setenv LD_LIBRARY_PATH')
#os.environ['LD_LIBRARY_PATH'] = SMLP_BOOST_PATH
#os.system("export LD_LIBRARY_PATH={boost_path}".format(boost_path=SMLP_BOOST_PATH)
print('import smlp')
print(os.getenv('LD_LIBRARY_PATH')); 
print('DONE')
#assert False

import smlp
print(smlp.__path__); 
#assert False


# imports from SMLP modules
from smlp_py.logs_common import DataFileInstance, SmlpLogger
from smlp_py.utils_common import str_to_bool #np_JSONEncoder, 
from smlp_py.train_common import ModelCommon
from smlp_py.data_common import DataCommon

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
        if 'default' in v:
            parser.add_argument('-'+v['abbr'], '--'+p, default=v['default'], 
                                type=v['type'], help=v['help'])
        else:
            parser.add_argument('-'+v['abbr'], '--'+p, type=v['type'], help=v['help'])
    
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
    inst = DataFileInstance(args.labeled_data, args.run_prefix, args.output_directory, args.new_data, args.model_name) 
    
    # logs.
    # TODO !!!: for now logger is independent from class inst; could be made part of it (any advantages ???)
    logger = loggerInst.create_logger('smlp_logger', inst.log_file, args.log_level, args.log_mode, args.log_time)
    logger.info('Executing smlp_train.py script: Start')
    logger.info('Params\n: {}'.format(vars(args)))
    dataInst.set_logger(logger)
    dataInst.set_paths(inst)
    modelInst.set_logger(logger)
    
    # extract response and feature names
    if args.response is None:
        raise Exception('Response names should be provided')
    resp_names = args.response.split(',')
    feat_names = args.features.split(',') if not args.features is None else None
            
    # prepare data for model training
    logger.info('PREPARE DATA FOR MODELING')    
    X, y, X_train, y_train, X_test, y_test, X_new, y_new, mm_scaler_feat, mm_scaler_resp, \
    levels_dict, model_features_dict = dataInst.process_data(
        inst, inst.data_fname, inst.new_data_fname, True, args.split_test, feat_names, resp_names, 
        args.train_first_n, args.train_random_n, args.train_uniform_n, args.interactive_plots, 
        args.data_scaler, args.mrmr_feat_count_for_prediction,
        args.save_model, args.use_model)
    
    # model training, validation, testing, prediction on training and labeled data as well as new data when available.    
    model = modelInst.build_models(inst, args.model, X, y, X_train, y_train, X_test, y_test, X_new, y_new,
        resp_names, mm_scaler_feat, mm_scaler_resp, levels_dict, model_features_dict, 
        modelInst.get_hyperparams_dict(args, args.model),
        args.interactive_plots, args.seed, args.sample_weights_coef, args.save_model, args.use_model, args.model_per_response)
    
    return model
    

if __name__ == "__main__":
    main(sys.argv)
