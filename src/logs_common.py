#
# This file is part of smlprover.
#
# Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
# See the LICENSE file for terms of distribution.

# TODO !!!:  inst actually contains info required for logging, so maybe all feilds and paths/filenames
# definitions need to be moved into a logger module/class (which does not exist now).

import os, sys, json
import logging


class DataFileInstance:
    # data_file_prefix is the name of the data file including the full path to
    #                  the file but excluding the .csv suffix
    # run_prefix is a uique identifier string for each run and is supposed to be
    #            used as a prefix of the reports/files written during the run
    # _filename_prefix is a string including the output directory and is supposed 
    #                  to be used as a path + prefix of all reports produced by a run
    # _out_dir defines directory where all reports / output files will be written.
    #          When it is not provided, the directory from which data was loaded
    #          will be used as the output directory
    # _new_data is path and filename of new data (unseen during training) w/o .csv suffix.
    def __init__(self, data_file_prefix : str, run_prefix : str, output_directory : str=None, 
            new_data_file_prefix=None):
        self._dir, self._pre = os.path.split(data_file_prefix)
        self._run_prefix = run_prefix
        self._out_dir = output_directory
        self._new_data = new_data_file_prefix
        
        if not self._out_dir is None:
            self._filename_prefix = os.path.join(self._out_dir, self._run_prefix + '_' + self._pre)
        else:
            self._filename_prefix = os.path.join(self._dir, self._run_prefix + '_' + self._pre)
        # if new_data is not None, its name is added to self._filename_prefix
        if not self._new_data is None:
            _, new_data_fname = os.path.split(self._new_data)
            self._filename_prefix = self._filename_prefix + '_' + new_data_fname

    # input/training data file name (including the directory path)
    @property
    def data_fname(self):
        return os.path.join(self._dir, self._pre + ".csv")
    
    # new (unseen during training) data file name (including the directory path)
    @property
    def new_data_fname(self):
        return self._new_data + ".csv"
    
    # filename with full path for logging verbosity and events during execution
    @property
    def log_file(self):
        return self._filename_prefix + ".txt"
    
    # filename with full path for logging error message before aborting
    @property
    def error_file(self):
        return self._filename_prefix + "_error.txt"
    
    # Saved neural network model in a specific tensorflow format (weights, metadata, etc.).
    # Generated using Keras model.save utility
    # The model can be also partially trained, and training can resume from that point.
    @property
    def model_file(self):
        return self._filename_prefix + "_model_complete.h5"

    # file to save NN Keras/tensorflow training / error convergence info, known as checkpoints
    @property
    def model_checkpoint_pattern(self):
        return self._filename_prefix + "_model_checkpoint.h5"

    # Currently unused -- genrated in train_keras.py but not consumed.
    # Saved neural network model in json format -- alternative to info saved in model_file).
    # Generated using Keras model.to_json() utility (alternative to Keras model.save utility)
    # The model can be also partially trained, and training can resume from that point.
    @property
    def model_config_file(self):
        return self._filename_prefix + "_model_config.json"

    # min/max info of all columns (features and reponses) in input data
    # (labeled data used for training and testing ML models)
    @property
    def data_bounds_file(self):
        return self._filename_prefix + "_data_bounds.json"

    # TODO !!!: add description
    @property
    def model_gen_file(self):
        return self._filename_prefix + "_model_gen.json"
    
    # required for generating file names of the reports containing model prediction reults;
    # might cover multiple models (algorithms like NN, DT, RF) as well as multiple responses
    def predictions_summary_filename(self, data_version):
        return self._filename_prefix + '_' + data_version + "_predictions_summary.csv"
    
    # required for generating file names of the reports containing model precision (rmse, R2, etc.) reults;
    # might cover multiple models (algorithms like NN, DT, RF) as well as multiple responses
    def prediction_precisions_filename(self, data_version):
        return self._filename_prefix + '_' + data_version + "_prediction_precisions.csv"
    

    
# LOGGER sources:
# https://docs.python.org/3/howto/logging.html
# https://stackoverflow.com/questions/29087297/is-there-a-way-to-change-the-filemode-for-a-logger-object-that-is-not-configured/29087645#29087645
# TODO !!!: create SmlpLogger class? class SmlpLogger:
# create python logger object with desired configuration  
def create_logger(logger_name, log_file, log_level, log_mode, log_time):
    # create logger for an application called logger_name
    logger = logging.getLogger(logger_name)
    
    def log_level_to_level_object(level_str):
        if level_str == 'critical':
            return logging.CRITICAL
        elif level_str == 'error':
            return logging.ERROR
        elif level_str == 'warning':
            return logging.WARNING
        elif level_str == 'info':
            return logging.INFO
        elif level_str == 'debug':
            return logging.DEBUG
        elif level_str == 'notset':
            return logging.NOTSET
        else:
            raise Exception('Unsupported logging level {}'.format(log_level))
    
    log_level_object = log_level_to_level_object(log_level)
    # set the logging level 
    logger.setLevel(log_level_object)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_file, mode=log_mode)
    fh.setLevel(log_level_object)
    
    # create console handler with a higher log level
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(log_level_object)
    
    # create formatter and add it to the handlers
    if log_time:
        formatter = logging.Formatter('\n%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        formatter = logging.Formatter('\n%(name)s - %(levelname)s - %(message)s')
    #formatter = logging.Formatter('[%(asctime)s] %(levelname)8s --- %(message)s ' +
    #                             '(%(filename)s:%(lineno)s)',datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # add the handlers to the logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

DEF_LOGGER_LEVEL = 'warning'
DEF_LOGGER_FMODE = 'w'
DEF_LOGGER_TIME = 'true'

# function to be applied to args options that are intended to be Boolean
# but are specified as strings
def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')
        
logger_params_dict = {
    'log_level': {'abbr':'log_level', 'default':'info', 'type':str,
        'help':'The logger level or severity of the events they are used to track. The standard levels ' + 
            'are (in increasing order of severity): notset, debug, info, warning, error, critical; ' +
            'only events of this level and above will be tracked [default {}]'.format(DEF_LOGGER_LEVEL)}, 
    'log_mode': {'abbr':'log_mode', 'default':'w', 'type':str,
        'help':'The logger filemode for logging into log file [default {}]'.format(DEF_LOGGER_FMODE)},
    'log_time': {'abbr':'log_time', 'default':DEF_LOGGER_TIME, 'type':str_to_bool,
        'help':'Should time stamp be logged along with every message issued by logger [default {}]'.format(DEF_LOGGER_TIME)}}

