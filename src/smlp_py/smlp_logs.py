#
# This file is part of smlprover.
#
# Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
# See the LICENSE file for terms of distribution.

# TODO !!!:  inst actually contains info required for logging, so maybe all feilds and paths/filenames
# definitions need to be moved into a logger module/class (which does not exist now).
# TODO !!! code to create argparse from json cnfig file https://gist.github.com/matthewfeickert/3b7d30e408fe4002aac728fc911ced35


import os, sys, json
import logging
#print(logging.__path__)
from smlp_py.smlp_utils import str_to_bool
    
# LOGGER sources:
# https://docs.python.org/3/howto/logging.html
# https://stackoverflow.com/questions/29087297/is-there-a-way-to-change-the-filemode-for-a-logger-object-that-is-not-configured/29087645#29087645
# create python logger object with desired configuration ; define the default configuration logger_params_dict
class SmlpLogger:
    def __init__(self):
        self._DEF_LOGGER_LEVEL = 'warning'
        self._DEF_LOGGER_FMODE = 'w'
        self._DEF_LOGGER_TIME = 'true'

        self.logger_params_dict = {
            'log_level': {'abbr':'log_level', 'default':'info', 'type':str,
                'help':'The logger level or severity of the events they are used to track. The standard levels ' + 
                    'are (in increasing order of severity): notset, debug, info, warning, error, critical; ' +
                    'only events of this level and above will be tracked [default {}]'.format(self._DEF_LOGGER_LEVEL)}, 
            'log_mode': {'abbr':'log_mode', 'default':'w', 'type':str,
                'help':'The logger filemode for logging into log file [default {}]'.format(self._DEF_LOGGER_FMODE)},
            'log_time': {'abbr':'log_time', 'default':self._DEF_LOGGER_TIME, 'type':str_to_bool,
                'help':'Should time stamp be logged along with every message issued by logger ' +
                    '[default {}]'.format(self._DEF_LOGGER_TIME)}}

    # create python logger object with desired configuration 
    def create_logger(self, logger_name, log_file, log_level, log_mode, log_time):
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
        if logger_name == 'smlp_logger':
            if log_time:
                formatter = logging.Formatter('\n%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            else:
                formatter = logging.Formatter('\n%(name)s - %(levelname)s - %(message)s')
        elif logger_name == 'smlp_tracer':
            if log_time:
                formatter = logging.Formatter('%(asctime)s - %(message)s')
            else:
                formatter = logging.Formatter('%(message)s')
        else:
            raise Exception('Unexpected logger ' + str(logger_name))
        #formatter = logging.Formatter('[%(asctime)s] %(levelname)8s --- %(message)s ' +
        #                             '(%(filename)s:%(lineno)s)',datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(ch)
        logger.addHandler(fh)

        return logger

# profiling SMLP run, the steps taken by the algorithm and solver runtimes
class SmlpTracer:
    def __init__(self):
        self._query_tracer = None
        self._trace_runtime = None
        self._trace_precision = None
        self._trace_anonymize = None
        
        # default values for controlling trace logs 
        self._DEF_TRACE_RUNTIME = 0 # whether / how many decinal points of runtimes to include in trace
        self._DEF_TRACE_PRECISION = 0 # whether / how many decinal points of fractions approx to include in trace
        self._DEF_TRACE_ANONYMIZE = False
        self.trace_params_dict = {
            'trace_runtime': {'abbr':'trace_runtime', 'default':str(self._DEF_TRACE_RUNTIME), 'type':int,
                'help':'Should trace include solver runtimes and what precision to use in terms of number ' +
                        'of decimal points after 0; the option value 0 means to not include the runtimes ' +
                        'in the trace [default {}]'.format(str(self._DEF_TRACE_RUNTIME))}, 
            'trace_precision':{'abbr':'trace_prec', 'default':self._DEF_TRACE_PRECISION, 'type':int,
                'help':'Decimals after 0 to use when rounding fractions; option value 0 means to use fractions ' +
                    '(implying no rounding) [default: {}]'.format(str(self._DEF_TRACE_PRECISION))},
            'trace_anonymize':{'abbr':'trace_anonym', 'default':self._DEF_TRACE_ANONYMIZE, 'type':str_to_bool,
                'help':'Should anonymized names of system inputs, knobs and outputs be uses in trace log file?' +
                    '[default: {}]'.format(str(self._DEF_TRACE_ANONYMIZE))}
        }

