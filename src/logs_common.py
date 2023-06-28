#
# This file is part of smlprover.
#
# Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
# See the LICENSE file for terms of distribution.

# And move parts not related to inst creation to another module -- say utils_common.py.
# inst actually contains info required for logging, so maybe all feilds and paths/filenames
# definitions need to be moved into a logger module/class (which does not exist now).

import os, datetime, sys, json
from fractions import Fraction

from pandas import DataFrame

from sklearn.preprocessing import MinMaxScaler

import numpy as np

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
        _, new_data_fname = os.path.split(self._new_data)
        if not self._out_dir is None:
            self._filename_prefix = os.path.join(self._out_dir, self._run_prefix + '_' + self._pre)
        else:
            self._filename_prefix = os.path.join(self._dir, self._run_prefix + '_' + self._pre)
        # if new_data is not None, its name is added to self._filename_prefix
        #if not self._new_data is None:
        #    self._filename_prefix = self._filename_prefix + '_' + new_data_fname
        
    # input/training data file name (including the directory path)
    @property
    def data_fname(self):
        return os.path.join(self._dir, self._pre + ".csv")
    
    # new (unseen during training) data file name (including the directory path)
    @property
    def new_data_fname(self):
        return self._new_data + ".csv"
    
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
    

'''  # part of the commented out functions below have been moved to utils_common.py
     # and some others (e.g., get_input_names, get_response_features) into the relevant modules 
NP2PY = {
    np.int64: int,
    np.float64: float,
}

def np2py(o, lenient=False):
    if lenient:
        ty = NP2PY.get(type(o), type(o))
    else:
        ty = NP2PY[type(o)]
    return ty(o)

class np_JSONEncoder(json.JSONEncoder):
    def default(self, o):
        return NP2PY.get(type(o), super().default)(o)


def timed(f, desc=None, log=lambda *args: print(*args, file=sys.stderr)):
    now = datetime.datetime.now()
    r = f()
    t = (datetime.datetime.now() - now).total_seconds()
    if desc is not None:
        log(desc, 'took', t, 'sec')
        return r
    return r, t


def get_input_names(spec):
    return [s['label'] for s in spec if s['type'] != 'response']

# response name of y
def get_response_features(df, input_names, resp_names):
    for resp_name in resp_names:
        assert resp_name in df.columns
    assert all(n in df.columns for n in input_names)
    return df.drop([n for n in df if n not in input_names], axis=1), df[resp_names]


def get_radii(spec, center):
    abs_radii = []
    for s,c in zip(spec, center):
        if s['type'] == 'categorical':
            abs_radii.append(0)
            continue
        if 'rad-rel' in s and c != 0:
            w = s['rad-rel']
            abs_radii.append(Fraction(w) * abs(c))
        else:
            try:
                w = s['rad-abs']
            except KeyError:
                w = s['rad-rel']
            abs_radii.append(w)
    return abs_radii

def scaler_from_bounds(spec, bnds):
    sc = MinMaxScaler()
    mmin = []
    mmax = []
    for s in spec:
        b = bnds[s['label']]
        mmin.append(b['min'])
        mmax.append(b['max'])
    sc.fit((mmin, mmax))
    return sc

def io_scalers(spec, gen, bnds):
    si = scaler_from_bounds([s for s in spec
                             if s['type'] in ('categorical', 'knob', 'input')],
                            bnds)
    so = scaler_from_bounds([s for s in spec
                             if s['type'] == 'response'
                             and s['label'] in gen['response']],
                            bnds)
    return si, so

def obj_range(gen, bnds):
    r = gen['response']
    for resp in r:
        if gen['objective'] == resp:
            so = scaler_from_bounds([{'label': resp}], bnds)
            return so.data_min_[0], so.data_max_[0]
    assert len(r) == 2
    sOu = None
    sOl = None
    for i in range(2):
        if gen['objective'] == ("%s-%s" % (r[i],r[1-i])):
            u = bnds[r[i]]
            l = bnds[r[1-i]]
            sOl = u['min']-l['max']
            sOu = u['max']-l['min']
            break
    assert sOl is not None
    assert sOu is not None
    return sOl, sOu

class MinMax:
    def __init__(self, min, max):
        self.min = min
        self.max = max
    @property
    def range(self):
        return self.max - self.min
    def norm(self, x):
        return (x - self.min) / self.range
    def denorm(self, y):
        return y * self.range + self.min

class Id:
    def __init__(self):
        pass
    def norm(self, x):
        return x
    def denorm(self, y):
        return y

SCALER_TYPE = {
    'min-max': lambda b: MinMax(b['min'], b['max']),
    None     : lambda b: Id(),
}

def input_scaler(gen, b):
    return SCALER_TYPE[gen['pp'].get('features')](b)

def response_scaler(gen, b):
    return SCALER_TYPE[gen['pp'].get('response')](b)

def response_scalers(gen, bnds):
    return [response_scaler(gen, bnds[r]) for r in gen['response']]


class SolverTimeoutError(Exception):
    pass
'''