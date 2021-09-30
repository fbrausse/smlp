#
# This file is part of smlprover.
#
# Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
# See the LICENSE file for terms of distribution.

import os, datetime, sys, json
from fractions import Fraction

from pandas import DataFrame

from sklearn.preprocessing import MinMaxScaler

import numpy as np

class DataFileInstance:
    # data_file_prefix is the name of the data file including the full path to
    #                  the file but excluding the .csv suffix
    def __init__(self, data_file_prefix):
        self._dir, self._pre = os.path.split(data_file_prefix)

    @property
    def data_fname(self):
        return os.path.join(self._dir, self._pre + ".csv")

    @property
    def _out_prefix(self):
        return self._pre + "_12"

    @property
    def model_file(self):
        return os.path.join(self._dir, "model_complete_" + self._out_prefix + ".h5")

    @property
    def model_checkpoint_pattern(self):
        return os.path.join(self._dir, "model_checkpoint_" + self._out_prefix + ".h5")

    @property
    def model_config_file(self):
        return os.path.join(self._dir, "model_config_" + self._out_prefix + ".json")

    @property
    def data_bounds_file(self):
        return os.path.join(self._dir, "data_bounds_" + self._out_prefix + ".json")

    @property
    def model_gen_file(self):
        return os.path.join(self._dir, "model_gen_" + self._out_prefix + ".json")


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
	#resp_ind = df.columns.get_loc(resp_names)
	#print('resp_ind', resp_ind)
	return df.drop([n for n in df if n not in input_names], axis=1), df[resp_names]
#	feat_df, resp_df = df.drop(resp_names, axis=1), df[resp_names]
#	#print(feat_df)
#	#print(resp_df)
#	return feat_df, resp_df


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
	                         if s['type'] in ('categorical', 'knob')],
	                        bnds)
	so = scaler_from_bounds([s for s in spec
	                         if s['type'] == 'response'
	                         and s['label'] in gen['response']],
	                        bnds)
	return si, so

def obj_range(gen, bnds):
	r = gen['response']
	if len(r) == 1 and gen['objective'] == r[0]:
		so = scaler_from_bounds([{'label': r[0]}], bnds)
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
