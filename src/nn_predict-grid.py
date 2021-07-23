#!/usr/bin/env python3
#
# This file is part of smlprover.
#
# Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
# See the LICENSE file for terms of distribution.

import datetime, argparse, csv, itertools, json, sys

from tensorflow.keras.models import load_model
from tensorflow import keras
import tensorflow as tf

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from common import *

# yields a sequence of pairs (i,r) where i is an index into spec and r is a
# finite sequence of values considered to be "safe" for that dimension
def safe_ranges(spec, bnds):
	for i,s in enumerate(spec):
		if s['type'] != 'knob':
			continue
		if 'safe' in s:
			yield i, s['safe']
		else:
			assert s['range'] == 'int'
			b = bnds[s['label']]
			yield i, range(round(b['min']), round(b['max'])+1)

def safe_grid(spec, bnds, log = lambda *args: None):
	ranges = dict(safe_ranges(spec, bnds))
	import functools
	log('generating grid with %d entries' %
	    functools.reduce(lambda a, b: a*b, map(len, ranges.values()), 1))
	return pd.DataFrame(itertools.product(*ranges.values()),
	                    columns=[spec[i]['label'] for i in ranges])

	#data = tf.data.Dataset.from_generator(
	#                        lambda: map(lambda r: ((r,),), itertools.product(*ranges.values())),
	#                        (tf.int32,),
	#                        (tf.TensorShape([None,6]),))
	#print(data)
	#print(list(data.take(3).as_numpy_iterator()))
	#print(list(map(lambda r: r[0].shape, data.take(3).as_numpy_iterator())))


def nn_predict_grid(spec, bnds, gen, model, values, obj_threshold = None,
                    log = lambda *args: None):
	safe = values

	si, so = io_scalers(spec, gen, bnds)
	assert gen['pp']['features'] == 'min-max'
	safe[gen['response']] = timed(lambda:
		pd.DataFrame(so.inverse_transform(model.predict(
			si.transform(safe), batch_size=1 << 16, workers=0))),
		'prediction', log=log)

	assert 'objective' in gen
	if 'objective' not in gen and len(gen['response']) == 1:
		gen['objective'] = gen['response'][0]

	resps = list(gen['response'])
	if len(gen['response']) != 1 or gen['objective'] != gen['response'][0]:
		safe[gen['objective']] = safe.eval(gen['objective'])
		resps.append(gen['objective'])

	if obj_threshold is not None:
		q = '(%s) >= %s' % (gen['objective'], obj_threshold)
		n = len(safe)
		safe = safe.query(q)
		log('query:', q, '->', len(safe), '/', n, 'grid points left')

	return get_response_features(safe, get_input_names(spec), resps)


def obj_scaler(gen, resp_bnds, log = lambda *args: None):
	rng = obj_range(gen, resp_bnds)
	log('obj range:', rng)
	return MinMax(*rng)


def parse_args(argv):
	p = argparse.ArgumentParser(prog=argv[0])
	p.add_argument('model', metavar='MODEL',
	               help='Path to serialized NN model')
	p.add_argument('-s', '--spec', required=True,
	               help='Path to .spec file')
	p.add_argument('-b', '--bounds', required=True,
	               help='Path to data_bounds file')
	p.add_argument('-B', '--response_bounds',
	               help='Path to response bounds CSV')
	p.add_argument('-g', '--gen', required=True,
	               help='Path to model_gen file')
	p.add_argument('-o', '--output', default=sys.stdout,
	               help='Path to output.csv')
	p.add_argument('-p', '--predicted',
	               help='Path to store the predictions to')
	p.add_argument('-t', '--threshold',
	               help='Threshold on objective to restrict outputs by')
	p.add_argument('-v', '--values',
	               help='Optional path to file with points to predict ' +
	                    '(generate grid otherwise)')
	return p.parse_args(argv[1:])

def main(argv):
	args = parse_args(argv)

	with open(args.spec, 'r') as f:
		spec = json.load(f)
	with open(args.bounds, 'r') as f:
		bnds = json.load(f)
	with open(args.gen, 'r') as f:
		gen = json.load(f)

	log = lambda *args: print(*args, file=sys.stderr)

	sc = obj_scaler(gen,
	                bnds if args.response_bounds is None
	                else pd.read_csv(args.response_bounds, index_col=0),
	                log=log)
	abs_t = sc.denorm(float(args.threshold)) if args.threshold is not None else None

	safe, pred = nn_predict_grid(spec, bnds, gen, load_model(args.model),
	                             (timed(lambda: safe_grid(spec, bnds, log=log),
	                                    'generating grid', log=log)
	                              if args.values is None
	                              else pd.read_csv(args.values)[[
	                                  s['label'] for s in spec
	                                  if s['type'] in ('knob','categorical')
	                              ]]),
	                             abs_t, log=log)

	timed(lambda: safe.to_csv(args.output, index=False),
	      'saving domain to CSV', log=log)
	if args.predicted is not None:
		timed(lambda: pred.to_csv(args.predicted, index=False),
		      'saving predictions to CSV', log=log)

if __name__ == "__main__":
	main(sys.argv)
