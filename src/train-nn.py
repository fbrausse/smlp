#!/usr/bin/env python3
#
# This file is part of smlprover.
#
# Copyright 2019 Konstantin Korovin
# Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
# See the LICENSE file for terms of distribution.

# coding: utf-8

import os, sys, argparse

from smlp.train import (nn_main, DEF_SPLIT_TEST, DEF_OPTIMIZER, DEF_EPOCHS,
                        HID_ACTIVATION, OUT_ACTIVATION, DEF_BATCH_SIZE,
                        DEF_LOSS, DEF_METRICS, DEF_MONITOR, DEF_SCALER)

from pandas import read_csv, DataFrame

from common import *

import json

def parse_args(argv):
	p = argparse.ArgumentParser(prog=argv[0])
	p.add_argument('data', metavar='DATA', type=str,
	               help='Path excluding the .csv suffix to input data ' +
	                    'file containing labels')
	p.add_argument('-a', '--nn_activation', default=HID_ACTIVATION, metavar='ACT',
	               help='activation for NN [default: ' + HID_ACTIVATION + ']')
	p.add_argument('-b', '--nn_batch_size', type=int, metavar='BATCH',
	               default=DEF_BATCH_SIZE,
	               help='batch_size for NN' +
	                    ' [default: ' + str(DEF_BATCH_SIZE) + ']')
	p.add_argument('-B', '--bounds', type=str, default=None,
	               help='Path to pre-computed bounds.csv')
	p.add_argument('-c', '--chkpt', type=str, default=None, nargs='?', const=(),
	               help='save model checkpoints after each epoch; ' +
	                    'optionally use CHKPT as path, can contain named ' +
	                    'formatting options "{ID:FMT}" where ID is one of: ' +
	                    "'epoch', 'acc', 'loss', 'val_loss'; if these are " +
	                    "missing only the best model will be saved" +
	                    ' [default: no, otherwise if CHKPT is missing: ' +
	                    'model_checkpoint_DATA.h5]')
	p.add_argument('-e', '--nn_epochs', type=int, default=DEF_EPOCHS,
	               metavar='EPOCHS',
	               help='epochs for NN [default: ' + str(DEF_EPOCHS) + ']')
	p.add_argument('-f', '--filter', type=float, default=0.0,
	               help='filter data set to rows satisfying RESPONSE >= quantile(FILTER) '+
	                    '[default: no]')
	p.add_argument('-l', '--nn_layers', default='1', metavar='LAYERS',
	               help='specify number and sizes of the hidden layers of the '+
	                    'NN as non-empty colon-separated list of positive '+
	                    'fractions in the number of input features in, e.g. '+
	                    '"1:0.5:0.25" means 3 layers: first of input size, '+
	                    'second of half input size, third of quarter input size; '+
	                    '[default: 1 (one hidden layer of size exactly '+
	                    '#input-features)]')
	p.add_argument('-o', '--nn_optimizer', default=DEF_OPTIMIZER, metavar='OPT',
	               help='optimizer for NN [default: ' + DEF_OPTIMIZER + ']')
	p.add_argument('-O', '--objective', type=str, default=None,
	               help='Objective function in terms of labelled outputs '+
	                    '[default: RESPONSE if it is a single variable]')
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
	args = p.parse_args(argv[1:])
	if args.objective is None and len(args.response.split(',')) == 1:
		args.objective = args.response
	return args


def main(argv):
	args = parse_args(argv)

	# data related args
	inst = DataFileInstance(args.data)

	chkpt = args.chkpt
	if chkpt == ():
	   chkpt = inst.model_checkpoint_pattern

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

	nn_main(inst, resp_names, input_names, args.split_test, chkpt,
	        args.nn_layers, #args.preprocess,
	        args.seed, args.filter, args.objective, bnds,
	        args.nn_epochs, args.nn_batch_size, args.nn_optimizer,
	        args.nn_activation)


if __name__ == "__main__":
	main(sys.argv)
