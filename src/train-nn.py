#!/usr/bin/env python3
#
# This file is part of smlprover.
#
# Copyright 2019 Konstantin Korovin
# Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
# See the LICENSE file for terms of distribution.

# coding: utf-8

import os, sys, argparse
print("OK")

import numpy as np
from pandas import read_csv, DataFrame
print("OK2")

import tensorflow as tf
from tensorflow import keras
print("OK3")

import matplotlib.pyplot as plt
from matplotlib import cm
print("OK4")

import seaborn as sns
print("OK5")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.metrics import mean_squared_error, r2_score
print("OK6")

from common import *
from math import ceil
import json

# defaults
DEF_SPLIT_TEST = 0.2
DEF_OPTIMIZER  = 'adam'  # options: 'rmsprop', 'adam', 'sgd', 'adagrad', 'nadam'
DEF_EPOCHS     = 2000
HID_ACTIVATION = 'relu'
OUT_ACTIVATION = 'linear'
DEF_BATCH_SIZE = 200
DEF_LOSS       = 'mse'
DEF_METRICS    = ['mse']    # ['mae'], ['mae','accuracy']
DEF_MONITOR    = 'loss'     # monitor for keras.callbacks
							# options: 'loss', 'val_loss', 'accuracy', 'mae'
DEF_SCALER     = 'min-max'  # options: 'min-max', 'max-abs'

if list(map(int, tf.version.VERSION.split('.'))) < [1]:
	assert False
elif list(map(int, tf.version.VERSION.split('.'))) < [2]:
	HIST_MSE = 'mean_squared_error'
	HIST_VAL_MSE = 'val_mean_squared_error'
elif list(map(int, tf.version.VERSION.split('.'))) < [3]:
	HIST_MSE = 'mse'
	HIST_VAL_MSE = 'val_mse'

if [0,9] <= list(map(int, sns.__version__.split('.'))) < [0,10]:
	def distplot_dataframe(inst, y, resp_names):
		plt.figure('Distribution of response features')
		cc = plt.cm.get_cmap('hsv')
		sns.distplot(y, hist=True, kde=False, bins=ceil(max(10, 50/len(resp_names))),
		             color=[cc(c/len(resp_names)) for c,n in enumerate(resp_names)])
		plt.legend(resp_names, loc='upper right')
		plot('resp-distr', os.path.join(inst._dir, inst._out_prefix), block=False)
elif [0,10] <= list(map(int, sns.__version__.split('.'))) < [0,12]:
	def distplot_dataframe(inst, y, resp_names):
		plt.figure('Distribution of response features')
		for c in y:
			sns.distplot(y[c], hist=True, kde=False, bins=ceil(max(10, 50/len(resp_names))))
		plot('resp-distr', os.path.join(inst._dir, inst._out_prefix), block=False)
else:
	print('seaborn version',sns.__version__,'not supported; required: 0.9.x or 0.10.x',
		  file=sys.stderr)
	sys.exit(1)


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


def plot(name, out_prefix=None, **show_kws):
	if out_prefix is not None:
		plt.savefig(out_prefix + '_' + name + '.png')
	plt.show(**show_kws)
	plt.clf()


def plot_data_columns(data):
   # columns = list(data)
	for c in data:
		#print(c,data[c])
		plt.figure() # start a new figure rather than draw on top of each other
		sns.distplot(data[c], hist=True, kde=False, bins=50)
		#print(c)


def nn_init_model(input_dim, optimizer, activation, layers_spec, n_out):
	model = keras.models.Sequential()
	first = True
	for fraction in map(float, layers_spec.split(',')):
		assert fraction > 0
		n = ceil(input_dim * fraction)
		print("dense layer of size", n)
		model.add(keras.layers.Dense(n, activation=activation,
		                             input_dim=input_dim if first else None))
		first = False
	model.add(keras.layers.Dense(n_out, activation=OUT_ACTIVATION))
	model.compile(optimizer=optimizer, loss=DEF_LOSS, metrics=DEF_METRICS)

	print("nn_init_model:model")
	print(model)
	return model


def nn_train(model, epochs, batch_size, model_checkpoint_path,
             X_train, X_test, y_train, y_test):

	checkpointer = None
	if model_checkpoint_path:
		checkpointer = keras.callbacks.ModelCheckpoint(
				filepath=model_checkpoint_path, monitor=DEF_MONITOR,
				verbose=0, save_best_only=True)

	earlyStopping = None
	if False:
		earlyStopping = keras.callbacks.EarlyStopping(
				monitor=DEF_MONITOR, patience=100, min_delta=0,
				verbose=0, mode='auto',
				restore_best_weights=True)

	rlrop = None
	if False:
		rlrop = keras.callbacks.ReduceLROnPlateau(
				monitor=DEF_MONITOR,
				# XXX: no argument 'lr' in docs; there is 'min_lr', however
				lr=0.000001, factor=0.1, patience=100)

	history = model.fit(X_train, y_train,
	                    epochs=epochs,
	                    validation_data=(X_test, y_test),
	#                   steps_per_epoch=10,
	                    callbacks=[c for c in (checkpointer,earlyStopping,rlrop)
	                               if c is not None],
	                    batch_size=batch_size)

	return history

  #  print("predict")
  #  predict=model.predict(x_test, batch_size=200)
  #  print(predict)

  #  print("evaluate")
  #  score = model.evaluate(x_test, y_test, batch_size=200)
  #  print(score)


def plot_true_pred_runtime1(y, y_pred, title, out_prefix=None, log_scale=False):
	print("{1} msqe: {0:.3f}".format(mean_squared_error(y, y_pred), title))
	print("{1} r2_score: {0:.3f}".format(r2_score(y, y_pred), title))

	#print("Train rmse: {0:.3f}".format(rmse(y_pred, y)))
	print()

	if log_scale:
		y = np.log10(y)
		y_pred = np.log10(y_pred)

	l_plot = np.linspace(y.min(), y.max(), 100)
	y_pred = DataFrame(y_pred, columns=y.columns)
	for c in y.columns:
		ax = sns.scatterplot(x=y[c].values, y=y_pred[c].values, marker='+', label=c)

		ax.set_title(title + ' col "%s"' % c)
		ax.set_xlabel('True values')
		ax.set_ylabel('Predicted values')

		plt.gcf().set_size_inches(16, 9)

		plt.plot(l_plot, l_plot, color='r')
		#plt.ylim(0, max(y) * 1.1)
		#plt.xlim(0, max(y) * 1.1)

		plot('eval-' + title + '-col-%s' % c, out_prefix)

def plot_true_pred_runtime0(y, y_pred, title, out_prefix=None, log_scale=False):
    try:
        plot_true_pred_runtime1(y, y_pred, title, out_prefix=out_prefix, log_scale=log_scale)
    except ValueError as e:
        print(e, file=sys.stderr)

def plot_true_pred_runtime(y, y_pred, title, out_prefix=None, log_scale=False,
						   objective=None):
	plot_true_pred_runtime0(y, y_pred, title, out_prefix=out_prefix, log_scale=log_scale)
	if objective is not None:
		plot_true_pred_runtime0(DataFrame(y.eval(objective), columns=[objective]),
		                        DataFrame(DataFrame(y_pred, columns=y.columns).eval(objective), columns=[objective]),
		                        title + '-obj', out_prefix=out_prefix,
		                        log_scale=log_scale)


def evaluate_model(model, X_train, X_test, y_train, y_test, out_prefix=None,
				   log_scale=False, objective=None):
	## Evaluate Train
	plot_true_pred_runtime(y_train, model.predict(X_train), "Train",
	                       out_prefix=out_prefix, log_scale=log_scale,
	                       objective=objective)

	## Evaluate Test
	plot_true_pred_runtime(y_test, model.predict(X_test), "Test",
	                       out_prefix=out_prefix, log_scale=log_scale,
	                       objective=objective)


def report_training_regression(history, epochs, out_prefix=None):
	acc = history.history[HIST_MSE]
	#print(acc)
	val_acc = history.history[HIST_VAL_MSE]

	loss = history.history['loss']
	val_loss = history.history['val_loss']

	epochs_range = range(epochs)

	plt.figure()
	#plt.figure(figsize=(12, 5))
	#plt.subplot(1, 2, 1)
	plt.plot(epochs_range, acc, label='Training mse')
	plt.plot(epochs_range, val_acc, label='Validation mse')
	plt.legend(loc='upper right')
	plt.title('Training and Validation mse')

	ind_5 = len(acc)-int(len(acc)/10)
	acc_5 = acc[-ind_5:]
	plt.ylim(0, max(acc_5))
#   plt.ylim(0, 2000)
	plot('train-reg', out_prefix)


def evaluate_nn(model, history, epochs, X_train, X_test, y_train, y_test,
                out_prefix=None, log_scale=False, objective=None):
	report_training_regression(history, epochs, out_prefix=out_prefix)
	evaluate_model(model, X_train, X_test, y_train, y_test,
	               out_prefix=out_prefix, log_scale=log_scale,
	               objective=objective)


# runs Keras NN algorithm, outputs lots of stats, saves model to disk
# epochs and batch_size are arguments of NN algorithm from keras library
def nn_main(inst, resp_names, input_names, split_test, chkpt_pattern,
            layers_spec, pp, seed, filter, objective, bnds,
            # keras NN arguments:
            epochs, batch_size, optimizer, activation):
	print('START  nn_main')

	print('data_fname', inst.data_fname)
	print('out_prefix', inst._out_prefix)
	print('out_model_file', inst.model_file)
	print('seed', seed)

	if seed is not None:
		tf.random.set_seed(seed)
		np.random.seed(seed)

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

	persist = {
		'response': resp_names,
		'objective': objective,
		'obj-bounds': (lambda s: {
			'min': s.data_min_[0],
			'max': s.data_max_[0],
		})(MinMaxScaler().fit(DataFrame(data.eval(objective)))) if objective is not None else None,
		'train': {
			'epochs': epochs,
			'batch-size': batch_size,
			'optimizer': optimizer,
			'activation': activation,
			'split-test': split_test,
			'seed': seed,
		},
	}

	with open(inst.data_bounds_file, 'w') as f:
		json.dump({
			col: { 'min': mm.data_min_[i], 'max': mm.data_max_[i] }
			for i,col in enumerate(data.columns)
		}, f, indent='\t', cls=np_JSONEncoder)

	if filter > 0:
		if len(resp_names) > 1:
			assert objective is not None
		obj = data.eval(objective)
		data = data[obj >= obj.quantile(filter)]
		persist['filter'] = { 'quantile': filter }
		del obj

	# normalize data
	data = DataFrame(mm.transform(data), columns=data.columns)
	persist['pp'] = {
		'response': 'min-max',
		'features': 'min-max',
	}

	with open(inst.model_gen_file, 'w') as f:
		json.dump(persist, f)

	X, y = get_response_features(data, input_names, resp_names)
	print(X)

	print("y.shape", y.shape)
	#print("y.min:", y.min())
	#print("y.max", y.max())
	print('scaled y', y)
	#plot_data_columns(DataFrame(y, columns=resp_names))

	distplot_dataframe(inst, y, resp_names)

	#y_0_9=np.argwhere(y>0.9) # get indecies
	#print('y[y>0.9]', y[[n>0.9 for n ]])
	#plot_data_columns(y)

	#scaler_std = StandardScaler()
	#X=scaler_std.fit_transform(X)
	#print(X)
	X_train, X_test, y_train, y_test = train_test_split(X, y,
	                                                    test_size=split_test,
	                                                    shuffle=True,
	                                                    random_state=17)
	#scaler = StandardScaler()
	#X_train=scaler.fit_transform(X_train)
	#X_test=scaler.fit(X_test)

	#print(X_test)
	#plot_data_columns(X_test)
	#print(y_test)
	#print(y_train)

	input_dim = X_train.shape[1] #num_columns

	model = nn_init_model(input_dim, optimizer, activation, layers_spec, len(resp_names))

	history = nn_train(model, epochs, batch_size, chkpt_pattern,
	                   X_train, X_test, y_train, y_test)
	model.save(inst.model_file)
	with open(inst.model_config_file, "w") as json_file:
		json_file.write(model.to_json())

	evaluate_nn(model, history, epochs, X_train, X_test, y_train, y_test,
	            out_prefix=os.path.join(inst._dir, inst._out_prefix),
	            log_scale=False, objective=objective)


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

	nn_main(inst, resp_names, input_names, args.split_test, chkpt, args.nn_layers,
	        args.preprocess, args.seed, args.filter, args.objective, bnds,
	        args.nn_epochs, args.nn_batch_size, args.nn_optimizer,
	        args.nn_activation)


if __name__ == "__main__":
	main(sys.argv)
