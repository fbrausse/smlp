#import os # TODO !!! this is for chkpt file only, temporary
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.metrics import mean_squared_error, r2_score

from pandas import read_csv, DataFrame

from logs_common import *
from math import ceil
import json
from smlp.smlp_plot import plot  #, evaluate_model, evaluate_prediction, distplot_dataframe, 
#from smlp.train_common import report_prediction_results

# defaults
#DEF_SPLIT_TEST = 0.2
DEF_OPTIMIZER  = 'adam'  # options: 'rmsprop', 'adam', 'sgd', 'adagrad', 'nadam'
DEF_EPOCHS     = 2000
HID_ACTIVATION = 'relu'
OUT_ACTIVATION = 'linear'
DEF_BATCH_SIZE = 200
DEF_LOSS       = 'mse'
DEF_METRICS    = ['mse']    # ['mae'], ['mae','accuracy']
DEF_MONITOR    = 'loss'     # monitor for keras.callbacks
                            # options: 'loss', 'val_loss', 'accuracy', 'mae'
#DEF_SCALER     = 'min-max'  # options: 'min-max', 'max-abs'
DEF_LAYERS_SPEC = '2,1'

if list(map(int, tf.version.VERSION.split('.'))) < [1]:
    assert False
elif list(map(int, tf.version.VERSION.split('.'))) < [2]:
    HIST_MSE = 'mean_squared_error'
    HIST_VAL_MSE = 'val_mean_squared_error'
elif list(map(int, tf.version.VERSION.split('.'))) < [3]:
    HIST_MSE = 'mse'
    HIST_VAL_MSE = 'val_mse'

MODEL_CHECKPOINT = 'model_checkpoint'

# TODO !!!: should chkpt default be defined in this module (and not in inst)?
# define default values and description of hyperparameters used for Keras NN training.
# Used also to generate the correponding parameter description in argparse.
def get_keras_hparam_deafult_dict():
    nn_hparam_deafult_dict = {
        'nn_layers': {'abbr':'layers', 'default': DEF_LAYERS_SPEC, 'type':str,
            'help':'specify number and sizes of the hidden layers of the '+
                    'NN as non-empty comma-separated list of positive '+
                    'fractions in the number of input features in, e.g. '+
                    'second of half input size, third of quarter input size; '+
                    '[default: {}]'.format(DEF_LAYERS_SPEC)}, 
        'nn_epochs': {'abbr':'epochs', 'default': DEF_EPOCHS, 'type':int,
            'help':'epochs for NN [default: {}]'.format(DEF_EPOCHS)}, 
        'nn_batch_size': {'abbr':'batch', 'default': DEF_BATCH_SIZE, 'type':int,
            'help':'batch_size for NN [default: not exposed]'.format(DEF_BATCH_SIZE)}, 
        'nn_optimizer': {'abbr':'optimizer', 'default': DEF_OPTIMIZER, 'type':str,
            'help':'optimizer for NN [default: {}]'.format(DEF_OPTIMIZER)},
        'nn_chkpt': {'abbr':'chkpt', 'default': MODEL_CHECKPOINT, 'type':str,
            'help':'save model checkpoints after each epoch; ' +
                'optionally use CHKPT as path, can contain named ' +
                'formatting options "{ID:FMT}" where ID is one of: ' +
                "'epoch', 'acc', 'loss', 'val_loss'; if these are " +
                "missing only the best model will be saved" +
                ' [default: no, otherwise if CHKPT is missing: ' +
                'model_checkpoint_DATA.h5]'}, 
        'nn_hid_activation': {'abbr':'hid_activation', 'default': HID_ACTIVATION, 'type':str,
            'help':'hidden layer activation for NN [default: {}]'.format(HID_ACTIVATION)}, 
        'nn_out_activation': {'abbr':'out_activation', 'default': OUT_ACTIVATION, 'type':str,
            'help':'output layer activation for NN [default: {}]'.format(OUT_ACTIVATION)},
        'nn_activation': {'abbr':'activation', 'default': HID_ACTIVATION, 'type':str,
            'help':'activation for NN [default: {}]'.format(HID_ACTIVATION)}}
    return nn_hparam_deafult_dict


# local names for model are 'dt', 'rf', ..., while global names are 'dt_sklearn'
# 'rf_sklearn', to distinguish dt, rf, ... implementation in different packages
def algo_name_local2global(algo):
    return algo+'_keras'

KERAS_MODELS = ['nn']    
SMLP_KERAS_MODELS = [algo_name_local2global(m) for m in KERAS_MODELS]

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

    #print("nn_init_model:model") ; print(model)
    return model


# the following function was implemented by adapting the example code below, from:
# https://medium.com/analytics-vidhya/keras-model-sequential-api-vs-functional-api-fc1439a6fb10
# Since we required to support various layer topologies, we needed to work with globals()
# to define input/internal and output layer names and the correponding program variables.
# This is another related link: https://machinelearningmastery.com/keras-functional-api-deep-learning/
'''
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras import layers

input1 = Input(shape=(10,2))
lay1 = layers.Dense(4, input_shape=(10,2))(input1)
lay2 = layers.Dense(4)(lay1)
out1 = layers.Dense(1)(lay2)
out2 = layers.Dense(1)(lay2)
func_model = Model(inputs=input1, outputs=[out1, out2])
func_model.summary()
'''
def nn_init_model_functional(input_dim, optimizer, activation, layers_spec, resp_names):
    layers_spec_list = [float(x) for x in layers_spec.split(',')]
    print('layers_spec_list', layers_spec_list)
    # layer 0 will be input layer, the rest internal layers. Outputs are defined separately below
    nn_layer_names = ['nn_layer_' + str(i) for i in range(len(layers_spec_list))]
    nn_layer_vars = [] # filled in as the input and internal layers are created
    nn_output_vars = [] # filled in as the outputs are created, each one separately as different layers  
    for i in range(len(layers_spec_list)):
        fraction =  layers_spec_list[i]
        print('fraction', fraction)
        assert fraction > 0
        # TODO !!!!!!! check why one would specify input layer fraction not equal to 1?
        #if i == 0:
        #    assert fraction == 1
        n = ceil(input_dim * fraction)
        print("dense layer of size", n)
        #print('assigning variable', nn_layer_names[i])
        if i == 0:
            globals()[nn_layer_names[i]] = keras.layers.Input(shape=(input_dim,))
        else:        
            globals()[nn_layer_names[i]] = keras.layers.Dense(n, activation=activation, input_dim=None)(globals()[nn_layer_names[i-1]])
            #print('var i-1', nn_layer_vars[i-1], 'gl name i-1', globals()[nn_layer_names[i-1]])
            assert (globals()[nn_layer_names[i-1]]) is (nn_layer_vars[i-1])
        nn_layer_vars.append(globals()[nn_layer_names[i]])
    #print('nn_layer_vars', nn_layer_vars)
    #print('nn_layer_names[-1]', nn_layer_names[-1])
    losses = {}
    for resp in resp_names:
        globals()[f"{resp}"] = keras.layers.Dense(1, name=resp, activation=OUT_ACTIVATION)(globals()[nn_layer_names[-1]])
        losses[resp] = DEF_LOSS
        nn_output_vars.append(globals()[f"{resp}"])
    #print('nn_output_vars', nn_output_vars)    
    model = keras.models.Model(nn_layer_vars[0], nn_output_vars)
    #model.compile(optimizer=optimizer, loss=DEF_LOSS, metrics=DEF_METRICS)
    model.compile(optimizer=optimizer, loss=losses, metrics=DEF_METRICS)    
    #print("nn_init_model:model"); print(model)
    return model

def nn_train(model, epochs, batch_size, model_checkpoint_path,
             X_train, X_test, y_train, y_test, weights):

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

    # build sequential model if and only if sample weights are defferent from the default (same weight for each sample)
    # Without sample weights, sequential model (build with sequential API) is expected to be better than functional model
    SEQUENTIAL_MODEL = weights == 0
    
    # for functional API, compute sample weights to give preference to samples with high values in the outputs
    if SEQUENTIAL_MODEL:
        history = model.fit(X_train, y_train,
                            epochs=epochs,
                            validation_data=(X_test, y_test),
                            #steps_per_epoch=10,
                            callbacks=[c for c in (checkpointer,earlyStopping,rlrop)
                                       if c is not None],
                            batch_size=batch_size)
    else:
        sw_dict={} # sample weights, defined per output
        for outp in y_train.columns.tolist(): 
            print('y_train', y_train[outp])
            sw = y_train[outp].values
            #print('sw', sw)
            sw = np.power(sw, abs(weights))
            if weights < 0:
                sw = 1 - sw
            assert any(sw <= 1) ; assert any(sw >= 0) ; print('sw', sw[0: min(5, len(sw))]);
            sw_dict[outp] = sw
        #sample_weight = np.ones(shape=(len(y_train),))
        #sample_weight[y_train >= 0.8] = 2.0
        history = model.fit(X_train, y_train,
                            epochs=epochs,
                            validation_data=(X_test, y_test),
                            #steps_per_epoch=10,
                            sample_weight=sw_dict,
                            callbacks=[c for c in (checkpointer,earlyStopping,rlrop)
                                       if c is not None],
                            batch_size=batch_size)

    return history


def report_training_regression(history, epochs, interactive, out_prefix=None):
    #print('history.history', history.history)
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
    #plt.ylim(0, 2000)
    plot('train-reg', interactive, out_prefix)


# this function extracts individual parameter values from hyperparameter values
# dictionary hparam_dict passed to function keras_main. if a parameter value in
# hparam_dict is None, we interpret this as instruction to replace None with the
# default value defined in this module. This function should not be applied to
# parameters for which None is an acceptable value (e.g. parameter seed might be 
# left as None if we do not want to pass a seed to training).
def get_parm_val(hparam_dict, param):
    if not param in hparam_dict:
        raise Exception('Parameter ' + str(param) + ' is missing in hparam_dict')
    return hparam_dict[param] #if not hparam_dict[param] is None else default

# runs Keras NN algorithm, outputs lots of stats, saves model to disk
# epochs and batch_size are arguments of NN algorithm from keras library
def keras_main(inst, input_names, resp_names : list, algo,
        X_train, X_test, y_train, y_test, hparam_dict, interactive_plots, 
        seed, weights, save_model, data=None):
    # TODO !!!: Can inst.model_checkpoint_pattern be defined locally instead of as part of inst? 
    chkpt_pattern = get_parm_val(hparam_dict, 'nn_chkpt') #inst.model_checkpoint_pattern)
    layers_spec = get_parm_val(hparam_dict, 'nn_layers') #DEF_LAYERS_SPEC
    epochs = get_parm_val(hparam_dict, 'nn_epochs') #DEF_EPOCHS)
    batch_size = get_parm_val(hparam_dict, 'nn_batch_size') #DEF_BATCH_SIZE
    optimizer = get_parm_val(hparam_dict, 'nn_optimizer') #DEF_OPTIMIZER
    activation = get_parm_val(hparam_dict, 'nn_hid_activation') #HID_ACTIVATION
    
    print('START  nn_main')
    print('formating check')
    print('layers_spec =', layers_spec, '; seed =', seed, '; weights =', weights)
    print('epochs =', epochs, '; batch_size =', batch_size, '; optimizer =', optimizer)
    print('chkpt_pattern =', chkpt_pattern, '; activation =', activation)
    
    # TODO !!!! check for None in smlp_train was: if chkpt == () and args.model == 'NN'
    #if chkpt_pattern is None:
    #    chkpt_pattern = inst.model_checkpoint_pattern
    
    # set the seed for reproducibility
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)
    
    # TODO !!!: should we drop 'split-test': split_test from hyperparam_persist ; if we want to record 
    # split_test we can add it to a new (and common for all ML models) data_persist dictionary?
    hyperparam_persist = {
        #'response': resp_names,
        #'objective': objective,
        #'obj-bounds': (lambda s: {
        #    'min': s.data_min_[0],
        #    'max': s.data_max_[0],
        #})(MinMaxScaler().fit(DataFrame(data.eval(objective)))) if objective is not None else None,
        'train': {
            'epochs': epochs,
            'batch-size': batch_size,
            'optimizer': optimizer,
            'activation': activation,
            #'split-test': split_test,
            'seed': seed,
        },
    }
    # TODO: need a different filename for persistance of hyperparameters
    with open(inst.model_gen_file + '_hyperparameters', 'w') as f:
        json.dump(hyperparam_persist, f)
    
    '''
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
    '''
    SEQUENTIAL_MODEL = weights == 0
    input_dim = X_train.shape[1] #num_columns
    if SEQUENTIAL_MODEL:
        model = nn_init_model(input_dim, optimizer, activation, layers_spec, len(resp_names))
    else:
        model = nn_init_model_functional(input_dim, optimizer, activation, layers_spec, resp_names)
    #TODO !!! -- maybe chkpt_pattern_filename should be generated in some other place, though it seems to be unique for KERAS NN only?
    ### chkpt_pattern_filename = os.path.join(inst._dir, "model_checkpoint_" + inst._out_prefix + ".h5")
    #chkpt_pattern_filename = model_checkpoint_pattern(inst)
    history = nn_train(model, epochs, batch_size, chkpt_pattern,
                       X_train, X_test, y_train, y_test, weights)
    model.save(inst.model_file)
    with open(inst.model_config_file, "w") as json_file:
        json_file.write(model.to_json())

    # plot how training iterations improve error/model precision
    report_training_regression(history, epochs, interactive_plots, inst._filename_prefix)   #out_prefix=os.path.join(inst._dir, inst._out_prefix)

    #  print("evaluate")
    #  score = model.evaluate(x_test, y_test, batch_size=200)
    #  print(score)
    
    return model
