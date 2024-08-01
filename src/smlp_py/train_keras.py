# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

import os
#os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # edded because of warning: 
os.unsetenv("TF_ENABLE_ONEDNN_OPTS")
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from math import ceil
import json
import numpy as np
import pandas as pd
import random as rn
import io
from contextlib import redirect_stdout

#from tensorflow.keras.initializers import GlorotUniform

from keras_tuner import Hyperband, BayesianOptimization, RandomSearch, Objective

# SMLP
from smlp_py.smlp_logs import *
from smlp_py.smlp_plots import plot
from smlp_py.smlp_utils import str_to_bool, str_to_str_list, str_to_str_list_list, str_to_float_list, str_to_int_list

# Methods for training and predction, results reporting with Tensorflow/KERAS package.
# Currently NN only (with sequential and functional APIs)
# When addig new models self._KERAS_MODELS = ['nn'] needs to be updated
class ModelKeras:
    def __init__(self): 
        #data_logger = logging.getLogger(__name__)
        self._keras_logger = None
        
        self._KERAS_MODELS = ['nn']
        self.SMLP_KERAS_MODELS = [self._algo_name_local2global(m) for m in self._KERAS_MODELS]
        
        # hyper parameter defaults
        self._DEF_LAYERS_SPEC = '2,1'
        self._DEF_EPOCHS     = 200
        self._DEF_BATCH_SIZE = 10
        self._DEF_OPTIMIZER  = 'adam'  # options: 'rmsprop', 'adam', 'sgd', 'adagrad', 'nadam'
        self._DEF_LEARNING_RATE = 0.001
        self._HID_ACTIVATION = 'relu'
        self._OUT_ACTIVATION = 'linear'
        self._SEQUENTIAL_API = True
        self._WEIGHTS_PRECISION = None
        
        # model accuracy parameters
        self._DEF_LOSS       = 'mse'
        self._DEF_METRICS    = ['mse']    # ['mae'], ['mae','accuracy']
        self._DEF_MONITOR    = 'loss'     # monitor for keras.callbacks. options: 'loss', 'val_loss', 'accuracy', 'mae'
        
        # Keras Tuner related params
        self._DEF_TUNER_ALGO = None
        self._DEF_LAYERS_SPEC_GRID = None # [self._DEF_LAYERS_SPEC]
        self._DEF_BATCH_SIZE_GRID = None # [self._DEF_BATCH_SIZE]
        self._DEF_LEARNING_RATES_GRID = None # [self._DEF_LEARNING_RATE]
        self._DEF_LOSS_FUNCTIONS_GRID = None # [self._DEF_LOSS]
        '''
        In Keras, the choice of metrics does not directly affect the training process in terms of how the model 
        weights are updated. Metrics are used for monitoring and evaluating the performance of the model during 
        training and validation, but they do not influence the optimization of the loss function, which is what 
        drives the training. However, metrics are important for the following reasons:
        * Model Evaluation: Metrics give you a way to evaluate the model's performance on the training and 
        validation datasets. They can help you diagnose issues such as overfitting or underfitting and guide 
        decisions about model architecture and hyperparameters.
        * Model Selection: When using techniques like cross-validation or hyperparameter tuning (e.g., with Keras  
        Tuner), metrics are used to compare different models or configurations and select the best one.
        * Early Stopping: If you use an EarlyStopping callback with a specific metric (e.g., val_accuracy orval_loss), 
        the metric's performance on the validation set can determine when to stop training to prevent overfitting.
        * Custom Training Loops: If you implement custom training loops, you can use metrics to monitor additional 
        aspects of model performance that are not captured by the loss function.
        '''
        # Dictionary mapping loss choice to Keras loss functions
        '''
        In Keras, there are several loss functions available for regression models. Some common ones include 
        Mean Squared Error (MSE), Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and 
        Mean Squared Logarithmic Error (MSLE). 
        
        The Huber loss is less sensitive to outliers than MSE because it behaves like MSE for small errors 
        and like MAE for large errors. The delta parameter determines the threshold at which the loss transitions 
        from quadratic to linear. You can adjust this parameter based on the scale of the errors in your data.

        The Log-Cosh loss is another option that is also less sensitive to outliers. It calculates the logarithm 
        of the hyperbolic cosine of the prediction error, which behaves similarly to MSE for small errors and 
        less aggressively for large errors.
        '''
        self._loss_functions = {
            'mse': keras.losses.MeanSquaredError(),
            'mae': keras.losses.MeanAbsoluteError(),
            'mape': keras.losses.MeanAbsolutePercentageError(),
            'msle': keras.losses.MeanSquaredLogarithmicError(),
            'huber': keras.losses.Huber(delta=1.0),  # adjust the delta value as needed
            'logcosh': keras.losses.LogCosh()
        }
        
        self._metrics = {
            'mse': keras.metrics.MeanSquaredError(name='mse'),
            'rmse': keras.metrics.RootMeanSquaredError(name='rmse'),
            'mae': keras.metrics.MeanAbsoluteError(name='mae'),
            'mape': keras.metrics.MeanAbsolutePercentageError(name='mape'),
            'msle': keras.metrics.MeanSquaredLogarithmicError(name='msle'),
            'cosine': keras.metrics.CosineSimilarity(name='cosine'),
            'logcosh': keras.metrics.LogCoshError(name='logcosh')
        } #'r2': keras.metrics.R2Score(name='r2')
        
        # Hard coded parameters, not exposed to user for now
        self._TUNER_MAX_EPOCHS = 100 # the max_epochs paramer for Keras Tuner Hyperband() and the epochs parameter for search()
        self._TUNER_MAX_TRIALS = 30 # the max_trails paramer for Keras Tuner BayesianOptimization() and RandomSearch functions
        self._TUNER_HYPERBAND_FACTOR = 3
        self._TUNER_HYPERBAND_ITERATIONS = 2
        self._TUNER_OVERWRITE = True
        self._TUNER_NUM_INITIAL_POINTS = 2 # Number of randomly chosen hyperparameters to start with, for BayesianOptimization()
        self._TUNER_EXECUTIONS_PER_TRIAL = 1 # Number of models to train for each trial, for RandomSearch()
        self._TUNER_EARLY_STOPPING_PATIENCE = 50
        self._TUNER_MODEL_SIZE_VS_ACCURACY_TRADEOFF = True
        
        if list(map(int, tf.version.VERSION.split('.'))) < [1]:
            assert False
        elif list(map(int, tf.version.VERSION.split('.'))) < [2]:
            self._HIST_MSE = 'mean_squared_error'
            self._HIST_VAL_MSE = 'val_mean_squared_error'
        elif list(map(int, tf.version.VERSION.split('.'))) < [3]:
            self._HIST_MSE = 'mse'
            self._HIST_VAL_MSE = 'val_mse'        
        
        # hyper params dictionary for keras model training
        self._keras_hparam_default_dict = {
            'layers': {'abbr':'layers', 'default': self._DEF_LAYERS_SPEC, 'type':str,
                'help':'specify number and sizes of the hidden layers of the NN as non-empty, comma-separated '+
                        'list of positive fractions in the number of input features in, e.g. "0.5,0.25" '+
                        'specifies the second layer of half input size, third layer of quarter input size '+
                        '(the input layer has one node per input) [default: {}]'.format(self._DEF_LAYERS_SPEC)}, 
            'epochs': {'abbr':'epochs', 'default': self._DEF_EPOCHS, 'type':int,
                'help':'epochs for NN [default: {}]'.format(self._DEF_EPOCHS)}, 
            'batch_size': {'abbr':'batch', 'default': self._DEF_BATCH_SIZE, 'type':int,
                'help':'batch_size for NN [default: not exposed]'.format(self._DEF_BATCH_SIZE)}, 
            'optimizer': {'abbr':'optimizer', 'default': self._DEF_OPTIMIZER, 'type':str,
                'help':'optimizer for NN [default: {}]'.format(self._DEF_OPTIMIZER)},
            'learning_rate': {'abbr':'learning_rate', 'default': self._DEF_LEARNING_RATE, 'type':float,
                'help':'optimizer for NN [default: {}]'.format(self._DEF_LEARNING_RATE)},
            'loss_function': {'abbr':'loss', 'default': self._DEF_LOSS, 'type':str,
                'help':'The loss function for NN training convergence. Possible options are: ' +
                '"mse" (MeanSquaredError), "mae" (MeanAbsoluteError), "mspe" (MeanAbsolutePercentageError) ' +
                '"msle" (MeanSquaredLogarithmicError), "huber" (Huber),  "logcosh" (LogCosh) ' +
                '[default: {}]'.format(self._DEF_LOSS)},
            'metrics': {'abbr':'metrics', 'default': self._DEF_METRICS, 'type':str_to_str_list,
                'help':'The metrics for NN training convergence. Possible options are: "rmse (RootMeanSquaredError), ' +
                '"mse" (MeanSquaredError), "mae" (MeanAbsoluteError), "mspe" (MeanAbsolutePercentageError) ' +
                '"msle" (MeanSquaredLogarithmicError), "logcosh" (LogCoshError), and "cosine" (CosineSimilarity) ' +
                '[default: {}]'.format(self._DEF_METRICS)},
            'hid_activation': {'abbr':'hid_activation', 'default': self._HID_ACTIVATION, 'type':str,
                'help':'hidden layer activation for NN [default: {}]'.format(self._HID_ACTIVATION)}, 
            'out_activation': {'abbr':'out_activation', 'default': self._OUT_ACTIVATION, 'type':str,
                'help':'output layer activation for NN [default: {}]'.format(self._OUT_ACTIVATION)},
            'sequential_api': {'abbr':'seq_api', 'default': self._SEQUENTIAL_API, 'type':str_to_bool,
                'help':'Should sequential api be used building NN layers or should functional ' +\
                        'api be used instead? [default: {}]'.format(str(self._SEQUENTIAL_API))},
            'weights_precision': {'abbr':'weights_precision', 'default': self._WEIGHTS_PRECISION, 'type':int,
                'help':'Decimal precison (theat is, decimal points after the dot) to use for rounding ' +
                        'model weights (after a NN model has been trained). The default value {} ' +
                        'implies that weight will not be rounded [default: {}]'.format(self._OUT_ACTIVATION)},
            'tuner_algo': {'abbr':'tuner', 'default': self._DEF_TUNER_ALGO, 'type':str,
                'help':'NN Keras tuner algorithm to be invoked. Supported options are ' +
                    'hyperband (Hyperband), bayesian (BayesianOptimization) and random (RandomSearch). '
                    'The option value None indicates that keras tuner will not be invoked ' + 
                    '[default: {}]'.format(self._DEF_TUNER_ALGO)},
            'layers_grid': {'abbr':'layers_grid', 'default':self._DEF_LAYERS_SPEC_GRID, 'type':str_to_str_list_list, 
                'help':'Semicolon separated list of NN Keras layers specifications, to be used by Keras tuner. ' +
                    'Each such specification itself is a comma separated list of numbers, see the layers options '
                    'for a detailed description [default: {}]'.format(str(self._DEF_LAYERS_SPEC_GRID))},
            #'epochs_grid': {'abbr':'epochs_grid', 'default':self._DEF_NN_KERAS_EPOCHS, 'type':str_to_int_list, 
            #    'help':'Comma separated list of NN Keras epochs. ' +
            #        '[default: {}]'.format(str(self._DEF_NN_KERAS_EPOCHS))},
            'batches_grid': {'abbr':'batches_grid', 'default':self._DEF_BATCH_SIZE_GRID, 'type':str_to_int_list, 
                'help':'Comma separated list of NN Keras batch sizes, to be used by Keras tuner. ' +
                    '[default: {}]'.format(str(self._DEF_BATCH_SIZE_GRID))},
            'learning_rates_grid': {'abbr':'lrates_grid', 'default':self._DEF_LEARNING_RATES_GRID, 'type':str_to_float_list, 
                'help':'Comma separated list of NN Keras learning rates, to be used by Keras tuner. ' +
                    '[default: {}]'.format(str(self._DEF_LEARNING_RATES_GRID))},
            'loss_functions_grid': {'abbr':'losses_grid', 'default':self._DEF_LOSS_FUNCTIONS_GRID, 'type':str_to_str_list, 
                'help':'Comma separated list of NN Keras loss functions, to be used by Keras tuner. ' +
                    'It can be a subset of loss functions mse, mae, mape, msle, huber, logcosh. ' +
                    '[default: {}]'.format(str(self._DEF_LOSS_FUNCTIONS_GRID))} 
        }

    # set logger from a caller script
    def set_logger(self, logger):
        self._keras_logger = logger 
    
    # set report_file_prefix from a caller script
    def set_report_file_prefix(self, report_file_prefix):
        self.report_file_prefix = report_file_prefix
        
     # set model_file_prefix from a caller script
    def set_model_file_prefix(self, model_file_prefix):
        self.model_file_prefix = model_file_prefix
        
    # local names for model is/are 'nn, ..., while global names are 'nn_ceras',...
    # to distinguish dt, rf, ... implementation in different packages
    def _algo_name_local2global(self, algo):
        return algo+'_keras'
    
    # file to save NN Keras/tensorflow training / error convergence info, known as checkpoints
    @property
    def model_checkpoint_pattern(self):
        assert self.model_file_prefix is not None
        return self.model_file_prefix + '_model_checkpoint.h5'
    
    # TODO !!!: add description
    @property
    def model_gen_file(self):
        assert self.model_file_prefix is not None
        return self.model_file_prefix + '_model_gen.json'
    
    # local name of hyper parameter (as in sklearn package) to global name;
    # the global name is obtained from local name, say 'epochs', by prefixing it
    # with the global name of the algorithm, which results in 'nn_keras_epochs'
    def _hparam_name_local_to_global(self, hparam, algo):
        #print('hparam global name', hparam, algo)
        return self._algo_name_local2global(algo) + '_' + hparam
    
    # given training algo name like dt and the hyper parameter dictionary param_dict  
    # for that algo in the python package used in this class), this function returns  
    # a modified dictionary obtained from param_dictby by adds algo name like nn_keras
    # (where keras is the name of the package used) to the parameter name and its
    # correponding abbriviated name in param_dict.
    def _param_dict_with_algo_name(self, param_dict, algo):
        #print('param_dict', param_dict)
        result_dict = {}
        for k, v in param_dict.items():
            v_updated = v.copy()
            v_updated['abbr'] = self._hparam_name_local_to_global(v['abbr'], algo) # algo + '_' + v['abbr']
            #print('updated abbrv', v_updated['abbr'])
            #print('updated key', self._hparam_name_local_to_global(k, algo))
            result_dict[self._hparam_name_local_to_global(k, algo)] = v_updated #algo + '_' + k
        #raise Exception('tmp')
        return result_dict
    
    # local hyper params dictionary
    def get_keras_hparam_default_dict(self):
        nn_keras_hyperparam_dict = self._param_dict_with_algo_name(self._keras_hparam_default_dict, 'nn')
        return nn_keras_hyperparam_dict
    
    # initialize Keras NN model using sequential API; it does not use sample weights.
    def _nn_init_model_sequential(self, resp_names:list[str], input_dim:int, optimizer:str, hid_activation:str, out_activation:str, 
            layers_spec_list:list[int], loss_function, metrics):
        self._keras_logger.info('building NN model using Keras Sequential API')
        # Initialize the Sequential model
        model = keras.Sequential()
        
        # Create the layers based on the selected topology
        for i, size in enumerate(layers_spec_list):
            if i == 0:
                # The first layer needs to specify the input shape
                self._keras_logger.info('input layer of size ' + str(input_dim))
                self._keras_logger.info('dense layer of size ' + str(size))
                model.add(keras.layers.Dense(units=size, activation=hid_activation, input_shape=(input_dim,)))
            else:
                self._keras_logger.info('dense layer of size ' + str(size))
                model.add(keras.layers.Dense(units=size, activation=hid_activation))

        # in sequential API, there is one "monolithic" output layer, we cannot distinguish
        # individual responses there and set the response names as the output layer names when
        # there are more than one responses (functional API is required for this).
        # We cannot add individual responses in a loop because then responses added later
        # use the responses added earlier in the loop as input layer, which is not intended.
        # Add the output layer(s)
        n_out = len(resp_names)
        self._keras_logger.info('output layer of size ' + str(n_out))
        if n_out == 1:
            # add a single output layer and specify the response name as its name
            model.add(keras.layers.Dense(n_out, activation=out_activation, name=resp_names[0]))
        else:
            # add a single output layer that covers all responses, do not secify its name
            model.add(keras.layers.Dense(n_out, activation=out_activation))
        
        model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
        #print("nn_init_model:model") ; print(model)
        
        return model

        
    # In case of a non-deterministic behaviour, one can try to use output_initializer wih a seed (output_initializer = GlorotUniform(seed=42)
    def _nn_init_model_functional(self, resp_names:list[str], input_dim:int, optimizer:str, hid_activation:str, out_activation:str, 
            layers_spec_list:list[int], loss_function, metrics):
        self._keras_logger.info('building NN model using Keras Functional API')
        self._keras_logger.info('input layer of size ' + str(input_dim))
        inputs = keras.Input(shape=(input_dim,))
        x = inputs

        # Create the layers based on the selected topology
        for size in layers_spec_list:
            self._keras_logger.info('dense layer of size ' + str(size))
            x = keras.layers.Dense(units=size, activation=hid_activation)(x)
        
        outputs = []
        # loss = {} -- required if we wanted to use different loss functions for different responses
        for resp in resp_names:
            self._keras_logger.info('output layer of size ' + str(1))
            # the following line would define a monolothic output layer like this is done for sequential API
            #outputs = keras.layers.Dense(len(resp_names), activation=out_activation)(x)
            output = keras.layers.Dense(1, name=resp, activation=out_activation)(x)
            outputs.append(output)
        
        # Initialize the Functional model
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
        return model
    
    # function for comparing model configurations model.get_config() for sequential vs functional models
    # it does not compare fields with keys having 'layer' as substring, differnces there are expected, but 
    # equivalence of the layers architectrures between func vs seq models can be checked in a different way.
    def compare_model_configs_from_files(self, file1, file2):
        # Load the configurations from the JSON files
        with open(file1, 'r') as f1:
            config1 = json.load(f1)
        with open(file2, 'r') as f2:
            config2 = json.load(f2)

        # Remove keys related to the model's layers to skip layer topology comparison
        keys_to_ignore = [key for key in config1.keys() if 'layers' in key]
        keys_to_ignore += [key for key in config2.keys() if 'layers' in key]  # Include keys from both configs
        for key in keys_to_ignore:
            config1.pop(key, None)
            config2.pop(key, None)

        # Compare the remaining configuration fields
        differences = {}
        for key in set(config1.keys()).union(config2.keys()):  # Union of keys from both configs
            val1 = config1.get(key)
            val2 = config2.get(key)
            if val1 != val2:
                differences[key] = (val1, val2)

        # Print out the differences
        if differences:
            print("Differences found in model configurations:")
            for key, (value1, value2) in differences.items():
                print(f" - {key}: File 1 -> {value1}, File 2 -> {value2}")
        else:
            print("No differences found in model configurations (excluding layer topology).")
    
    # log model details on the layers, loss function, learning rate, optimizer, and model configuration
    def _log_model_summary(self, model, epochs, batch_size, sample_weights, callbacks):
        # Create a StringIO buffer
        buffer = io.StringIO()

        # Redirect the standard output to the buffer
        with redirect_stdout(buffer):
            model.summary()

        # Get the summary string from the buffer
        model_summary = buffer.getvalue()
        
        self._keras_logger.info('model summary: start')
        self._keras_logger.info(model_summary)
        
        '''
        # print weights of layers:
        self._keras_logger.info('Layer weights:')
        for layer in model.layers:
            weights = layer.get_weights()
            self._keras_logger.info(str(weights))
        '''    
        # Print optimizer details, Learning rate, Loss function, metrics, model configuration, sample weights
        self._keras_logger.info("Optimizer: " + str(model.optimizer.get_config()))
        self._keras_logger.info("Learning rate: " + str(model.optimizer.learning_rate.numpy()))
        if isinstance(model.loss, dict): # functiona API, and NN Keras Tuner is not used
            self._keras_logger.info("Loss function: " + str(model.loss))
        else: # sequential API or when NN Keras tuner is used
            for k, v in self._loss_functions.items():
                if str(v) in str(model.loss) or str(k) in str(model.loss):
                    self._keras_logger.info("Loss function: " + str(k))
        if hasattr(model, 'compiled_metrics'):
            compiled_metrics = model.compiled_metrics._metrics  # Access the private _metrics attribute
            self._keras_logger.info("Metrics: " + str([m.name for m in compiled_metrics]))
        else:
            self._keras_logger.info("Metrics: " + str([]))
        #self._keras_logger.info("Metrics: " + str(model.metrics))
        self._keras_logger.info("Model configuration: " + str(model.get_config()))
        self._keras_logger.info("Epochs: " + str(epochs))
        self._keras_logger.info("Batch size: " + str(batch_size))
        self._keras_logger.info("Callbacks: " + str([str(type(clb)) for clb in callbacks]))
        #self._keras_logger.info("Sample weights:\n" + str(sample_weights))
        self._keras_logger.info('model summary: end')
        '''
        with open(self.report_file_prefix + '_model_training_config.json', 'w') as f:
            f.write(json.dumps(model.get_config(),  indent=4, sort_keys=True))
            f.close()
        # Assume 'model1_config.json' and 'model2_config.json' are your JSON files
        self.compare_model_configs_from_files(file1, file2); assert False
        '''
    
    # round weights after model was trained
    def round_model_weights(self, model:keras.Model, num_decimal_places:int):
        for layer in model.layers:
            # Get the current weights of the layer
            weights = layer.get_weights(); #print('current weights\n', weights)

            # Round the weights to the specified number of decimal places
            rounded_weights = [np.round(w, num_decimal_places) for w in weights]

            # Set the rounded weights back to the layer
            layer.set_weights(rounded_weights); #print('rounded weights\n', rounded_weights)
    
    # train keras NN model
    def _nn_train(self, model, epochs, batch_size, weights_precision, model_checkpoint_path,
                 X_train, X_test, y_train, y_test, sample_weights_dict, sequential_api):
        checkpointer = None
        if model_checkpoint_path:
            checkpointer = keras.callbacks.ModelCheckpoint(
                    filepath=model_checkpoint_path, monitor=self._DEF_MONITOR,
                    verbose=0, save_best_only=True)

        earlyStopping = None
        if False:
            earlyStopping = keras.callbacks.EarlyStopping(
                    monitor=self._DEF_MONITOR, patience=100, min_delta=0,
                    verbose=0, mode='auto',
                    restore_best_weights=True)

        rlrop = None
        if False:
            rlrop = keras.callbacks.ReduceLROnPlateau(
                    monitor=self._DEF_MONITOR,
                    # XXX: no argument 'lr' in docs; there is 'min_lr', however
                    lr=0.000001, factor=0.1, patience=100)

        callbacks = [c for c in (checkpointer,earlyStopping,rlrop) if c is not None]
        # log model details
        #self._log_model_summary(model, epochs, batch_size, sample_weights)
        #print('X_train\n', X_train, '\ny_train\n', y_train, '\ntypes', type(X_train), type(y_train))
        #print('X_test\n', X_test, '\ny_test\n', y_test, '\ntypes', type(X_test), type(y_test))
        # train model with sequential or functional API
        if sequential_api: #SEQUENTIAL_MODEL
            if sample_weights_dict is not None:
                sample_weights_df = pd.DataFrame.from_dict(sample_weights_dict)
                sample_weights_vect = np.array(list(sample_weights_df.agg('mean', axis=1)))
            else:
                sample_weights_vect = None
                
            # log model details
            self._log_model_summary(model, epochs, batch_size, sample_weights_vect, callbacks)
            history = model.fit(X_train, y_train,
                                epochs=epochs,
                                validation_data=(X_test, y_test),
                                #steps_per_epoch=10,
                                sample_weight=sample_weights_vect,
                                callbacks=callbacks,
                                batch_size=batch_size)
        else:
            '''
            # this code is for debugging only
            #print('sample_weights_dict', sample_weights_dict)
            sample_weights_df = pd.DataFrame.from_dict(sample_weights_dict); #print('sample_weights_df\n', sample_weights_df)
            sample_weights_vect = None if sample_weights_dict is None else np.array(list(sample_weights_df.agg('mean', axis=1))); #print('sample_weights_vect', sample_weights_vect)
            #for k in sample_weights_dict.keys():
            #    sample_weights_dict[k] = sample_weights_vect
            # log model details
            self._log_model_summary(model, epochs, batch_size, sample_weights_vect, callbacks)
            history = model.fit(X_train, y_train,
                                epochs=epochs,
                                validation_data=(X_test, y_test),
                                #steps_per_epoch=10,
                                sample_weight=sample_weights_vect,
                                callbacks=callbacks, #[c for c in (checkpointer,earlyStopping,rlrop) if c is not None],
                                batch_size=batch_size)
            '''
            # log model details
            self._log_model_summary(model, epochs, batch_size, sample_weights_dict, callbacks)
            history = model.fit(X_train, y_train,
                                epochs=epochs,
                                validation_data=(X_test, y_test),
                                #steps_per_epoch=10,
                                sample_weight=sample_weights_dict,
                                callbacks=callbacks,
                                batch_size=batch_size)
            #'''
        if weights_precision is not None:
            self.round_model_weights(model, int(weights_precision))
        return history

    
    def _report_training_regression(self, history, metrics, epochs, interactive, resp_names, out_prefix=None):
        #print('history.history', history.history)
        epochs_range = range(epochs)
        
        # the metric argument here is string, the name of a Keras metric
        def plot_error_convergence(acc:list[float], val_acc:list[float], resp_name:str, metric:str):
            #plt.figure() -- commented out since otherwise an extra empty plot was displayed
            #plt.figure(figsize=(12, 5))
            #plt.subplot(1, 2, 1)
            #print('acc', acc, '\nval_acc', val_acc)
            plt.plot(epochs_range, acc, label='Training {}'.format(metric))
            plt.plot(epochs_range, val_acc, label='Validation {}'.format(metric))
            #plt.legend(loc='upper right')
            plt.legend(loc='best')
            plt.title('Response {} Training and Validation {}'.format(resp_name, metric))
            plt.xlabel('epochs')
            plt.ylabel(metric.upper()) #'MSE'
            #ind_5 = len(acc)-int(len(acc)/10); #print('ind_5', ind_5)
            #acc_5 = acc[-ind_5:]; #print('acc_5', acc_5)
            #plt.ylim(0, max(acc_5))
            #plt.ylim(0, 2000)
            plot('train-reg_{}_{}'.format(resp_name, metric), interactive, out_prefix)

        # When multiple responses are trained, the format (dictionary keys) of the training history
        # might be different (the there might be keys per respose and their names contain individual
        # response names). probably this depends whether particular responses required more/dedicated
        # training iterations (this is just a guess...). To address this, below we have a case split
        # based on history_for_all_responses which checks whether specific response related keys occur 
        # in training hstory history.history:
        #print('history', history.history.keys())
        for metric in metrics:
            #print(metric in history.history, 'val_'+metric.name in history.history, 'loss' in history.history, 'val_loss' in history.history)
            history_for_all_responses = metric.name in history.history and 'val_'+metric.name in history.history and \
                'loss' in history.history and 'val_loss' in history.history; 
            #print('history_for_all_responses', history_for_all_responses);
            if len(resp_names) == 1 or history_for_all_responses:
                acc = history.history[metric.name]; #print(acc) #self._HIST_MSE
                val_acc = history.history['val_'+metric.name] # self._HIST_VAL_MSE
                loss = history.history['loss']
                val_loss = history.history['val_loss']
                if len(resp_names) == 1:
                    plot_error_convergence(acc, val_acc, resp_names[0], metric.name)
                else:
                    plot_error_convergence(acc, val_acc, 'all_responses', metric.name)
            else:
                for rn in resp_names:
                    acc = history.history[rn + '_' + metric.name]; #print(acc)
                    val_acc = history.history['val_' + rn + '_' + metric.name]
                    loss = history.history[rn + '_' + 'loss']
                    val_loss = history.history['val_' + rn + '_loss']
                    plot_error_convergence(acc, val_acc, rn, metric.name)
            
    # Custom objective function to trade off model accuracy with model size (smaller models are easier for solvers)
    def model_size_accuracy_tradeoff_objective(self, accuracy, model_size, accuracy_tradeoff, size_tradeoff):
        """
        Calculate a custom score that trades off accuracy for model size.
        :param accuracy: The accuracy of the model.
        :param model_size: The size of the model (e.g., number of parameters).
        :param accuracy_tradeoff: How much accuracy we are willing to trade for a smaller model.
        :param size_tradeoff: How much smaller the model should be to trade off some accuracy.
        :return: The custom score, lower is better.
        """
        return accuracy - (accuracy_tradeoff * np.exp(-model_size / size_tradeoff))
    
    def tuner_objective(self, trial):
        # Get the model
        model = trial.hypermodel.build(trial.hyperparameters)

        # Calculate the number of parameters in the model
        model_size = model.count_params()

        # Get the validation accuracy from the executed trial
        val_accuracy = trial.metrics.get_best_value('val_mse')

        # Define tradeoff parameters (you can adjust these)
        accuracy_tradeoff = 0.01  # Example: willing to give up 1% accuracy
        size_tradeoff = 10000     # Example: for a model 10x smaller

        # Calculate the custom score
        score = self.model_size_accuracy_tradeoff_objective(val_accuracy, model_size, accuracy_tradeoff, size_tradeoff)

        return score
    
    # build and compile model using NN Keras functional or sequential API (controlled by the argument "sequential_api").
    # Example: layers_grid = ['2,1', '2,2,2']; these layers specs do not include the input layer -- the latter is added independently
    # from the spec based on the input_dim parameter which defines the number of inputs (by code inputs = keras.Input(shape=(input_dim,))).
    def build_model(self, hp, input_dim:int, resp_names:list[str], sequential_api:bool, hid_activation:str, out_activation:str, 
            metrics, layers_grid:list[str], losses_grid:list, lrates_grid:list):
        # Define the choice of topologies
        topology_choice = hp.Choice('topology', values=layers_grid)

        # Define the layer sizes for each topology
        topology_sizes = dict([(layers_str, [e*input_dim for e in str_to_int_list(layers_str)]) for layers_str in layers_grid])

        # Get the selected topology layer sizes
        selected_topology = topology_sizes[topology_choice]

        # Choose the learning rate
        lr = hp.Choice('learning_rate', values=lrates_grid) if lrates_grid is not None else 0.001
        
        # Choose the loss function
        loss = hp.Choice('loss_function', values=losses_grid) if losses_grid is not None else self._DEF_LOSS
        
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        if sequential_api: #SEQUENTIAL_MODEL
            model = self._nn_init_model_sequential(resp_names, input_dim, optimizer, 
                hid_activation, out_activation, selected_topology, loss, metrics)
        else:
            model = self._nn_init_model_functional(resp_names, input_dim, optimizer, 
                hid_activation, out_activation, selected_topology, loss, metrics)

        return model
        '''
        if self._TUNER_MODEL_SIZE_VS_ACCURACY_TRADEOFF:        
            # Add a custom metric to the model that combines val_loss and model size
            def custom_metric(y_true, y_pred):
                # Here we assume that lower val_loss is better and smaller model size is better
                # You can adjust the tradeoff formula as needed
                val_loss = keras.backend.mean(keras.losses.mean_squared_error(y_true, y_pred))
                # two equivalent methods to comute model size 
                # This method uses the Keras backend function count_params() to return the total number of scalar  
                # values (parameters) in a tensor, which is then summed across all trainable weights of the model. 
                # It gives you the total number of trainable parameters in the model.
                model_size = sum([keras.backend.count_params(p) for p in model.trainable_weights])
                # This method uses the TensorFlow function tf.size() to get the number of elements in a tensor, 
                # which is then converted to a NumPy scalar using .numpy() and summed across all trainable weights 
                # of the model. It also gives you the total number of trainable parameters in the model.
                model_size = sum([tf.size(v).numpy() for v in self.model.trainable_weights])
                tradeoff_score = val_loss + hp.Float('size_tradeoff', 0, 1e-5) * model_size
                return tradeoff_score
            model.add_metric(custom_metric(y_true, y_pred), name='custom_metric', aggregation='mean')
        '''
        
    
    def initialize_tuner(self, input_dim:int, resp_names:list[str], sequential_api:bool, hid_activation:str, out_activation:str, 
            metrics, layers_grid:list, losses_grid:list, lrates_grid:list, tuner_algo:str):
        objective = 'val_loss'
        if tuner_algo == 'hyperband':
            self.tuner = Hyperband(
                lambda hp: self.build_model(hp, input_dim, resp_names, sequential_api, hid_activation, out_activation, metrics, layers_grid, losses_grid, lrates_grid),
                objective=objective, #'val_loss',  # Assuming you want to minimize the validation loss
                max_epochs=self._TUNER_MAX_EPOCHS, 
                factor=self._TUNER_HYPERBAND_FACTOR,
                hyperband_iterations=self._TUNER_HYPERBAND_ITERATIONS,
                directory=self.report_file_prefix + '-keras_tuner_dir',
                project_name='_keras_tuner_hyperband',
                overwrite=self._TUNER_OVERWRITE)
        elif tuner_algo == 'bayesian':
            self.tuner = BayesianOptimization(
                lambda hp: self.build_model(hp, input_dim, resp_names, sequential_api, hid_activation, out_activation, metrics, layers_grid, losses_grid, lrates_grid),
                objective=objective, #'val_loss',  # The objective to minimize
                max_trials=self._TUNER_MAX_TRIALS,  # The maximum number of trials to run
                num_initial_points=self._TUNER_NUM_INITIAL_POINTS,
                directory=self.report_file_prefix + '_keras_tuner_dir',
                project_name='_keras_tuner_bayesian_optimization',
                overwrite=self._TUNER_OVERWRITE)
        elif tuner_algo == 'random':
            self.tuner = RandomSearch(
                lambda hp: self.build_model(hp, input_dim, resp_names, sequential_api, hid_activation, out_activation, metrics, layers_grid, losses_grid, lrates_grid),
                objective=objective, #'val_loss',  # The objective to minimize
                max_trials=self._TUNER_MAX_TRIALS,  # The maximum number of trials to run
                executions_per_trial=self._TUNER_EXECUTIONS_PER_TRIAL,  # Number of models to train for each trial
                directory=self.report_file_prefix + '_keras_tuner_dir',
                project_name='keras_tuner_random_search',
                overwrite=self._TUNER_OVERWRITE)
        else:
            raise Exception('Unexpected NN Keras tuner ' + str(tuner_algo))

    # performing hyperparameter tuning (search)
    def search(self, X_train:pd.DataFrame, y_train:pd.DataFrame, X_val:pd.DataFrame, y_val:pd.DataFrame, input_dim:int, resp_names:list[str], sequential_api:bool,
            hid_activation:str, out_activation:str, metrics, layers_grid:list, losses_grid:list, lrates_grid:list, batches_grid:list, tuner_algo:str):
        self._keras_logger.info('Tuning model hyperparameters using Keras Tuner algorithm ' + str(tuner_algo) + ': start')
        #print('X_train\n', X_train); print('y_train\n', y_train); print('X_val\n', X_val); print('y_val\n', y_val); 
        #print('input_dim =', input_dim); print('resp_names =', resp_names);
        #print('hid_activation =', hid_activation); print('out_activation =', out_activation)
        self.initialize_tuner(input_dim, resp_names, sequential_api, hid_activation, out_activation, metrics, layers_grid, losses_grid, lrates_grid, tuner_algo)
        self.tuner.search(
            x=X_train,
            y=y_train,
            epochs=self._TUNER_MAX_EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=[keras.callbacks.EarlyStopping(patience=self._TUNER_EARLY_STOPPING_PATIENCE)],
            batch_size=self.tuner.oracle.hyperparameters.Choice('batch_size', values=batches_grid) if batches_grid is not None else None        
        )
        best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        # Print the best hyperparameters
        self._keras_logger.info('Best hyperparameters found: start')
        for hyperparam in best_hps.space:
            self._keras_logger.info(f"{hyperparam.name}: {best_hps.get(hyperparam.name)}")
        self._keras_logger.info('Best hyperparameters found: end')
        self._keras_logger.info('Tuning model hyperparameters using Keras Tuner algorithm ' + str(tuner_algo) + ': end')
        
    # Fit / train model with tuned values of hyperparameters (obtained using Keras Tuner search() and strored within self)
    def get_best_model(self, X_train, X_test, y_train, y_test, epochs, weights_coef, batch_size, loss_function_str, learning_rate, sequential_api):
        best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]; #print('best_hps', best_hps)
        best_model = self.tuner.hypermodel.build(best_hps)
        
        callbacks=[] #[keras.callbacks.EarlyStopping(patience=5)]
        self._log_model_summary(best_model, epochs, batch_size, weights_coef, callbacks)
        
        best_batch_size = best_hps.get('batch_size'); #print('best_batch_size', best_batch_size)
        
        '''
        # this code is for debugging
        override_best_params = False
        if override_best_params:
            #print('loss_function_str', loss_function_str, 'batch_size', batch_size)
            new_optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            best_model.compile(optimizer=new_optimizer, loss=loss_function_str, metrics=self._DEF_METRICS)
            history = best_model.fit(
                x=X_train,
                y=y_train,
                epochs=epochs,
                validation_data=(X_test, y_test),
                batch_size=batch_size,
                sample_weight=weights_coef,
                callbacks=callbacks
            )
            return best_model
        '''
        #print('X_train\n', X_train, '\ny_train\n', y_train, '\ntypes', type(X_train), type(y_train))
        #print('X_test\n', X_test, '\ny_test\n', y_test, '\ntypes', type(X_test), type(y_test))
        if sequential_api: #SEQUENTIAL_MODEL
            if weights_coef is not None:
                sample_weights_df = pd.DataFrame.from_dict(weights_coef)
                sample_weights = np.array(list(sample_weights_df.agg('mean', axis=1)))
            else:
                sample_weights = None
        else:
            sample_weights = weights_coef
        #print('sample_weights', sample_weights, type(sample_weights))
        #print('weights_coef', weights_coef)
        history = best_model.fit(
            x=X_train.to_numpy(),
            y=y_train.to_numpy(),
            epochs=epochs,
            validation_data=(X_test.to_numpy(), y_test.to_numpy()),
            batch_size=best_batch_size,
            sample_weight=sample_weights, #weights_coef,
            callbacks=None #[keras.callbacks.EarlyStopping(patience=5)]
        )
        
        return best_model, history

    # Function to build NN Keras model using Keras Tuner for hyperparameter tuning. 
    # NN Keras functional API is used for building the models (even if nn_keras_sequential_api is set to True).
    def train_best_model(self, resp_names:list[str], algo:str, X_train, X_test, y_train, y_test, sequential_api:bool,  
            interactive_plots, seed, weights_coef, model_per_response:bool, hid_activation, out_activation, epochs, weights_precision:int,
            layers_grid, losses_grid, lrates_grid, batches_grid, tuner_algo, batch_size, loss_function_str, metrics, learning_rate):
        # search for best hyperparam values using Keras Tuner
        self.search(X_train, y_train, X_test, y_test, X_train.shape[1], resp_names, sequential_api, 
            hid_activation, out_activation, metrics, layers_grid, losses_grid, lrates_grid, batches_grid, tuner_algo)
        # train the final model using the selected hyper-param values (and by fully executing specified epochs count)
        best_model, history = self.get_best_model(X_train, X_test, y_train, y_test, epochs, weights_coef, batch_size, 
            loss_function_str, learning_rate, sequential_api); 
        # plot how training iterations improve error/model precision
        self._report_training_regression(history, metrics, epochs, interactive_plots, resp_names, self.report_file_prefix)
        if weights_precision is not None:
            assert weights_precision >= 0
            self.round_model_weights(best_model, int(weights_precision))
        return best_model

    # This function extracts individual parameter values from hyperparameter values
    # dictionary hparam_dict passed to function keras_main. if a parameter value in
    # hparam_dict is None, we interpret this as instruction to replace None with the
    # default value defined in this module. This function should not be applied to
    # parameters for which None is an acceptable value (e.g. parameter seed might be 
    # left as None if we do not want to pass a seed to training).
    def _get_parm_val(self, hparam_dict, param):
        if not param in hparam_dict:
            raise Exception('Parameter ' + str(param) + ' is missing in hparam_dict')
        return hparam_dict[param] #if not hparam_dict[param] is None else default

    def _keras_train_multi_response(self, resp_names:list[str], algo:str,
            X_train, X_test, y_train, y_test, hparam_dict, interactive_plots:bool, 
            seed, weights_coef, model_per_response:bool):
        layers_spec = self._get_parm_val(hparam_dict, self._hparam_name_local_to_global('layers', algo)) #DEF_LAYERS_SPEC 'nn_layers'
        epochs = self._get_parm_val(hparam_dict, self._hparam_name_local_to_global('epochs', algo)) #DEF_EPOCHS)
        batch_size = self._get_parm_val(hparam_dict, self._hparam_name_local_to_global('batch_size', algo)) #DEF_BATCH_SIZE
        optimizer = self._get_parm_val(hparam_dict, self._hparam_name_local_to_global('optimizer', algo)) #DEF_OPTIMIZER
        learning_rate = self._get_parm_val(hparam_dict, self._hparam_name_local_to_global('learning_rate', algo)) #DEF_OPTIMIZER
        loss_function_str = self._get_parm_val(hparam_dict, self._hparam_name_local_to_global('loss_function', algo)) #DEF_LOSS
        metrics_str_list = self._get_parm_val(hparam_dict, self._hparam_name_local_to_global('metrics', algo)) #DEF_METRICS
        hid_activation = self._get_parm_val(hparam_dict, self._hparam_name_local_to_global('hid_activation', algo)) #HID_ACTIVATION
        out_activation = self._get_parm_val(hparam_dict, self._hparam_name_local_to_global('out_activation', algo)) #OUT_ACTIVATION
        sequential_api = self._get_parm_val(hparam_dict, self._hparam_name_local_to_global('sequential_api', algo)) #SEQUENTIAL_API
        weights_precision = self._get_parm_val(hparam_dict, self._hparam_name_local_to_global('weights_precision', algo))
        tuner_algo = self._get_parm_val(hparam_dict, self._hparam_name_local_to_global('tuner_algo', algo))

        unknown_metrics = set(metrics_str_list).difference(set(self._metrics.keys()))
        if len(unknown_metrics) > 0:
            raise Exception('Unsupported metrics ' + str(unknown_metrics))
        
        # Get the loss function from the dictionary
        loss_function = self._loss_functions.get(loss_function_str)
        if not loss_function:
            raise ValueError(f"Unsupported loss function: {loss_function_str}")
        metrics = [self._metrics.get(metrics_str) for metrics_str in metrics_str_list]; #print('metrics', metrics_str_list, metrics)
        
        self._keras_logger.info('_keras_train_multi_response: start')
        #print('layers_spec =', layers_spec, '; seed =', seed, '; weights =', weights_coef)
        #print('epochs =', epochs, '; batch_size =', batch_size, '; optimizer =', optimizer)
        #print('hid_activation =', hid_activation, 'out_activation =', out_activation)

        # set the seed for reproducibility
        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)
            rn.seed(seed)
        
        # run Keras Tuner to optimize hyperparams, then train model using these hparam values
        if tuner_algo is not None:
            layers_grid = self._get_parm_val(hparam_dict, self._hparam_name_local_to_global('layers_grid', algo))
            if layers_grid is None:
                assert layers_spec is not None
                layers_grid = [layers_spec]
            lrates_grid = self._get_parm_val(hparam_dict, self._hparam_name_local_to_global('learning_rates_grid', algo))
            if lrates_grid is None:
                assert learning_rate is not None
                lrates_grid = [learning_rate]
            losses_grid = self._get_parm_val(hparam_dict, self._hparam_name_local_to_global('loss_functions_grid', algo))
            if lrates_grid is None:
                assert loss_function_str is not None
                lrates_grid = [loss_function_str]
            batches_grid = self._get_parm_val(hparam_dict, self._hparam_name_local_to_global('batches_grid', algo))
            if batches_grid is None:
                assert batch_size is not None
                batches_grid = [batch_size]
            tuned_model = self.train_best_model(resp_names, algo, X_train, X_test, y_train, y_test, sequential_api, 
                interactive_plots, seed, weights_coef, model_per_response, hid_activation, out_activation, epochs, weights_precision,
                layers_grid, losses_grid, lrates_grid, batches_grid, tuner_algo, batch_size, loss_function_str, metrics, learning_rate)
            
            self._keras_logger.info('_keras_train_multi_response: end')
            return tuned_model

        
        hyperparam_persist = {
            'train': {
                'layers': layers_spec,
                'epochs': epochs,
                'batch-size': batch_size,
                'optimizer': optimizer,
                'learning_rate': learning_rate,
                'loss_function': loss_function_str,
                'hid_activation': hid_activation,
                'out_activation': out_activation,
                'sequential_api': sequential_api,
                'seed': seed
            },
        }
        # TODO: need a different filename for persistance of hyperparameters
        with open(self.model_gen_file, 'w') as f:
            json.dump(hyperparam_persist, f)
        
        if False: # experiemnt with learning rate decay, to see whether it is worth adding it as an option
            # initial_learning_rate * decay_rate ^ (step / decay_steps)
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=10000,
                decay_rate=0.9)
            optimizer = Adam(learning_rate=lr_schedule); #print('optimizer', optimizer)
        else:
            optimizer = Adam(learning_rate=learning_rate); #print('optimizer', optimizer)
        input_dim = X_train.shape[1] #num_columns
        
        layers_spec_list = [float(x) for x in layers_spec.split(',')] #map(float, layers_spec.split(',')); 
        self._keras_logger.info('layers_spec_list ' + str(layers_spec_list))
        layers_spec_list = [ceil(float(e)*input_dim) for e in layers_spec_list]

        if sequential_api: #SEQUENTIAL_MODEL
            model = self._nn_init_model_sequential(resp_names, input_dim, optimizer, 
                hid_activation, out_activation, layers_spec_list, loss_function, metrics)
        else:
            model = self._nn_init_model_functional(resp_names, input_dim, optimizer, 
                hid_activation, out_activation, layers_spec_list, loss_function, metrics)
        
        history = self._nn_train(model, epochs, batch_size, weights_precision, self.model_checkpoint_pattern,
            X_train, X_test, y_train, y_test, weights_coef, sequential_api)

        # plot how training iterations improve error/model precision
        self._report_training_regression(history, metrics, epochs, interactive_plots, resp_names, self.report_file_prefix)

        self._keras_logger.info('_keras_train_multi_response: end')
        return model
        
    # Runs Keras NN algorithm, outputs lots of stats, saves model to disk
    # epochs and batch_size are arguments of NN algorithm from keras library
    def keras_main(self, resp_names:list[str], algo:str,
            X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.DataFrame, y_test:pd.DataFrame, hparam_dict:dict, 
            interactive_plots:bool, seed:float, weights_coef:dict, model_per_response:bool):
        self._keras_logger.info('keras_main: start')
        #print('resp_names', resp_names)
        #print('X_train', X_train.shape, 'X_test', X_test.shape, 'y_train', y_train.shape, 'y_test', y_test.shape)
        if model_per_response:
            model = {}
            for rn in resp_names:
                rn_weights_coef = {rn:weights_coef[rn]} if weights_coef is not None else weights_coef
                rn_model = self._keras_train_multi_response([rn], algo,
                    X_train, X_test, y_train[[rn]], y_test[[rn]], hparam_dict, interactive_plots, 
                    seed, rn_weights_coef, model_per_response)
                model[rn] = rn_model   
        else:
            model = self._keras_train_multi_response(resp_names, algo,
                X_train, X_test, y_train, y_test, hparam_dict, interactive_plots, 
                seed, weights_coef, model_per_response)
        self._keras_logger.info('keras_main: end')
        return model

    