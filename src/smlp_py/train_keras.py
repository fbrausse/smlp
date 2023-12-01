import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from math import ceil
import json
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score

# SMLP
from smlp_py.smlp_logs import *
from smlp_py.smlp_plots import plot  #, evaluate_model, evaluate_prediction, distplot_dataframe,
from smlp_py.smlp_utils import str_to_bool

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
        self._DEF_EPOCHS     = 2000
        self._DEF_BATCH_SIZE = 200
        self._DEF_OPTIMIZER  = 'adam'  # options: 'rmsprop', 'adam', 'sgd', 'adagrad', 'nadam'
        self._HID_ACTIVATION = 'relu'
        self._OUT_ACTIVATION = 'linear'
        self._SEQUENTIAL_API = True
        
        # model accuracy parameters
        self._DEF_LOSS       = 'mse'
        self._DEF_METRICS    = ['mse']    # ['mae'], ['mae','accuracy']
        self._DEF_MONITOR    = 'loss'     # monitor for keras.callbacks. options: 'loss', 'val_loss', 'accuracy', 'mae'
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
                'help':'specify number and sizes of the hidden layers of the '+
                        'NN as non-empty, comma-separated list of positive '+
                        'fractions in the number of input features in, e.g. '+
                        'second of half input size, third of quarter input size; '+
                        '[default: {}]'.format(self._DEF_LAYERS_SPEC)}, 
            'epochs': {'abbr':'epochs', 'default': self._DEF_EPOCHS, 'type':int,
                'help':'epochs for NN [default: {}]'.format(self._DEF_EPOCHS)}, 
            'batch_size': {'abbr':'batch', 'default': self._DEF_BATCH_SIZE, 'type':int,
                'help':'batch_size for NN [default: not exposed]'.format(self._DEF_BATCH_SIZE)}, 
            'optimizer': {'abbr':'optimizer', 'default': self._DEF_OPTIMIZER, 'type':str,
                'help':'optimizer for NN [default: {}]'.format(self._DEF_OPTIMIZER)},
            'hid_activation': {'abbr':'hid_activation', 'default': self._HID_ACTIVATION, 'type':str,
                'help':'hidden layer activation for NN [default: {}]'.format(self._HID_ACTIVATION)}, 
            'out_activation': {'abbr':'out_activation', 'default': self._OUT_ACTIVATION, 'type':str,
                'help':'output layer activation for NN [default: {}]'.format(self._OUT_ACTIVATION)},
            'sequential_api': {'abbr':'seq_api', 'default': self._SEQUENTIAL_API, 'type':str_to_bool,
                'help':'Should sequential api be used building NN layers or should functional ' +\
                        'api be used instead? [default: {}]'.format(str(self._SEQUENTIAL_API))}
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
    def _nn_init_model_sequential(self, input_dim, optimizer, hid_activation, out_activation, layers_spec, n_out):
        self._keras_logger.info('building NN model using Keras Sequential API')
        model = keras.models.Sequential()
        first = True
        layers_spec_list = [float(x) for x in layers_spec.split(',')] #map(float, layers_spec.split(',')); 
        self._keras_logger.info('layers_spec_list ' + str([1] + layers_spec_list))
        for fraction in layers_spec_list:
            assert fraction > 0
            n = ceil(input_dim * fraction);
            self._keras_logger.info('dense layer of size ' + str(n))
            model.add(keras.layers.Dense(n, activation=hid_activation,
                                         input_dim=input_dim if first else None))
            first = False
        model.add(keras.layers.Dense(n_out, activation=out_activation)) # OUT_ACTIVATION
        model.compile(optimizer=optimizer, loss=self._DEF_LOSS, metrics=self._DEF_METRICS)

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
    # initialize Leras NN model using functional API. It uses sample weights.
    def _nn_init_model_functional(self, input_dim, optimizer, hid_activation, out_activation, layers_spec, resp_names):
        self._keras_logger.info('building NN model using Keras Functional API')
        layers_spec_list = [float(x) for x in layers_spec.split(',')]
        layers_spec_list = [1] + layers_spec_list
        self._keras_logger.info('layers_spec_list ' + str(layers_spec_list))
        # layer 0 will be input layer, the rest internal layers. Outputs are defined separately below
        nn_layer_names = ['nn_layer_' + str(i) for i in range(len(layers_spec_list))]
        nn_layer_vars = [] # filled in as the input and internal layers are created
        nn_output_vars = [] # filled in as the outputs are created, each one separately as different layers
        for i in range(len(layers_spec_list)):
            fraction =  layers_spec_list[i]
            #print('fraction', fraction)
            assert fraction > 0

            # the size of the current layer
            n = ceil(input_dim * fraction)            
            if i == 0:
                self._keras_logger.info('input layer of size ' + str(n))
                assert fraction == 1
                globals()[nn_layer_names[i]] = keras.layers.Input(shape=(input_dim,))
            else:        
                self._keras_logger.info('dense layer of size ' + str(n))
                globals()[nn_layer_names[i]] = keras.layers.Dense(n, activation=hid_activation, input_dim=None)(globals()[nn_layer_names[i-1]])
                #print('var i-1', nn_layer_vars[i-1], 'gl name i-1', globals()[nn_layer_names[i-1]])
                assert (globals()[nn_layer_names[i-1]]) is (nn_layer_vars[i-1])
            nn_layer_vars.append(globals()[nn_layer_names[i]])
        #print('nn_layer_vars', nn_layer_vars)
        #print('nn_layer_names[-1]', nn_layer_names[-1])
        losses = {}
        for resp in resp_names:
            globals()[f"{resp}"] = keras.layers.Dense(1, name=resp, activation=out_activation)(globals()[nn_layer_names[-1]])
            losses[resp] = self._DEF_LOSS
            nn_output_vars.append(globals()[f"{resp}"])
        #print('nn_output_vars', nn_output_vars)    
        model = keras.models.Model(nn_layer_vars[0], nn_output_vars)
        #model.compile(optimizer=optimizer, loss=self._DEF_LOSS, metrics=self._DEF_METRICS)
        model.compile(optimizer=optimizer, loss=losses, metrics=self._DEF_METRICS)    
        #print("nn_init_model:model"); print(model)
        return model

    # train keras NN model
    def _nn_train(self, model, epochs, batch_size, model_checkpoint_path,
                 X_train, X_test, y_train, y_test, sample_weights_dict):

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

        # Build sequential model if and only if sample weights are defferent from the default 
        # (same weight for each sample). Without sample weights, a sequential model (built with 
        # sequential API) is expected to be better than a functional model.
        SEQUENTIAL_MODEL = (sample_weights_dict is None) #weights_coef == 0

        # train model with sequential or functional API
        if SEQUENTIAL_MODEL:
            history = model.fit(X_train, y_train,
                                epochs=epochs,
                                validation_data=(X_test, y_test),
                                #steps_per_epoch=10,
                                callbacks=[c for c in (checkpointer,earlyStopping,rlrop)
                                           if c is not None],
                                batch_size=batch_size)
        else:
            history = model.fit(X_train, y_train,
                                epochs=epochs,
                                validation_data=(X_test, y_test),
                                #steps_per_epoch=10,
                                sample_weight=sample_weights_dict,
                                callbacks=[c for c in (checkpointer,earlyStopping,rlrop)
                                           if c is not None],
                                batch_size=batch_size)

        return history


    def _report_training_regression(self, history, epochs, interactive, resp_names, out_prefix=None):
        #print('history.history', history.history)
        epochs_range = range(epochs)

        def plot_error_convergence(acc, resp_name):
            #plt.figure() -- commented out sinceotherwise an extra empty plot was displayed
            #plt.figure(figsize=(12, 5))
            #plt.subplot(1, 2, 1)
            plt.plot(epochs_range, acc, label='Training mse')
            plt.plot(epochs_range, val_acc, label='Validation mse')
            plt.legend(loc='upper right')
            plt.title('Response {} Training and Validation mse'.format(resp_name))
            plt.xlabel('epochs')
            plt.ylabel('MSE')
            ind_5 = len(acc)-int(len(acc)/10); #print('ind_5', ind_5)
            acc_5 = acc[-ind_5:]; #print('acc_5', acc_5)
            plt.ylim(0, max(acc_5))
            #plt.ylim(0, 2000)
            plot('train-reg', interactive, out_prefix)

        # When multiple responses are trained, the format (dictionary keys) of the training history
        # might be different (the there might be keys per respose and their names contain individual
        # response names). probably this depends whether particular responses required more/dedicated
        # training iterations (this is just a guess...). To adress this, below we have a case split
        # based on history_for_all_responses which checks whether specific response related keys occur 
        # in training hstory history.history:
        history_for_all_responses = self._HIST_MSE in history.history and self._HIST_VAL_MSE in history.history and \
            'loss' in history.history and 'val_loss' in history.history
        if len(resp_names) == 1 or history_for_all_responses:
            acc = history.history[self._HIST_MSE]; #print(acc)
            val_acc = history.history[self._HIST_VAL_MSE]
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            if len(resp_names) == 1:
                plot_error_convergence(acc, resp_names[0])
            else:
                plot_error_convergence(acc, 'all_responses')
        else:
            for rn in resp_names:
                acc = history.history[rn + '_' + self._HIST_MSE]; #print(acc)
                val_acc = history.history['val_' + rn + '_' + self._HIST_MSE]
                loss = history.history[rn + '_' + 'loss']
                val_loss = history.history['val_' + rn + '_loss']
                plot_error_convergence(acc, rn)


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

    def _keras_train_multi_response(self, feat_names, resp_names : list, algo,
            X_train, X_test, y_train, y_test, hparam_dict, interactive_plots, 
            seed, weights_coef, model_per_response:bool):
        layers_spec = self._get_parm_val(hparam_dict, self._hparam_name_local_to_global('layers', algo)) #DEF_LAYERS_SPEC 'nn_layers'
        epochs = self._get_parm_val(hparam_dict, self._hparam_name_local_to_global('epochs', algo)) #DEF_EPOCHS)
        batch_size = self._get_parm_val(hparam_dict, self._hparam_name_local_to_global('batch_size', algo)) #DEF_BATCH_SIZE
        optimizer = self._get_parm_val(hparam_dict, self._hparam_name_local_to_global('optimizer', algo)) #DEF_OPTIMIZER
        hid_activation = self._get_parm_val(hparam_dict, self._hparam_name_local_to_global('hid_activation', algo)) #HID_ACTIVATION
        out_activation = self._get_parm_val(hparam_dict, self._hparam_name_local_to_global('out_activation', algo)) #OUT_ACTIVATION
        sequential_api = self._get_parm_val(hparam_dict, self._hparam_name_local_to_global('sequential_api', algo)) #SEQUENTIAL_API
        
        # sample weights are supported only with functional api, therefore if sample weights are provided (different from None)
        # functional api will be used even if sequential_api was specifed as True
        # Old: SEQUENTIAL_MODEL = weights_coef == None; #print(weights_coef); print(SEQUENTIAL_MODEL)
        SEQUENTIAL_MODEL = sequential_api
        if not weights_coef is None:
            SEQUENTIAL_MODEL = False
        
        self._keras_logger.info('_keras_train_multi_response: start')
        #print('layers_spec =', layers_spec, '; seed =', seed, '; weights =', weights_coef)
        #print('epochs =', epochs, '; batch_size =', batch_size, '; optimizer =', optimizer)
        #print('hid_activation =', hid_activation, 'out_activation =', out_activation)

        # set the seed for reproducibility
        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)

        # TODO !!!: should we drop 'split-test': split_test from hyperparam_persist ; if we want to record 
        # split_test we can add it to a new (and common for all ML models) data_persist dictionary?
        hyperparam_persist = {
            'train': {
                'epochs': epochs,
                'batch-size': batch_size,
                'optimizer': optimizer,
                'hid_activation': hid_activation,
                'out_activation': out_activation,
                'seed': seed,
            },
        }
        # TODO: need a different filename for persistance of hyperparameters
        with open(self.model_gen_file, 'w') as f:
            json.dump(hyperparam_persist, f)

        input_dim = X_train.shape[1] #num_columns
        if SEQUENTIAL_MODEL:
            model = self._nn_init_model_sequential(input_dim, optimizer, hid_activation, out_activation, 
                layers_spec, len(resp_names))
        else:
            model = self._nn_init_model_functional(input_dim, optimizer, hid_activation, out_activation, 
                layers_spec, resp_names)

        history = self._nn_train(model, epochs, batch_size, self.model_checkpoint_pattern,
            X_train, X_test, y_train, y_test, weights_coef)

        # plot how training iterations improve error/model precision
        self._report_training_regression(history, epochs, interactive_plots, resp_names, self.report_file_prefix)

        #  print("evaluate")
        #  score = model.evaluate(x_test, y_test, batch_size=200)
        #  print(score)

        self._keras_logger.info('_keras_train_multi_response: end')
        return model
        
    # runs Keras NN algorithm, outputs lots of stats, saves model to disk
    # epochs and batch_size are arguments of NN algorithm from keras library
    def keras_main(self, feat_names, resp_names : list, algo,
            X_train, X_test, y_train, y_test, hparam_dict, interactive_plots, 
            seed, weights_coef, model_per_response:bool):
        self._keras_logger.info('keras_main: start')
        print('feat_names', feat_names, 'resp_names', resp_names)
        if model_per_response:
            model = {}
            for rn in resp_names:
                rn_model = self._keras_train_multi_response(feat_names, [rn], algo,
                    X_train, X_test, y_train[[rn]], y_test[[rn]], hparam_dict, interactive_plots, 
                    seed, weights_coef, model_per_response)
                model[rn] = rn_model   
        else:
            model = self._keras_train_multi_response(feat_names, resp_names, algo,
                X_train, X_test, y_train, y_test, hparam_dict, interactive_plots, 
                seed, weights_coef, model_per_response)
            
        self._keras_logger.info('keras_main: end')
        return model
