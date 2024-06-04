import pandas as pd
import numpy as np
import pickle
import json
import os

from sklearn.metrics import mean_squared_error, r2_score
from pycaret.regression import predict_model as caret_predict_model
from pycaret.regression import save_model as caret_save_model
from pycaret.regression import load_model as caret_load_model

from keras.models import load_model as keras_load_model

from smlp_py.smlp_plots import evaluate_prediction
from smlp_py.train_keras import ModelKeras 
from smlp_py.train_caret import ModelCaret 
from smlp_py.train_sklearn import ModelSklearn
from smlp_py.smlp_utils import str_to_bool

# Methods for model training, prediction, results reporting (including plots), exporting model formulae.
# Currently supports multiple (but not all) training algorithms from Keras, Sklearm and Caret packages.
# Model training parameter model_per_response controls whther one model is build that covers all responses
# or a separate model is built for each response. In the latter case, when MRMR option mrmr_pred is on,
# model for each response is built from the subset of features selected by MRMR algorithm fro that 
# response -- these subsets of features might be different for different responses. Also, in this case
# (when model_per_response is true) result of training is a dictionary with response names as keys and
# the model trained for a given response as the correponding value in the dictionary. When model_per_response
# is false, the trained model is not a dictionary, it is a model of the type that corresponds to the training
# algorithm; and in this case the features used for training are all features as specifed in command line
# if MRMR is not used, and otherwise is the union of features selected by MRMR for at least one response.
# In model exploration modes (lie verification, querying, optimixzation) if SMLP terms and solver instances
# need to be built, each model in the dictionary of the models or a model trained for all responses is 
# converted to terms separtely, the constraints and assertions built on the top of model responses are added
# to solver instance separately (as many as required, depending on whether all responses are anlysed together).
class SmlpModels:
    def __init__(self):
        self._model_logger = None
        self._DEF_SAVE_MODEL = True
        self._DEF_USE_MODEL = False
        self._DEF_SAVE_MODEL_CONFIG = True
        self._MODEL_PER_RESPONSE = False
        self._DEF_PREDICTION_PLOTS = True

        self._model_params_common_dict = {
            'model': {'abbr': 'model', 'type':str,
                'help': 'Type of model to train (NN, Poly, ... [default: none]'},
            'save_model': {'abbr':'save_model', 'default': self._DEF_SAVE_MODEL, 'type':str_to_bool,
                'help': 'Should the trained models be saved for future use? ' +
                    '[default: ' + str(self._DEF_SAVE_MODEL) + ']'},
            'use_model': {'abbr':'use_model', 'default': self._DEF_USE_MODEL, 'type':str_to_bool,
                'help': 'Should the saved models be reused (and training skipped)? ' +
                    '[default: ' + str(self._DEF_USE_MODEL) + ']'},
            'model_name': {'abbr':'model_name', 'type':str,
                'help': 'Name of saved model. If not specified, the name is defined as follows: ' +
                    'filename_prefix + "_" + model_algo + "_model_complete" + model_format ' +
                    'where filename_prefix is concatenation of the output directory and the prefix '
                    'identifying the run, model_algo is the training algo name and model_format ' +
                    'is .h5 for nn_keras and .pkl for models trained using sklearn and keras packages.'},
            'save_model_rerun_configuration': {'abbr':'save_model_config', 
                    'default': self._DEF_SAVE_MODEL_CONFIG, 'type':str_to_bool,
                'help': 'Should a config file enabling to re-run a saved model be written out? ' +
                    '[default: ' + str(self._DEF_SAVE_MODEL_CONFIG) + ']'},
            'model_per_response': {'abbr':'model_per_response', 'default': self._MODEL_PER_RESPONSE, 'type':str_to_bool,
                'help': 'Should a separate model, possible with a different, dedicated feature set, ' +
                    'be built per response (as opposite to building one multi-response model)?' +
                    '[default: ' + str(self._MODEL_PER_RESPONSE) + ']'},
            'prediction_plots': {'abbr':'pred_plots', 'default': self._DEF_PREDICTION_PLOTS, 'type':str_to_bool,
                'help': 'Should response distribution plots and plots comparing response values in ' +
                    'data with the predicted values be generated? A related option interactive_plots ' +
                    'controls whether the generated plots should be displayed interactively during runtime ' +
                    '[default: ' + str(self._DEF_PREDICTION_PLOTS) + ']'}
        }
        self._instKeras = ModelKeras()
        self._instSklearn = ModelSklearn()
        self._instCaret = ModelCaret()
        self._sklearn_dict = self._instSklearn.get_sklearn_hparam_default_dict()
        self._caret_dict = self._instCaret.get_caret_hparam_default_dict()
        self._keras_dict = self._instKeras.get_keras_hparam_default_dict()
        self.model_params_dict = self._model_params_common_dict | self._keras_dict | self._sklearn_dict | self._caret_dict
    
    # report_file_prefix is a string used as prefix in all report files of SMLP
    def set_report_file_prefix(self, report_file_prefix):
        self.report_file_prefix = report_file_prefix
        self._instKeras.report_file_prefix = report_file_prefix
        self._instSklearn.report_file_prefix = report_file_prefix
        self._instCaret.report_file_prefix = report_file_prefix
        
    # model_file_prefix is a string used as prefix in all outut files of SMLP that are used to 
    # save a trained ML model and to re-run the model on new data (without need for re-training)
    def set_model_file_prefix(self, model_file_prefix):
        self.model_file_prefix = model_file_prefix
        self._instKeras.model_file_prefix = model_file_prefix
        self._instSklearn.model_file_prefix = model_file_prefix
        self._instCaret.model_file_prefix = model_file_prefix
    
    # required for generating file names of the reports containing model prediction results;
    # might cover multiple models (algorithms like NN, DT, RF) as well as multiple responses
    def predictions_summary_filename(self, data_version):
        assert self.report_file_prefix is not None
        return self.report_file_prefix + '_' + data_version + '_predictions_summary.csv'
    
    # required for generating file names of the reports containing model precision (rmse, R2, etc.) reults;
    # might cover multiple models (algorithms like NN, DT, RF) as well as multiple responses
    def prediction_precisions_filename(self, data_version):
        assert self.report_file_prefix is not None
        return self.report_file_prefix + '_' + data_version + '_prediction_precisions.csv'
        
    # saved model file name -- model_algo is the training algo name like nn_keras, sklearn_dt;
    # model_format is the format used to save the model -- for nn_keras .h5 format is used,
    # (implies suffix .h5); for sklearn and caret models pickle format is used (implies suffix .pkl)
    def model_filename(self, model_algo, model_format, response=None):
        assert self.model_file_prefix is not None
        if response is None:
            return self.model_file_prefix + '_' + model_algo + '_model_complete' + model_format
        else:
            return self.model_file_prefix + '_' + str(response) + '_' + model_algo + '_model_complete' + model_format

    # configuration file created when saving a trained model, and loaded when re-using
    # the model
    @property
    def model_rerum_config_file(self):
        return self.model_file_prefix + '_rerun_model_config.json'
    
    # Compute file name prefix (including the directory path) to model files.
    # If there are multiple responses in the "responses" argument, the response names
    # are ignored in model file prefix generation. We also want the response name
    # to be ignored when there is only one response, say called resp_name, 
    # therefore in this case the function should be called with None for argument 
    # "responses" instead of passing list [resp_name] for the "responses" argument.
    def get_model_file_prefix(self, responses=None, algo=None):
        assert self.model_file_prefix is not None
        res = self.model_file_prefix
        if not responses is None:
            if len(responses) == 1:
                res = res + '_' + str(responses[0])
        if not algo is None:
            res = res + '_' + str(algo)
        return res

    # save model configuration file enabling to re-run a saved model with the same  
    # configuration for which it was generated
    def _save_model_rerun_config(self, model_rerun_config):
        if model_rerun_config is not None:
            with open(self.model_rerum_config_file, 'w') as f:
                f.write(json.dumps(model_rerun_config,  indent=4, sort_keys=True))
                f.close()

    # load model configuration file enabling to re-run a saved model with the same  
    # configurationfor which it was generated
    def _load_model_rerun_config(self):
        self._model_logger.info('Seving model rerun configuration in file ' + \
            str(self.model_rerum_config_file))
        if os. path. exists(self.model_rerum_config_file):
            with open(self.model_rerum_config_file, 'r') as f:
                return json.load(f)
        else:
            return None
    
    # sanity check that the order of features in model_features_dict, feat_names, X_train, X_test, X is 
    # the same; this is mostly important for model exploration modes 
    def model_features_sanity_check(self, model_features_dict:dict, feat_names:list[str], train_df:pd.DataFrame, 
            test_df:pd.DataFrame, feat_df:pd.DataFrame):
        if feat_names is not None: 
            if train_df is not None:
                assert feat_names == train_df.columns.tolist()
            if test_df is not None:
                assert feat_names == test_df.columns.tolist()    
            if feat_df is not None:
                assert feat_names == feat_df.columns.tolist()
        if model_features_dict is not None:
            for resp, feats in model_features_dict.items():
                if feat_names is not None:
                    feats_ordered = [ft for ft in feat_names if ft in feats]
                    assert feats == feats_ordered
                if train_df is not None:
                    feats_ordered = [ft for ft in train_df.columns.tolist() if ft in feats]
                    assert feats == feats_ordered
                if test_df is not None:
                    feats_ordered = [ft for ft in test_df.columns.tolist() if ft in feats]
                    assert feats == feats_ordered
                if feat_df is not None:
                    feats_ordered = [ft for ft in feat_df.columns.tolist() if ft in feats]
                    assert feats == feats_ordered
    
    
    # Several of model training packages return prediction results as np.array.
    # This function converts prediction results from np.array to pd.DataFrame.
    def _pred_results_to_df(self, algo, resp_names, resp, pred):
        #print('pred\n', pred); print('\npred_type', type(pred));
        #print('resp\n', resp); print('\nresp_type', type(resp));
        predictions_colnames = [rn+'_'+algo for rn in resp_names]

        # expecting here pred to be np array (while resp is expected to be a data frames)
        assert resp is None or type(resp) == type(pd.DataFrame())
        assert type(pred) == type(np.array([]))
        if not isinstance(pred[0], list):
            # we have a single response prediction
            if not resp is None:
                if len(pred) != resp.shape[0]:
                    raise Exception('Implementation error in function pred_results_to_df')   
            pred_ind = resp.index if not resp is None else range(len(pred)); #print('pred_ind', pred_ind)
            predictions_df = pd.DataFrame(pred, index=pred_ind, columns=predictions_colnames)
        else:
            # we have multiple response prediction
            pred_ind = resp.index if not resp is None else range(pred.shape[0]); #print('pred_ind', pred_ind)
            predictions_df = pd.DataFrame(pred, index=pred_ind, columns=predictions_colnames)

        assert type(predictions_df) == type(pd.DataFrame())
        assert isinstance(predictions_df, pd.DataFrame)
        return predictions_df

    
    # compute sample weights per response or mean value of all responses:
    # resp_vals is either a response column or mean of all responses (per sample).
    def _sample_weights_per_response_vals(self, resp_vals, sw_coef):
        mid_range = (resp_vals.max() - resp_vals.min()) / 2
        w_coef = sw_coef / mid_range
        #print('resp_vals', resp_vals) ; 
        sw = [w_coef * (v - mid_range) + 1 for v in resp_vals]; #print('sw', sw)
        assert any([w >= 0 for w in sw])
        return np.array(sw)

    # compute sample weights per response using _sample_weights_per_response_vals
    # and return as a dictionary with the responses as keys and the corresponding 
    # sample weight as the respective values. Required for training with algorithms
    # that can take sample weights per response (e.g., nn_keras).
    def _compute_sample_weights_dict(self, y_train, sw_coef):
        if sw_coef == 0:
            return None
        sw_dict = {}
        for resp in y_train.columns.tolist(): 
            #print('y_train', y_train[outp])
            sw = y_train[resp].values; #print('sw', len(sw), type(sw))
            sw = self._sample_weights_per_response_vals(sw, sw_coef); #print('sw', len(sw))
            sw_dict[resp] = sw
        return sw_dict

    # compute sample weights for all responses by applying _sample_weights_per_response_vals
    # to the vector of mean values of all responses (per sample). Required for training
    # for algorithms / packages that cannot take sample weights per response.
    def _compute_sample_weights_vect(self, y_train, sw_coef):
        #print('y_train\n', y_train, '\nsw_coef', sw_coef); 
        #print(y_train.shape[0]); print([1] * y_train.shape[0])
        if sw_coef == 0:
            return np.array([1] * y_train.shape[0])
        resp_vals = y_train.mean(axis='columns').values;
        return self._sample_weights_per_response_vals(resp_vals, sw_coef)
    
    # set a logger to ModelCommon from the caller script
    # then set the same logger to the used instances of ModelKeras, ModelCaret, ModelSklearn
    def set_logger(self, logger):
        self._model_logger = logger 
        self._instKeras.set_logger(logger)
        self._instCaret.set_logger(logger)
        self._instSklearn.set_logger(logger)
    
    # generate out_dir/prefix_data_{train/test/labeled/new/}_prediction_precision.csv and 
    # out_dir/prefix_data_{training/test/labeled/new/}_prediction_summary.csv files with
    # msqe and R2_score (for now) precision columns and prediction results in the original
    # scale of training data, respectively. The argument data_version indicates the data origin:
    # training (train), test/validation, full labeled data (labeled) and new/unseen data (new).
    def _report_prediction_results(self, algo:str, resp_names:list[str], resp_df:pd.DataFrame, pred_df:pd.DataFrame,
            mm_scaler_resp, interactive_plots:bool, prediction_plots:bool, data_version:str):
        self._model_logger.info('Reporting prediction results: start')
        #print('pred\n', pred_df); print('\npred_type', type(pred_df));
        #print('resp\n', resp_df); print('\nresp_type', type(resp_df));
        pred_colnames = [rn+'_'+algo for rn in resp_names]
        assert pred_df.columns.tolist() == pred_colnames
        
        orig_resp_df = resp_df.copy() if not resp_df is None else None
        orig_pred_df = pred_df.copy(); 
        #print('orig_pred_df\n', orig_pred_df); print('pred_df\n', pred_df); 
        
        if not mm_scaler_resp is None:
            orig_pred_df[ : ] = mm_scaler_resp.inverse_transform(pred_df)
            if not resp_df is None:
                orig_resp_df[ : ] = mm_scaler_resp.inverse_transform(resp_df) 
        #print('orig_resp_df\n', orig_resp_df); print('orig_pred_df\n', orig_pred_df)
        predictions_df = pd.concat([orig_resp_df, orig_pred_df], axis=1) 
        self._model_logger.info('Saving predictions summary into file: \n' + \
                                str(self.predictions_summary_filename(data_version)))
        predictions_df.to_csv(self.predictions_summary_filename(data_version), index=True)
        #print('predictions_df\n', predictions_df) 

        # generate prediction precisions table / file
        if not resp_df is None:
            r2_vec = [r2_score(orig_resp_df[resp_names[i]], orig_pred_df[pred_colnames[i]]) 
                for i in range(len(resp_names))]
            msqe_vec = [mean_squared_error(orig_resp_df[resp_names[i]], orig_pred_df[pred_colnames[i]]) 
                for i in range(len(resp_names))]
            precisions_df = pd.DataFrame(data={'response' : resp_names, 'msqe' : msqe_vec, 'r2_score' : r2_vec})
            self._model_logger.info('Saving prediction precisions into file: \n' + \
                                    str(self.prediction_precisions_filename(data_version)))
            precisions_df.to_csv(self.prediction_precisions_filename(data_version), index=False)

        if not resp_df is None:
            # renaming columns is required because evaluate_prediction is using column names
            # of resp_df to refer to values in orig_pred_df
            orig_pred_df.columns = resp_names
            legend = 'Prediction on ' + data_version + ' data -- '
            self._model_logger.info("{1} msqe: {0:.3f}".format(mean_squared_error(orig_resp_df, orig_pred_df), legend))
            self._model_logger.info("{1} r2_score: {0:.3f}".format(r2_score(orig_resp_df, orig_pred_df), legend))
            if prediction_plots:
                evaluate_prediction(algo, orig_resp_df, orig_pred_df, data_version, interactive_plots,
                                    out_prefix=self.report_file_prefix, log_scale=False)
        
        assert isinstance(orig_pred_df, pd.DataFrame)
        assert isinstance(orig_resp_df, pd.DataFrame) or resp_df is None

        self._model_logger.info('Reporting prediction results: end')


    # extract hyperparameters required for training model with slgorithm algo
    # from args after it has been populated with command-line and default values.
    def get_hyperparams_dict(self, args, algo):
        if algo in self._instKeras.SMLP_KERAS_MODELS:
            hparams_dict = dict((k, vars(args)[k]) for k in self._keras_dict.keys())
        elif algo in self._instSklearn.SMLP_SKLEARN_MODELS:
            #print('sklearn_dict', self._sklearn_dict)
            hparams_dict = dict((k, vars(args)[k]) for k in self._sklearn_dict.keys())
        elif algo in self._instCaret.SMLP_CARET_MODELS:
            hparams_dict = dict((k, vars(args)[k]) for k in self._caret_dict.keys())
        else:
            raise Exception('Unsupprted model training algo ' + str(algo))
        return hparams_dict
    
    # training model for all supported algorithms from verious python packages
    def model_train(self, feat_names_dict:dict, resp_names:list, algo, 
            X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.DataFrame, y_test:pd.DataFrame,
            hparams_dict:dict, plots:bool, seed:int, sample_weights_coef:float, model_per_response:bool):
        self._model_logger.info('Model training: start')
        self.model_features_sanity_check(feat_names_dict, None, X_train, X_test, None)
            
        if algo == 'nn_keras':
            keras_algo = algo[:-len('_keras')]
            sample_weights_dict = self._compute_sample_weights_dict(y_train, sample_weights_coef)
            model = self._instKeras.keras_main(feat_names_dict, resp_names, keras_algo,
                X_train, X_test, y_train, y_test, hparams_dict, plots,
                seed, sample_weights_dict, model_per_response)
        elif algo in ['dt_sklearn', 'et_sklearn', 'rf_sklearn', 'poly_sklearn']:
            sklearn_algo = algo[:-len('_sklearn')]
            sample_weights_vect = self._compute_sample_weights_vect(y_train, sample_weights_coef)
            model = self._instSklearn.sklearn_main(self.get_model_file_prefix, feat_names_dict, resp_names, sklearn_algo,
                X_train, X_test, y_train, y_test, hparams_dict, plots, 
                seed, sample_weights_vect, model_per_response)
        elif algo in self._instCaret.SMLP_CARET_MODELS:
            caret_algo = algo[:-len('_caret')]
            sample_weights_vect = self._compute_sample_weights_vect(y_train, sample_weights_coef)
            model = self._instCaret.caret_main(self.get_model_file_prefix, feat_names_dict, resp_names, caret_algo,
                X_train, X_test, y_train, y_test, hparams_dict, plots,
                seed, sample_weights_vect, False)       
        else:
            raise Exception('Unsuported algorithm ' + str(algo))

        self._model_logger.info('Model training: end')
        return model

    # function for prediction for the ML module.
    # The argument "model" is eaither a model trained using one of the supported packages
    # (e.g., sklearn, keras, caret) or a dictionary with the response names as keys and the 
    # correponding models as values. For example, the caret package currently does not support 
    # training of multiple responses at once therefore we train a caret model per response and
    # combine prediction results as one np.array(). 
    def _model_predict(self, model, X:pd.DataFrame, y:pd.DataFrame, resp_names:list, algo:str, model_per_response:bool):
        self._model_logger.info('Model prediction: start')
        model_lib = algo.rsplit('_', 1)[1]
        if model_lib in ['keras', 'sklearn'] and not model_per_response:
            # we have a single model
            if algo == 'poly_sklearn':
                y_pred = self._instSklearn.poly_sklearn_predict(model, X)
            else:
                y_pred = model.predict(X)
            # NN Keras might return a list of lists or a list of np.array-s as prediction results.
            # List of lists is a more common format for predicted results in python, therefore
            # here if y_pred is a list of np.array-s then we transform it into a list of lists 
            # (each list within this list of lists correponds to a row in pandas dataframe).
            if algo == 'nn_keras' and len(resp_names) > 1 and isinstance(y_pred, list):
                # format y_pred as np.array with each column being predction of one response
                #print('y_pred\n', y_pred)
                if isinstance(y_pred[0], np.ndarray):
                    y_pred = np.concatenate(y_pred, axis=1)
        elif model_lib == 'caret' or model_per_response:
            # we have multiple models (if there are multiple responses) -- one per response
            # iterate over all responses and merge all predicitions into one return value y_pred
            y_pred = pd.DataFrame(index=np.arange(X.shape[0]), columns=np.arange(0))
            for rn in model.keys():
                #print('rn', rn); print('model_dict', model); print('model', model[rn])
                if model_lib == 'caret':
                    y_pred[rn] = list(caret_predict_model(model[rn], data=X)['prediction_label']); 
                elif model_lib in ['keras', 'sklearn']:
                    if algo == 'poly_sklearn':
                        #print('X', X.columns.tolist())
                        y_pred[rn] = self._instSklearn.poly_sklearn_predict(model[rn], X)
                        #rn_model, rn_poly_reg = model[rn] #, rn_X_train, rn_X_test
                        #y_pred[rn] = rn_model.predict(rn_poly_reg.transform(X))
                    else:
                        #print('model', model); print(' model[rn]',  model[rn])
                        #print('model[rn].predict(X)', model[rn].predict(X))
                        y_pred[rn] = model[rn].predict(X)
                else:
                    assert False
            #print('y_pred df\n', y_pred)
            y_pred = np.array(y_pred); #print('y_pred array\n', y_pred)
        else:
            raise Exception('Unsupported model_lib ' + str(model_lib) + ' in function _model_predict')
        
        y_pred_df = self._pred_results_to_df(algo, resp_names, y, y_pred)
        assert type(y_pred_df) == type(pd.DataFrame())
        assert isinstance(y_pred_df, pd.DataFrame)
        self._model_logger.info('Model prediction: end')
        return y_pred_df

    
    # training, validation, testing of models and prediction on training, test, entire labeled data
    # as well as new data, if available. Reporting model prediction results as well as accuracy scores.
    def build_models(self, algo:str, X:pd.DataFrame, y:pd.DataFrame, X_train:pd.DataFrame, y_train:pd.DataFrame, 
            X_test:pd.DataFrame, y_test:pd.DataFrame, X_new:pd.DataFrame, y_new:pd.DataFrame, resp_names:list,
            mm_scaler_feat, mm_scaler_resp, levels_dict:dict, feat_names_dict:dict, 
            hparams_dict:dict, plots:bool, pred_plots:bool, seed:int, sample_weights_coef:float, save_model:bool, 
            use_model:bool, model_per_response:bool, model_rerun_config:dict):
        if not y_train is None:
            assert resp_names == y_train.columns.tolist()
        if not y_test is None:
            assert resp_names == y_test.columns.tolist()
        
        model_lib = algo.rsplit('_', 1)[1]
        if use_model:
            self._model_logger.info('LOAD TRAINED MODEL')
            model_rerun_config_dict = self._load_model_rerun_config()
            if model_lib =='sklearn':
                model = pickle.load(open(self.model_filename(algo, '.pkl'), 'rb'))
            elif model_lib == 'caret':
                # caret currently does not support training models with multiple responses 
                # (or we missed to see in documentation how this is done); thus caret trained
                # models are always dictionaries with responses as keys and models per response as values
                #print('model file', [self.model_filename(algo, '', resp_name) for resp_name in resp_names])
                model = dict([(resp_name, caret_load_model(self.model_filename(algo, '', resp_name))) 
                    for resp_name in resp_names])
            elif model_lib == 'keras':
                if model_per_response:
                    if model_rerun_config_dict is not None:
                        #print('model_per_response', model_per_response)
                        #print(' model_rerun_config_dict[model_per_response]',  model_rerun_config_dict['model_per_response'])
                        assert model_rerun_config_dict['model_per_response'] == model_per_response
                    # models are dictionaries with responses as keys and models per response as values
                    model = dict([(resp_name, keras_load_model(self.model_filename(algo, '.h5', resp_name))) 
                        for resp_name in resp_names])
                else:
                    model = keras_load_model(self.model_filename(algo, '.h5'))
            else:
                raise Exception('Unsupported lib (package) ' + str(model_lib) + ' in function build_models')
        else:
            # run model training
            self._model_logger.info('TRAIN MODEL')
            #feat_names = X_train.columns.tolist()
            model = self.model_train(feat_names_dict, resp_names, algo, X_train, X_test, y_train, y_test,
                hparams_dict, plots, seed, sample_weights_coef, model_per_response)

            if save_model:
                self._save_model_rerun_config(model_rerun_config)
                if model_lib == 'sklearn':
                    self._model_logger.info('Seving model in file ' + \
                        str(self.model_filename(algo, '.pkl')))
                    pickle.dump(model, open(self.model_filename(algo, '.pkl'), 'wb'))
                elif model_lib == 'caret':
                    # when saving a model, save_model() adds '.pkl' suffic to the filename supplied to it;
                    # we therefore use '' instead of 'pkl' when computing model_filename
                    # argument model is actually a dict with response names as keys and the correponding 
                    # models as values, thus we save/dump multiple models, one for each response:
                    for resp in model:
                        caret_save_model(model[resp], self.model_filename(algo, '', resp))
                    #pickle.dump(model, open(self.model_filename(algo, '.pkl'), 'wb'))
                elif model_lib == 'keras':
                    # could save the model in two ways; currently saved model in json format is not used
                    if isinstance(model, dict):
                        for resp in model:
                            model[resp].save(self.model_filename(algo, '.h5', resp))
                    else:
                        model.save(self.model_filename(algo, '.h5'))
                else:
                    raise Exception('Unsupported lib (package) ' + str( model_lib) + ' in function build_models')
        
        if not X_train is None and not y_train is None:
            self._model_logger.info('PREDICT ON TRAINING DATA')
            #print('(2)'); print('y\n', y);  print('y_train\n', y_train); print('y_test\n', y_test);
            y_train_pred = self._model_predict(model, X_train, y_train, resp_names, algo, model_per_response)
            self._report_prediction_results(algo, resp_names, y_train, y_train_pred, mm_scaler_resp,
                plots, pred_plots, 'training')
        
        if not X_test is None and not y_test is None:
            self._model_logger.info('PREDICT ON TEST DATA')
            #print('(3)'); print('y\n', y);  print('y_train\n', y_train); print('y_test\n', y_test);
            y_test_pred = self._model_predict(model, X_test, y_test, resp_names, algo, model_per_response)
            #print('(3b)'); print('y\n', y);  print('y_train\n', y_train); print('y_test\n', y_test); 
            self._report_prediction_results(algo, resp_names, y_test, y_test_pred, mm_scaler_resp, 
                plots, pred_plots, 'test')

        if X is not None and y is not None:
            self._model_logger.info('PREDICT ON LABELED DATA')
            #print('(4)'); print('y\n', y);  print('y_train\n', y_train); print('y_test\n', y_test); 
            # In case a polynomial model was run, polynomial features have been added to X_train and X_test,
            # therefore we need to reconstruct X before evaluating the model on all labeled features. 
            # once X has been updated, we need to update y as well in case X_train/y_train and/or X_test/y_test
            # was modified after generating them from X,y and before feeding to training.
            # use original X vs using X_train and X_test that were generated using sampling X_tran and/or X_test
            run_on_orig_X = True         
            #if run_on_orig_X and algo == 'poly_sklearn':
            #    X = poly_reg.transform(X)
            if not run_on_orig_X:
                X = np.concatenate((X_train, X_test)); 
                y = pd.concat([y_train, y_test])
            #print('(4)'); print('y\n', y)
            y_pred = self._model_predict(model, X, y, resp_names, algo, model_per_response)
            self._report_prediction_results(algo, resp_names, y, y_pred, mm_scaler_resp,
                plots, pred_plots, 'labeled')

        if X_new is not None:
            self._model_logger.info('PREDICT ON NEW DATA')
            #print('(5)'); print('y_new\n', y_new)
            y_new_pred = self._model_predict(model, X_new, y_new, resp_names, algo, model_per_response)
            #print('y_new\n', y_new, '\ny_new_pred\n', y_new_pred)
            self._report_prediction_results(algo, resp_names, y_new, y_new_pred, mm_scaler_resp, 
                plots, pred_plots, 'new')

        return model
