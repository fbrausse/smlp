import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
from pycaret.regression import predict_model

from smlp.smlp_plot import evaluate_prediction
from smlp.train_keras import ModelKeras, keras_hparam_dict #keras_main, get_keras_hparam_deafult_dict, SMLP_KERAS_MODELS
from smlp.train_caret import ModelCaret, caret_hparam_dict #caret_main, get_caret_hparam_deafult_dict, SMLP_CARET_MODELS
from smlp.train_sklearn import ModelSklearn, sklearn_hparam_dict #sklearn_main, get_sklearn_hparam_deafult_dict, SMLP_SKLEARN_MODELS, 
from logs_common import create_logger

'''
# model training hyperparameters
keras_dict = get_keras_hparam_deafult_dict()
sklearn_dict = get_sklearn_hparam_deafult_dict()
caret_dict = get_caret_hparam_deafult_dict()
'''
model_params_dict = keras_hparam_dict | sklearn_hparam_dict | caret_hparam_dict


class ModelCommon:
    def __init__(self, log_file : str, log_level : str, log_mode : str, log_time : str):    
        #data_logger = logging.getLogger(__name__)
        self._model_logger = create_logger('model_logger', log_file, log_level, log_mode, log_time)
        self._instKeras = ModelKeras(log_file, log_level, log_mode, log_time)
        self._instSklearn = ModelSklearn(log_file, log_level, log_mode, log_time)
        self._instCaret = ModelCaret(log_file, log_level, log_mode, log_time)
        self._sklearn_dict = self._instSklearn.get_sklearn_hparam_default_dict()
        self._caret_dict = self._instCaret.get_caret_hparam_default_dict()
        self._keras_dict = self._instKeras.get_keras_hparam_default_dict()
        self.model_params_dict = self._keras_dict | self._sklearn_dict | self._caret_dict
        #print('model logger\n', self._model_logger)

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


    # generate out_dir/prefix_data_{train/test/labeled/new/}_prediction_precision.csv and 
    # out_dir/prefix_data_{train/test/labeled/new/}_prediction_summary.csv files with
    # msqe and R2_score (for now) precision columns and prediction results, respectively.
    # data_version indicates the data origin -- training (train), test (rather, validation), 
    # full labeled data (labeled) and new/unseen data (new).
    def report_prediction_results(self, inst, algo, resp_names, resp_df, pred_df,
            reverse_scale, mm_scaler_resp, interactive_plots, data_version):
        self._model_logger.info('Reporting prediction results: start')

        #print('pred\n', pred_df); print('\npred_type', type(pred_df));
        #print('resp\n', resp_df); print('\nresp_type', type(resp_df));
        pred_colnames = [rn+'_'+algo for rn in resp_names]

        orig_resp_df = resp_df.copy()
        orig_pred_df = pred_df.copy()

        if reverse_scale:
            orig_pred_df[ : ] = mm_scaler_resp.inverse_transform(pred_df)
            if not resp_df is None:
                orig_resp_df[ : ] = mm_scaler_resp.inverse_transform(resp_df)

        predictions_df = pd.concat([orig_resp_df, orig_pred_df], axis=1) 
        self._model_logger.info('Saving predictions summary into file: \n' + \
                                str(inst.predictions_summary_filename(data_version)))
        predictions_df.to_csv(inst.predictions_summary_filename(data_version), index=True)
        #print('predictions_df\n', predictions_df) 

        # generate prediction precisions table / file
        if not resp_df is None:
            r2_vec = [r2_score(orig_resp_df[resp_names[i]], orig_pred_df[pred_colnames[i]]) for i in range(len(resp_names))]
            msqe_vec = [mean_squared_error(orig_resp_df[resp_names[i]], orig_pred_df[pred_colnames[i]]) for i in range(len(resp_names))]
            precisions_df = pd.DataFrame(data={'response' : resp_names, 'msqe' : msqe_vec, 'r2_score' : r2_vec})
            self._model_logger.info('Saving prediction precisions into file: \n' + \
                                    str(inst.prediction_precisions_filename(data_version)))
            precisions_df.to_csv(inst.prediction_precisions_filename(data_version), index=False)

        if not resp_df is None:
            # renaming columns is required because evaluate_prediction is using column names
            # of resp_df to refer to values in orig_pred_df
            orig_pred_df.columns = resp_names
            legend = 'Prediction on ' + data_version + ' data -- '
            self._model_logger.info("{1} msqe: {0:.3f}".format(mean_squared_error(resp_df, pred_df), legend))
            self._model_logger.info("{1} r2_score: {0:.3f}".format(r2_score(resp_df, pred_df), legend))
    
            evaluate_prediction(algo, orig_resp_df, orig_pred_df, data_version, interactive_plots,
                                out_prefix=inst._filename_prefix, log_scale=False)

        assert isinstance(orig_pred_df, pd.DataFrame)
        assert isinstance(orig_resp_df, pd.DataFrame) or resp_df is None

        self._model_logger.info('Reporting prediction results: end')


    # extract hyperparameters required for training model with slgorithm algo
    # from args after it has been popylated with command-line and default values.
    def get_hyperparams_dict(self, args, algo):
        if algo in self._instKeras.SMLP_KERAS_MODELS:
            hparams_dict = dict((k, vars(args)[k]) for k in self._keras_dict.keys())
        elif algo in self._instSklearn.SMLP_SKLEARN_MODELS:
            hparams_dict = dict((k, vars(args)[k]) for k in self._sklearn_dict.keys())
        elif algo in self._instCaret.SMLP_CARET_MODELS:
            hparams_dict = dict((k, vars(args)[k]) for k in self._caret_dict.keys())
        else:
            raise Exception('Unsupprted model training algo ' + str(algo))
        return hparams_dict

    # compute sample weights per response or mean value of all reponses:
    # resp_vals is either a response column or mean of all reponses (per sample).
    def _sample_weights_per_response_vals(self, resp_vals, sw_coef):
        mid_range = (resp_vals.max() - resp_vals.min()) / 2
        w_coef = sw_coef / mid_range
        #print('resp_vals', resp_vals) ; 
        sw = [w_coef * (v - mid_range) + 1 for v in resp_vals]; #print('sw', sw)
        assert any([w >= 0 for w in sw])
        return np.array(sw)

    # compute sample weights per reponse using sample_weights_per_response_vals
    # and return as a dictionary with the reponses as keys and corresponding 
    # sample weight as the respective values. Required for training with algorithms
    # that can take sample weights per response (e.g., nn_keras).
    def _compute_sample_weights_dict(self, y_train, sw_coef):
        if sw_coef == 0:
            return None
        sw_dict = {}
        for resp in y_train.columns.tolist(): 
            #print('y_train', y_train[outp])
            sw = y_train[resp].values; print('sw', len(sw), type(sw))
            sw = sample_weights_per_response_vals(sw, sw_coef); #print('sw', len(sw))
            sw_dict[resp] = sw
        return sw_dict

    # compute sample weights for all responses by applying sample_weights_per_response_vals
    # to the vector of mean values of all responses (per sample). Required for training
    # for algorithms / packages that cannot take sample weights per response.
    def _compute_sample_weights_vect(self, y_train, sw_coef):
        #print('y_train\n', y_train, '\nsw_coef', sw_coef); 
        #print(y_train.shape[0]); print([1] * y_train.shape[0])
        if sw_coef == 0:
            return np.array([1] * y_train.shape[0])
        resp_vals = y_train.mean(axis='columns').values;
        return sample_weights_per_response_vals(resp_vals, sw_coef)

    # training model for all supported algorithms from verious python packages
    def model_train(self, inst, input_names, resp_names, algo, X_train, X_test, y_train, y_test,
        hparams_dict, plots, seed, sample_weights_coef, save_model, data=None):
        self._model_logger.info('Model training: start')

        if algo == 'nn_keras':
            keras_algo = algo[:-len('_keras')]
            sample_weights_dict = self._compute_sample_weights_dict(y_train, sample_weights_coef)
            model = self._instKeras.keras_main(inst, input_names, resp_names, keras_algo,
                X_train, X_test, y_train, y_test, hparams_dict, plots,
                seed, sample_weights_dict, True, None)
        elif algo in ['dt_sklearn', 'et_sklearn', 'rf_sklearn', 'poly_sklearn']:
            sklearn_algo = algo[:-len('_sklearn')]
            sample_weights_vect = self._compute_sample_weights_vect(y_train, sample_weights_coef)
            if algo in ['dt_sklearn', 'et_sklearn', 'rf_sklearn']:
                model = self._instSklearn.sklearn_main(inst, input_names, resp_names, sklearn_algo,
                    X_train, X_test, y_train, y_test, hparams_dict, plots, 
                    seed, sample_weights_vect, True)
            else:
                model, poly_reg, X_train, X_test = self._instSklearn.sklearn_main(inst, input_names, resp_names, sklearn_algo,
                    X_train, X_test, y_train, y_test, hparams_dict, plots,
                    seed, sample_weights_vect, True)
        elif algo in self._instCaret.SMLP_CARET_MODELS:
            caret_algo = algo[:-len('_caret')]
            sample_weights_vect = self._compute_sample_weights_vect(y_train, sample_weights_coef)
            model = self._instCaret.caret_main(inst, input_names, resp_names, caret_algo,
                X_train, X_test, y_train, y_test, hparams_dict, plots,
                seed, sample_weights_vect, False, True)       
        else:
            raise Exception('Unsuported algorithm ' + str(algo))

        self._model_logger.info('Model training: end')
        if algo == 'poly_sklearn':
            return model, poly_reg, X_train, X_test
        else:
            return model

    # function for prediction for the ML module.
    # The argument "model" is eaither a model trained using one of the supported packages
    # (e.g., sklearn, keras, caret) or a dictionary with the response names as keys and the 
    # correponding models as values. For example, the caret package currently does not support 
    # training of multiple reponses at once therefore we train a caret model per response and
    # combine prediction results as one np.array(). 
    def model_predict(self, inst, model, X, y, resp_names : list, algo : str):
        self._model_logger.info('Model prediction: start')

        model_lib = algo.rsplit('_', 1)[1]
        if model_lib in ['keras', 'sklearn']:
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
        elif model_lib == 'caret':
            # iterate over all responses and merge all predicitions into one return valee y_pred
            y_pred = pd.DataFrame(index=np.arange(y.shape[0]), columns=np.arange(0))
            for rn in model.keys():
                y_pred[rn] = list(predict_model(model[rn], data=X)['prediction_label']); 
            #print('y_pred df\n', y_pred)
            y_pred = np.array(y_pred); #print('y_pred array\n', y_pred)
        else:
            raise Exception('Unsupported model_lib ' + str(model_lib) + ' in function model_predict')

        y_pred_df = self._pred_results_to_df(algo, resp_names, y, y_pred)
        assert type(y_pred_df) == type(pd.DataFrame())
        assert isinstance(y_pred_df, pd.DataFrame)
        self._model_logger.info('Model prediction: end')
        return y_pred_df