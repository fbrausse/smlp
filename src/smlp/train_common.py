import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
from pycaret.regression import predict_model

from smlp.smlp_plot import evaluate_prediction
from smlp.train_keras import (keras_main, get_keras_hparam_deafult_dict, SMLP_KERAS_MODELS)
from smlp.train_caret import (caret_main, get_caret_hparam_deafult_dict, SMLP_CARET_MODELS)
from smlp.train_sklearn import (sklearn_main, get_sklearn_hparam_deafult_dict, SMLP_SKLEARN_MODELS)

keras_dict = get_keras_hparam_deafult_dict()
sklearn_dict = get_sklearn_hparam_deafult_dict()
caret_dict = get_caret_hparam_deafult_dict()


def pred_results_to_df(algo, resp_names, resp, pred):
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
def report_prediction_results(inst, algo, resp_names, resp_df, pred_df, #feat_df, 
        reverse_scale, mm_scaler_resp, interactive_plots, data_version):
    print('Reporting prediction results: start')
    
    #print('pred\n', pred_df); print('\npred_type', type(pred_df));
    #print('resp\n', resp_df); print('\nresp_type', type(resp_df));
    #print('feat\n', feat_df); print('\nfeat_type', type(resp_df));
    pred_colnames = [rn+'_'+algo for rn in resp_names]
    
    if reverse_scale:
        pred_df[ : ] = mm_scaler_resp.inverse_transform(pred_df)
        if not resp_df is None:
            resp_df[ : ] = mm_scaler_resp.inverse_transform(resp_df)
            
    predictions_df = pd.concat([resp_df, pred_df], axis=1) 
    print('Saving predictions summary into file', inst.predictions_summary_filename(data_version))
    predictions_df.to_csv(inst.predictions_summary_filename(data_version), index=True)
    #print('predictions_df\n', predictions_df) 
    
    # generate prediction precisions table / file
    if not resp_df is None:
        r2_vec = [r2_score(resp_df[resp_names[i]], pred_df[pred_colnames[i]]) for i in range(len(resp_names))]
        msqe_vec = [mean_squared_error(resp_df[resp_names[i]], pred_df[pred_colnames[i]]) for i in range(len(resp_names))]
        precisions_df = pd.DataFrame(data={'response' : resp_names, 'msqe' : msqe_vec, 'r2_score' : r2_vec})
        print('Saving prediction precisions into file', inst.prediction_precisions_filename(data_version))
        precisions_df.to_csv(inst.prediction_precisions_filename(data_version), index=False)
    
    if not resp_df is None:
        # renaming columns is required because evaluate_prediction is using column names
        # of resp_df to refer to values in pred_df
        pred_df.columns = resp_names
        evaluate_prediction(algo, resp_df, pred_df, data_version, interactive_plots,
                            out_prefix=inst._filename_prefix, log_scale=False)
        
    assert isinstance(pred_df, pd.DataFrame)
    assert isinstance(resp_df, pd.DataFrame) or resp_df is None
    
    print('Reporting prediction results: end')


# training model for all supported algorithms from verious python packages
def model_train(inst, input_names, resp_names, algo, X_train, X_test, y_train, y_test,
    hparams_dict, plots, seed, sample_weights, save_model, data=None):
    print('Model training: start')

    if algo == 'nn_keras':
        keras_algo = algo[:-len('_keras')]
        model = keras_main(inst, input_names, resp_names, keras_algo,
            X_train, X_test, y_train, y_test, hparams_dict, plots,
            seed, sample_weights, True, None)
    elif algo in ['dt_sklearn', 'et_sklearn', 'rf_sklearn', 'poly_sklearn']:
        sklearn_algo = algo[:-len('_sklearn')]
        if algo in ['dt_sklearn', 'et_sklearn', 'rf_sklearn']:
            model = sklearn_main(inst, input_names, resp_names, sklearn_algo,
                X_train, X_test, y_train, y_test, hparams_dict, plots, 
                seed, sample_weights, True)
        else:
            model, poly_reg, X_train, X_test = sklearn_main(inst, input_names, resp_names, sklearn_algo,
                X_train, X_test, y_train, y_test, hparams_dict, plots,
                seed, sample_weights, True)
    elif algo in SMLP_CARET_MODELS:
        caret_algo = algo[:-len('_caret')]
        model = caret_main(inst, input_names, resp_names, caret_algo,
            X_train, X_test, y_train, y_test, hparams_dict, plots,
            seed, sample_weights, False, True)       
    else:
        raise Exception('Unsuported algorithm ' + str(algo))
        
    print('Model training: end')
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
def model_predict(inst, model, X, y, resp_names : list, algo : str, data_version : str, 
        interactive_plots : bool):
    print('Model prediction: start')
    
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
    
    y_pred_df = pred_results_to_df(algo, resp_names, y, y_pred)
    assert type(y_pred_df) == type(pd.DataFrame())
    assert isinstance(y_pred_df, pd.DataFrame)
    print('Model prediction: end')
    return y_pred_df