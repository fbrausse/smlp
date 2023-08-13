# data processing -- mainly to prepare data for model training
import os
import logging
import numpy as np
import pandas as pd
import pickle
import json

#from mrmr import mrmr_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from smlp.smlp_plot import response_distribution_plot
from utils_common import np_JSONEncoder, list_intersection, lists_union_order_preserving_without_duplicates
from smlp.mrmr_features import MrmrFeatures
# Methods for data processing, traing vs test splitting, handling responses vs features, and more.
class DataCommon:
    def __init__(self):
        #data_logger = logging.getLogger(__name__)
        self._data_logger = None # create_logger('data_logger', log_file, log_level, log_mode, log_time)
        self._pathInst = None # will be set by a caller / application
        # TODO !!!: currently we instantiate MrmrFeatures here within DataCommon calss because MRMR is only
        # used as part of data processing before modeling. In the furture MRMR will be used as part of feature
        # selection say in Rage Analysis (and model building might or might not be required in the overall flow)
        # and in that case we might need to instantiate MrmrFeatures at a top level script (like smslp_train.py)
        # along with instantiating DataCommon (and maybe ModelsCommon).
        self._mrmrInst = MrmrFeatures() 
        # default values of parameters related to dataset; used to generate args.
        self._DEF_TRAIN_FIRST = 0 # subseting first_n rows from training data
        self._DEF_TRAIN_RAND = 0  # sampling random_n rows from training data
        self._DEF_TRAIN_UNIF = 0  # sampling from training data to acheive uniform distribution
        self._DEF_SCALER = 'min_max'  # options: 'min_max', 'max-abs'
        self._DEF_SPLIT_TEST = 0.2 # ratio to split training data into training and validation subsets
        self._DEF_SAMPLE_WEIGHTS = 0 # degree/exponent of the power function that computes the
                                     # sample weights based on response values on these samples
            
        # default values of dataset processing related parameters; used to generate args parser.
        self.data_params_dict = {
            'response': {'abbr':'resp', 'default':None, 'type':str,
                'help':'Names of response variables, must be provided [default None]'}, 
            'features': {'abbr':'feat', 'default':None, 'type':str,
                'help':'Names of input features (can be computed from data) [default None]'},
            'new_data': {'abbr':'new_data', 'default':None, 'type':str,
                'help':'Path excluding the .csv suffix to new data file [default: None]'},
            'data_scaler': {'abbr':'data_scaler', 'default':self._DEF_SCALER, 'type':str, 
                'help':'Should features and responses be scaled and with which scaling optionton ?'+
                    'Value "none" implies no scaling; the only other supported option in "min_max" scaler ' +
                        '[default: {}]'.format(str(self._DEF_SCALER))},
            'split_test': {'abbr':'split', 'default':self._DEF_SPLIT_TEST, 'type':float,
                'help':'Fraction in (0,1] of data samples to split from training data' +
                    ' for testing; when the option value is 1,the dataset will be used ' +
                    ' both for training and testing [default: {}]'.format(str(self._DEF_SPLIT_TEST))}, 
            'train_random_n': {'abbr':'train_rand', 'default':self._DEF_TRAIN_RAND, 'type':int,
                'help':'Subset random n rows from training data to use for training ' + 
                    '[default: {}]'.format(str(self._DEF_TRAIN_RAND))}, 
            'train_first_n': {'abbr':'train_first', 'default':self._DEF_TRAIN_FIRST, 'type':int,
                'help':'Subset first n rows from training data to use for training ' + 
                    '[default: {}]'.format(str(self._DEF_TRAIN_FIRST))},
            'train_uniform_n': {'abbr':'train_unif', 'default':self._DEF_TRAIN_UNIF, 'type':int,
                'help':'Subset random n rows from training data with close to uniform ' + 
                    'distribution to use for training [default: {}]'.format(str(self._DEF_TRAIN_UNIF))}, 
            'sample_weights_coef': {'abbr':'sw_coef', 'default':self._DEF_SAMPLE_WEIGHTS, 'type':float,
                'help':'Coefficient in range ]-1, 1[ to compute sample weights for model training; ' +
                    'weights are defined as [sw_coef * (v - mid_range) + 1 for v in resp_vals] ' +
                    'where resp_vals is the response value vector or vector of mean values of all ' +
                    'responses per sample, and mid_range is the mid point of the range of resp_vals. ' +
                    'The value of sw_coef is chosen positive (resp. negative) when one wants to ' +
                    'to assign higher weights to samples with high (resp. low) values in resp_vals. ' +
                    'As an example, sw_coef = 0.2 assigns weight=1.2 to samples with max(resp_vals) ' +
                    'and weight=0.8 to samples with min(resp_vals), and sw_coef = 0 implies weight=1 ' +
                    'for each sample [default: {}]'.format(self._DEF_SAMPLE_WEIGHTS)}
            } | self._mrmrInst.mrmr_params_dict
        
    
    # set logger from a caller script
    def set_logger(self, logger):
        self._data_logger = logger 
        self._mrmrInst.set_logger(logger)
    
    # set paths from outside. It is instance of class (currently calleed DataFileInstance that contains
    # definitions of functions for computing file names of SMLP reports and files that store model info. 
    def set_paths(self, paths):
        self._pathInst = paths
        
    def _get_data_scaler(self, scaler_type):
        if scaler_type == 'none':
            return None
        elif scaler_type == 'min_max':
            return MinMaxScaler()
        else:
            raise Exception('Unsupported scaler type ' + str(scaler_type))
                            
    # Encode levels in categorical features of dataframe df as intgers.
    # When applied to labaled data (training, CV & test), argument levels_dict
    # is passed as None and filled-in in the function, to re-use for new data.
    # If in new data in some categorical feature we see a level (value) not
    # present in that feature in labeled data, an error is issued (for now),
    # end the justification is that the trained model does not know anything
    # about that level and therefore prediction might be arbitrarily imprecise.
    # TODO: here we do not include as categorical the columns that were declared 
    # as 'category': e,g., the following declares column 'A' as 'category':
    # X = DataFrame({'A':Series(range(3)).astype('category'), 'B':range(3), 'C':list('abc')}
    # we do add a sanity check to deal with dtype 'category' in the future.
    def _encode_categorical(self, df, levels_dict):
        for i in df.columns:
            assert df.dtypes[i]!='category'
        categ_features=[i for i in df.columns if df.dtypes[i]=='object']

        # correponds to application of the function on labaled data -- 
        # need to filling in levels_dict
        is_labeled_data = levels_dict == None

        if is_labeled_data:
            levels_dict = {}
            for cf in categ_features:
                lvls = df[cf].unique(); #print('lvls', lvls, sorted(lvls))
                levels_dict[cf] = list(lvls)

        # replace levels with integer values based on levels_dict
        unseen_levels_dict = {}
        for cf in categ_features:
            #print('cf', cf, '\n', df[cf].values, '\n', df[cf].sort_values())
            lvls = levels_dict[cf] # 
            if not is_labeled_data:
                lvls_new = df[cf].unique(); #print('lvls', lvls, sorted(lvls))
                unseen_levels = [l for l in lvls_new if not l in lvls]
                if len(unseen_levels) > 0:
                    unseen_levels_dict[cf] = unseen_levels
                    continue
            df[cf].replace(lvls, range(len(lvls)), inplace=True)

        if len(unseen_levels_dict.keys()) > 0:
            err_msg = 'Categorical features in new data have lavels not seen in labeled data\n'
            err_msg = err_msg + str(unseen_levels_dict)
            self._data_logger.error(err_msg) #self._data_
            raise Exception(err_msg) 
        #print('df\n', df)                   
        return df, levels_dict

    # Sanity-check the response names aginst the just loaded input labeled (training) or a new data.
    # For labeled training data, raises an exception if some of the responses specified 
    # by user are missing in training data;
    # For new data, checks that it contains eirther all the user specified responses or none.
    # raises an exception if the latter is not the case. Otherwise reports new data as labelled
    # or a unlabelled, respectively, using boolean return value new_labeled.
    def _sanity_check_responses(self, data, resp_names, is_training):
        if resp_names is None:
            self._data_logger.error('error: no response names were provided', file=sys.stderr)
            raise Exception('Response names must be provided')
        missing_responses = [rn for rn in resp_names if not rn in data.columns]
        if is_training:
            new_labeled = False
            if len(missing_responses) > 0:
                raise Exception('Responses ' + str(missing_responses) + ' do not occur in input data')
        else:
            if len(missing_responses) > 0 and len(missing_responses) != len(resp_names):
                raise Exception('The fillowing responses ' + str(missing_responses) + ' do not occur in input data')
            # does new data have response columns?
            new_labeled = len(missing_responses) == 0
        return new_labeled
    
    # split df into responses and features and return them as separate data frames
    def _get_response_features(self, df, feat_names, resp_names, is_training, new_labeled):
        assert not feat_names is None
        assert not df is None
        assert all(n in df.columns for n in feat_names)
        df_feat = df.drop([n for n in df if n not in feat_names], axis=1)
        if is_training or new_labeled:
            for resp_name in resp_names:
                assert resp_name in df.columns    
            return df_feat, df[resp_names]
        else:
            return df_feat, None

    
    # drop rows in training data for which at least one response is NA (is a missing value)
    def _drop_rows_with_na_in_responses(self, df, resp_names, data_name):
        rows_before = df.shape[0]
        df.dropna(subset=resp_names, inplace=True)
        rows_after = df.shape[0]
        if rows_before != rows_after:
            self._data_logger.info(str(rows_before-rows_after) + 
                'rows where at least one response is NA have been dropped from ' +str(data_name) + ' data')
        return df
    
    # drop features with single value (applies both to categorical and numeric features),
    # both to features and to responses, in training data only (not new data)
    def _drop_constant_features(self, df, data_name):
        constant_cols = []
        for col in df.columns.tolist():
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) == 0:
                constant_cols.append(col)
        df.drop(constant_cols, axis=1, inplace=True)
        if len(constant_cols) > 0:
            self._data_logger.info('The following constant features have been droped from ' + str(data_name) + ' data:')
            self._data_logger.info(str(constant_cols))
        return df, constant_cols
    
    # Save feature names to be used in model training per response as a dictionary
    # with response names as keys and the correonding model features as values.
    # Currently all reponses use the same features for training, this will change say
    # if model features per response will be selected based on MRMR algorithm or if
    # model features will be synthesised from original features per response.
    def _save_model_features(self, resp_names, feat_names, model_features_dict_file):
        model_features_dict = {}
        for resp in resp_names:
            model_features_dict[resp] = feat_names
        with open(model_features_dict_file, 'w') as f:
            json.dump(model_features_dict, f, indent='\t', cls=np_JSONEncoder)
    
    # Load feature names dictionary with reponse names as keys and the correponding features used 
    # in model training as the values. It was saved using _save_model_features during data processing.
    def _load_model_features(self, model_features_dict_file):
        with open(model_features_dict_file, 'r') as f:
            return json.load(f)
    
    def _save_model_levels(self, model_levels_dict, model_levels_dict_file):
        #pickle.dump(levels_dict, open(model_levels_dict_file, 'wb'))
        with open(model_levels_dict_file, 'w') as f:
            json.dump(model_levels_dict, f, indent='\t', cls=np_JSONEncoder)
    
    def _load_model_levels(self, model_levels_dict_file):
        #return pickle.load(open(model_levels_dict_file, 'rb'))
        with open(model_levels_dict_file, 'r') as f:
            return json.load(f)
    
    # saving the column min/max info into json file to be able to scale model prediction
    # results back to the original scale of the responses. The information in this file
    # is essetially the same as that avilable within mm_scaler but is easier to consume.
    def _save_data_bounds(self, scale, feat_names, resp_names, data_bounds_file, mm_scaler_feat, mm_scaler_resp):
        if scale:
            with open(data_bounds_file, 'w') as f:
                json.dump({
                    col: { 'min': mm_scaler_feat.data_min_[i], 'max': mm_scaler_feat.data_max_[i] }
                    for i,col in enumerate(feat_names) } |
                    {col: { 'min': mm_scaler_resp.data_min_[i], 'max': mm_scaler_resp.data_max_[i] }
                    for i,col in enumerate(resp_names)
                }, f, indent='\t', cls=np_JSONEncoder)
    
    def _save_data_scaler(self, scale:bool, mm_scaler_feat, mm_scaler_resp, features_scaler_file:str, 
        responses_scaler_file:str):
        #print('features_scaler_file to save into', features_scaler_file); 
        #print('responses_scaler_file to save into', responses_scaler_file)
        if scale:
            #print('mm_scaler_resp as saved', mm_scaler_resp.scale_, mm_scaler_resp.feature_names_in_); 
            #print('mm_scaler_feat as saved', mm_scaler_feat.scale_, mm_scaler_feat.feature_names_in_);
            pickle.dump(mm_scaler_feat, open(features_scaler_file, 'wb'))
            pickle.dump(mm_scaler_resp, open(responses_scaler_file, 'wb'))

    def _load_data_scaler(self, scale:bool, features_scaler_file, responses_scaler_file):
        #print('features_scaler_file to load from', features_scaler_file); 
        #print('responses_scaler_file to load from', responses_scaler_file)
        if scale:
            mm_scaler_feat = pickle.load(open(features_scaler_file, 'rb'))
            mm_scaler_resp = pickle.load(open(responses_scaler_file, 'rb'))
            #print('mm_scaler_resp as loaded', mm_scaler_resp.scale_, mm_scaler_resp.feature_names_in_); 
            #print('mm_scaler_feat as loaded', mm_scaler_feat.scale_, mm_scaler_feat.feature_names_in_);
        else:
            mm_scaler_feat = None
            mm_scaler_resp = None
        
        return mm_scaler_feat, mm_scaler_resp

    # Sample rows from dataframes X and y with the same number of rows in 
    # one of the three ways below (usually X defines features and y responses):
    # (a) select first n rows; 
    # (b) randomply select random_n rows;
    # (c) select uniform_n rows, with replacemnt, so that the
    #     mean values of the responses y in a row will be uniformly  
    #     distributed in the resulting dataset.
    def _sample_first_random_unifirm(self, X, y, first_n, random_n, uniform_n):
        self._data_logger.info('Sampling from training data: start')

        assert not (first_n >= 1 and random_n >= 1)
        assert not (first_n >= 1 and uniform_n >= 1)
        assert not (random_n >= 1 and uniform_n >= 1)
        if first_n >= 1:
            to_subset = nm.min(X.shape[0], first_n)
            X = X.iloc[:to_subset]
            y = y.iloc[:to_subset]
        elif random_n >= 1:
            # select with replacement only if we want more samples than available in X
            selct_with_replacement = random_n > X.shape[0]
            X = X.sample(random_n, replace=selct_with_replacement)
            y = y[y.index.isin(X.index)]
            y = y.reindex(X.index)
            #print(X.iloc[44832]) ; print(y.iloc[44832])
            # reset index in case of selection with replacement in order to ensure uniquness of indices
            if selct_with_replacement:
                X.reset_index(inplace=True, drop=True)
                y.reset_index(inplace=True, drop=True)
                #print(X.iloc[44832]) ; print(y.iloc[44832])        
        elif uniform_n >= 1:
            # select rows from X and y with repitition to acheive uniform destribution of 
            # values of y in the resumpled training data.
            y_seq = [] ; X_seq = []
            for y_i in y.columns:
                uniform_n_i = round(uniform_n / y.shape[1]); print('uniform_n_i', uniform_n_i)
                uniform_n_y_i = np.random.uniform(low=y[y_i].min(), high=y[y_i].max(), size=uniform_n_i) 
                filter_samples = [(y[y_i] - v).abs().idxmin() for v in uniform_n_y_i]
                #filter_samples = [(y.mean(axis=1) - v).abs().idxmin() for v in uniform_n_y_i]
                # takes nearly the same time: filter_samples = list(map(select_closest_row, np.array(uniform_n_y_i)))
                #print('y', y.shape)
                # .loc[] is required to sample exactly len(filter_samples) with replacement
                # cannot use .iloc[] because the indices are not continuous from 0 to k -- [0:k].
                # cannot use .isin() because it will not perform selection with replacement.
                print('y_i', y_i, '\n', y.loc[filter_samples])
                y_seq.append(y.loc[filter_samples]);  
                X_seq.append(X.loc[filter_samples]);  

            X = pd.concat(X_seq, axis=0)
            y = pd.concat(y_seq, axis=0)
            #print('y after uniform sampling', y.shape)
            # reset index in case of selection with replacement in order to ensure uniquness of indices
            X.reset_index(inplace=True, drop=True); #print('X', X.shape)
            y.reset_index(inplace=True, drop=True); #print('y', y.shape)

        self._data_logger.info('Sampling from training data: end')
        return X, y

    # load data, scale using sklearn MinMaxScaler(), then split into training and test 
    # subsets with ratio given by split_test. Optionally (based on arguments train_first_n, 
    # train_random_n, train_uniform_n), subsample/resample training data using function
    # sample_first_random_unifirm() explained above. Optionally (hard coded), subset also
    # the test data to see how the model performs on a subset of interest in test data 
    # (say samples where the responses have high values). 
    # Select from data only relevant features -- the responses that must be provided, and
    # required input features which are either specified using feat_names or are computed
    # from the data (features that are not reponses are used in analysis as input features).
    # Sanity cjeck on response names resp_names and feat_names are also performed.
    # Besides training and test subsets, the function returns also the MinMaxScaler object 
    # used for data normalization, to be reused for applying the model to unseen datasets
    # and also to rescale back the prediction results to the original scale where needed.
    def prepare_data_for_modeling(self, data_file, is_training, split_test, feat_names, resp_names,
            out_prefix, train_first_n:int, train_random_n:int, train_uniform_n:int, interactive_plots:bool, 
            mrmr_features_n:int, scaler_type:str, mm_scaler_feat=None, mm_scaler_resp=None, levels_dict=None, 
            model_features_dict=None):
        data_version_str = 'training' if is_training else 'new'
        self._data_logger.info('Preparing ' + data_version_str + ' data for modeling: start')
        
        # sanity check that the function is called correctly (as intended)
        if is_training:
            assert levels_dict==None
            assert mm_scaler_feat==None
            assert mm_scaler_resp==None

        self._data_logger.info('loading ' + data_version_str + ' data')
        data = pd.read_csv(data_file)
        self._data_logger.info('data summary\n' + str(data.describe()))
        #plot_data_columns(data)
        self._data_logger.info(data_version_str + ' data\n' + str(data))

        # sanity-check the response names aginst input data
        new_labeled = self._sanity_check_responses(data, resp_names, is_training)
        
        # if feature names are not provided, we assume all features in the data besides
        # the responses should be used in the analysis as input features.
        if feat_names is None and is_training:
            feat_names = [col for col in data.columns.tolist() if not col in resp_names]
        elif feat_names is None: # not is_training
            print('model_features_dict', model_features_dict)
            feat_names = lists_union_order_preserving_without_duplicates(list(model_features_dict.values()))
            
        if is_training:
            model_features_dict = {}
            for rn in resp_names:
                model_features_dict[rn] = feat_names
            print('model_features_dict used for feat_names', model_features_dict)

        # extract the required columns in data -- features and responses
        if is_training or new_labeled:
            data = data[feat_names + resp_names]
        else:
            data = data[feat_names]
        print('data 0\n', data)
        # in training data, drop all rows where at least one response has a missing value
        if is_training:
            data = self._drop_rows_with_na_in_responses(data, resp_names, 'training'); print('data 1\n', data)
            data, constant_feat = self._drop_constant_features(data, 'training'); print('data 2\n', data)
            resp_names = [rn for rn in resp_names if not rn in constant_feat]
            feat_names = [fn for fn in feat_names if not rn in constant_feat]
            for rn in model_features_dict.keys():
                if not rn in resp_names:
                    del model_features_dict[rn]
        
        #print('data\n', data)
        # impute missing values
        imp = SimpleImputer(strategy="most_frequent")
        data[ : ] = imp.fit_transform(data)
        self._data_logger.info(data_version_str + ' data after imputing missing values\n' + str(data))

        X, y = self._get_response_features(data, feat_names, resp_names, is_training, new_labeled)
        if not y is None:
            assert set(resp_names) == set(y.columns.tolist())
        
        # Feature selection / MRMR go here, will refine model_features_dict
        if is_training:
            #print('features before mrmr', feat_names)
            for rn in resp_names:
                mrmr_feat = self._mrmrInst.mrmr_regres(X, y[rn], rn, mrmr_features_n)
                model_features_dict[rn] = mrmr_feat
            feat_names = lists_union_order_preserving_without_duplicates(list(model_features_dict.values()))
            #print('features after mrmr', feat_names);
            X = X[feat_names]
            #print('model_features_dict after MRMR', model_features_dict);
        
        # encode levels of categorical features as integers for model training (in feature selection tasks 
        # it is best to use the original categorical features). 
        X, levels_dict = self._encode_categorical(X, levels_dict)
        self._data_logger.info(data_version_str + ' data after encoding levels of categorical features with integers\n' + str(data))
        
        #print('X\n', X); print('y\n', y); 
        # feature and response scaling (optional) will be done only on responses ane features
        # that occur in model_features_dict. For the optimization task, undoing scaling is done in prover-nn.py
        scale = not self._get_data_scaler(scaler_type) is None
        if is_training:
            if scale:
                mm_scaler_feat = self._get_data_scaler(scaler_type) #MinMaxScaler()
                mm_scaler_feat.fit(X)
                if not y is None:
                    mm_scaler_resp = self._get_data_scaler(scaler_type) #MinMaxScaler()
                    mm_scaler_resp.fit(y)
                print('mm_scaler_feat as computed', mm_scaler_feat.scale_, mm_scaler_feat.feature_names_in_);
                print('mm_scaler_resp as computed', mm_scaler_resp.scale_, mm_scaler_resp.feature_names_in_);
        if scale:
            X = pd.DataFrame(mm_scaler_feat.transform(X), columns=X.columns)
            if not y is None:
                y = pd.DataFrame(mm_scaler_resp.transform(y), columns=y.columns)
        else:
            assert mm_scaler_feat is None
            assert mm_scaler_resp is None

        self._data_logger.info(data_version_str + ' data after scaling (normalizing) features and responses\n' + str(data))
        #print(data.describe())
        #X, y = get_response_features(data, feat_names, resp_names)
        #print('normalized training data\n', X)
        #print("y.shape after normilization", y.shape)
        #print('scaled y\n', y)
        #plot_data_columns(DataFrame(y, columns=resp_names))
        if is_training or new_labeled:
            response_distribution_plot(out_prefix, y, resp_names, interactive_plots)

        # split data into training and test sets. TODO !!! do we rather use seed instead of 17?
        # another (maybe more robust) to make the split deterministic without using seed is using :
        # hashing https://engineeringfordatascience.com/posts/ml_repeatable_splitting_using_hashing/
        if is_training:
            if split_test == 1:
                X_train, X_test, y_train, y_test = (X, X, y, y)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y,
                    test_size=split_test, shuffle=True, random_state=17) 

            # make the training data small for purpose of quicker code development,
            # resample with repitition to make data larger if that might help, or
            # make its distribution close to uniform using sampling with replacement)
            X_train, y_train = self._sample_first_random_unifirm(X_train, y_train, 
                train_first_n, train_random_n, train_uniform_n)
            self._data_logger.info('X_train after sampling: ' + str(X_train.shape))
            self._data_logger.info('y_train after sampling: ' + str(y_train.shape))

            # for experirmentation: in case we want to see how model performs on a subrange 
            # of the the domain of y, say on samples where y > 0.9 (high values in y)
            if False and not y_test is None:
                #print(y_test.head()); print(X_test.head())
                filter_test_samples = y_test[resp_names[0]] > 0.9
                y_test = y_test[filter_test_samples]; 
                #print('y_test with y_test > 0.9', y_test.shape); print(y_test.head())
                X_test = X_test[filter_test_samples]; 
                #print('X_test with y_test > 0.9', X_test.shape); print(X_test.head())
            if scale:
                print('mm_scaler_feat as returned', mm_scaler_feat.scale_, mm_scaler_feat.feature_names_in_);
                print('mm_scaler_resp as returned', mm_scaler_resp.scale_, mm_scaler_resp.feature_names_in_);
            res = X, y, X_train, y_train, X_test, y_test, mm_scaler_feat, mm_scaler_resp, feat_names, \
                levels_dict, model_features_dict
        else:
            res = X, y
            
        self._data_logger.info('Preparing ' + data_version_str + ' data for modeling: end')
        return res
    
    # Process data to prepare components required for training models and prediction, and reporting results in
    # original scale. Supports also prediction and results reporting in origibal scale from saved model
    def process_data(self, inst, data_file, new_data_file, is_training, split_test, feat_names, resp_names,
            train_first_n:int, train_random_n:int, train_uniform_n:int, interactive_plots, scaler_type:str, 
            mrmr_features_n:int, save_model:bool, use_model:bool):
        
        scale = not self._get_data_scaler(scaler_type) is None
        
        if not data_file is None:
            split_test = self._DEF_SPLIT_TEST if split_test is None else split_test
            X, y, X_train, y_train, X_test, y_test, mm_scaler_feat, mm_scaler_resp, \
            feat_names, levels_dict, model_features_dict = self.prepare_data_for_modeling(
                data_file, True, split_test, feat_names, resp_names, inst.get_report_name_prefix(), 
                train_first_n, train_random_n, train_uniform_n, interactive_plots, 
                mrmr_features_n, scaler_type, None, None, None, None)

            assert scaler_type == 'none' or (not mm_scaler_feat is None)
            assert scaler_type == 'none' or (not mm_scaler_resp is None)
            self._save_model_levels(levels_dict, self._pathInst.model_levels_dict_file)
            self._save_model_features(resp_names, feat_names, self._pathInst.model_features_dict_file)
            self._save_data_scaler(scale, mm_scaler_feat, mm_scaler_resp, self._pathInst.features_scaler_file, 
                                   self._pathInst.responses_scaler_file)
            self._save_data_bounds(scale, feat_names, resp_names, self._pathInst.data_bounds_file, 
                                   mm_scaler_feat, mm_scaler_resp)
        else:
            assert use_model
            #print('features_scaler_file before use', self._pathInst.features_scaler_file); 
            #print('responses_scaler_file before use', self._pathInst.responses_scaler_file)
            mm_scaler_feat, mm_scaler_resp = self._load_data_scaler(scale, self._pathInst.features_scaler_file, 
                                                                    self._pathInst.responses_scaler_file)
            #mm_scaler_feat, mm_scaler_resp = self._load_data_scaler2(scale, responses)
            levels_dict = self._load_model_levels(self._pathInst.model_levels_dict_file)
            model_features_dict = self._load_model_features(self._pathInst.model_features_dict_file)
            feat_names = lists_union_order_preserving_without_duplicates(list(model_features_dict.values()))
            print('model_features_dict loaded', model_features_dict); print('feat_names loaded', feat_names)
            X, y, X_train, y_train, X_test, y_test =  None, None, None, None, None, None 
        
        if not new_data_file is None:
            X_new, y_new = self.prepare_data_for_modeling(
                new_data_file, False, None, feat_names, resp_names, inst.get_report_name_prefix(), 
                None, None, None, interactive_plots, mrmr_features_n, 
                scaler_type, mm_scaler_feat, mm_scaler_resp, levels_dict, model_features_dict)
        else:
            X_new, y_new = None, None

        # make sure data and new data have the same features to be used in the models
        if not data_file is None and not new_data_file is None:
            common_features = list_intersection(X.columns.tolist(), X_new.columns.tolist())
            X = X[common_features]
            X_train = X_train[common_features]
            X_test = X_test[common_features]
            X_new = X_new[common_features]
        print('model_features_dict returned', model_features_dict)
        return X, y, X_train, y_train, X_test, y_test, X_new, y_new, mm_scaler_feat, mm_scaler_resp, levels_dict, model_features_dict