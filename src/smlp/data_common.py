# data processing -- mainly to prepare data for model training
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from smlp.smlp_plot import response_distribution_plot



# Encode levels in categorical features of dataframe df as intgers.
# When applied to labaled data (training, CV & test), argument levels_dict
# is passed as None and filled-in in the function, to re-use for new data.
# If in new data in some categorical feature we see a level (value) not
# present in that feature in labeled data, an error is issued (for now),
# end the justification is that the trained model does not know anything
# about that level and therefore prediction might be arbitrarily imprecise.
def encode_categorical(df, levels_dict):
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
        print(err_msg)
        raise Exception(err_msg) 
    #print('df\n', df)                   
    return df, levels_dict

# split df into responses and features and return them as separate data frames
def get_response_features(df, feat_names, resp_names, is_training, new_labeled):
    assert all(n in df.columns for n in feat_names)
    df_feat = df.drop([n for n in df if n not in feat_names], axis=1)
    if is_training or new_labeled:
        for resp_name in resp_names:
            assert resp_name in df.columns    
        return df_feat, df[resp_names]
    else:
        return df_feat, None


# Sample rows from dataframes X and y with the same number of rows in 
# one of the three ways below (usually X defines features and y responses):
# (a) select first n rows; 
# (b) randomply select random_n rows;
# (c) select uniform_n rows, with replacemnt, so that the
#     mean values of the responses y in a row will be uniformly  
#     distributed in the resulting dataset.
def sample_first_random_unifirm(X, y, first_n, random_n, uniform_n):
    print('Sampling from training data: start')
    
    assert not (first_n >= 1 and first_n >= 1)
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
        uniform_n_01 = np.random.uniform(low=0.0, high=1.0, size=uniform_n) 
        #filter_samples = [(y[resp_names[0]] - v).abs().idxmin() for v in uniform_n_01]
        #print('y\n', y, '\n', y.mean(axis=1)); assert False
        filter_samples = [(y.mean(axis=1) - v).abs().idxmin() for v in uniform_n_01]
        # takes nearly the same time: filter_samples = list(map(select_closest_row, np.array(uniform_n_01)))
        #print('y', y.shape)
        # .loc[] is required to sample exactly len(filter_samples) with replacement
        # cannot use .iloc[] because the indices are not continuous from 0 to k -- [0:k].
        # cannot use .isin() because it will not perform selection with replacement.
        y = y.loc[filter_samples];  #print('y', y.shape)
        X = X.loc[filter_samples];  #print('X', X.shape)
        #print('y after uniform sampling', y.shape)
        # reset index in case of selection with replacement in order to ensure uniquness of indices
        X.reset_index(inplace=True, drop=True); #print('X', X.shape)
        y.reset_index(inplace=True, drop=True); #print('y', y.shape)
    
    print('Sampling from training data: end')
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
def prepare_data_for_modeling(data_file, is_training, split_test, feat_names, resp_names, 
    out_prefix, train_first_n, train_random_n, train_uniform_n, interactive_plots, scale,
    mm_scaler_feat=None, mm_scaler_resp=None, levels_dict=None):
    print('Preparing data for modeling: start')
    
    # sanity check that the function is called correctly (as intended)
    if is_training:
        assert levels_dict==None
        assert mm_scaler_feat==None
        assert mm_scaler_resp==None
    
    print('loading training data')
    data = pd.read_csv(data_file)
    print(data.describe())
    #plot_data_columns(data)
    print(data)
    
    # sanity-check the reponse names aginst input data
    if resp_names is None:
        print('error: no response names were provided', file=sys.stderr)
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
    
    # if feature names are not provided, we assume all features in the data besides
    # the responses should be used in the analysis as input features.
    if feat_names is None and is_training:
        feat_names = [col for col in data.columns.tolist() if not col in resp_names]
    elif feat_names is None: # not is_training
        raise Exception('Feature names must be provided for prediction')
    
    # extract the required columns in data -- features and responses
    if is_training or new_labeled:
        data = data[feat_names + resp_names]
    else:
        data = data[feat_names]
    
    # impute missing values
    imp = SimpleImputer(strategy="most_frequent")
    data[ : ] = imp.fit_transform(data)
    print('\ndata after imputing missing values\n', data)
    
    # encode levels of categorical features as integers
    # TODO !!!: need to only encode categorial features not responses as in case of
    # a classification task we want to leave the responses as categorical ????
    data, levels_dict = encode_categorical(data, levels_dict)
    print('\ndata after encoding levels of categorical features with integers\n', data)
    
    X, y = get_response_features(data, feat_names, resp_names, is_training, new_labeled)
    
    # normalize data
    # For optimization, undoing scaling is done in prover-nn.py
    if is_training:
        if scale:
            mm_scaler_feat = MinMaxScaler()
            mm_scaler_feat.fit(X)
            if not y is None:
                mm_scaler_resp = MinMaxScaler()
                mm_scaler_resp.fit(y)

    if scale:
        X = pd.DataFrame(mm_scaler_feat.transform(X), columns=X.columns)
        if not y is None:
            y = pd.DataFrame(mm_scaler_resp.transform(y), columns=y.columns)
    else:
        assert mm_scaler_feat is None
        assert mm_scaler_resp is None

    print('\ndata after scaling (normalizing) features and responses\n', data)
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
        X_train, X_test, y_train, y_test = train_test_split(X, y,
            test_size=split_test, shuffle=True, random_state=17) 

        # make the training data small for purpose of quicker code development,
        # resample with repitition to make data larger if that might help, or
        # make its distribution close to uniform using sampling with replacement)
        X_train, y_train = sample_first_random_unifirm(X_train, y_train, 
            train_first_n, train_random_n, train_uniform_n)
        print('X_train after sampling', X_train.shape); print('y_train after sampling', y_train.shape)
    
        # for experirmentation: in case we want to see how model performs on a subrange 
        # of the the domain of y, say on samples where y > 0.9 (high values in y)
        if False and not y_test is None:
            #print(y_test.head()); print(X_test.head())
            filter_test_samples = y_test[resp_names[0]] > 0.9
            y_test = y_test[filter_test_samples]; 
            #print('y_test with y_test > 0.9', y_test.shape); print(y_test.head())
            X_test = X_test[filter_test_samples]; 
            #print('X_test with y_test > 0.9', X_test.shape); print(X_test.head())
    
        print('Preparing data for modeling: end')
        return X, y, X_train, y_train, X_test, y_test, mm_scaler_feat, mm_scaler_resp, levels_dict
    else:
        return X, y
    