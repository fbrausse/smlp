# data processing -- mainly to prepare data for model training
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from logs_common import get_response_features
from smlp.smlp_plot import response_distribution_plot


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
def prepare_data_for_training(data_file, split_test, resp_names, feat_names, 
    out_prefix, train_first_n, train_random_n, train_uniform_n, interactive_plots):
    print('Preparing data for training: start')
    
    print('loading data')
    data = pd.read_csv(data_file)
    print(data.describe())
    #plot_data_columns(data)
    print(data)
    
    # sanity-check the reponse names aginst input data
    if resp_names is None:
        print('error: no response names were provided', file=sys.stderr)
        raise Exception('Response names must be provided')
    for rn in resp_names:
        if not rn in data.columns:
            raise Exception('Response ' + str(rn) + ' does not occur in input data')
    
    # if feature names are not provided, we assume all features in the data besides
    # the reponses should be used in the analysis as input features.
    if feat_names is None:
        feat_names = [col for col in data.columns.tolist() if not col in resp_names]
    
    # extract the required columns in data -- features and responses
    data = data[feat_names + resp_names]
    
    # normalize data
    # TODO !!!: the code that reverses data scaling might be missing for now????
    mm = MinMaxScaler()
    mm.fit(data)
    data = pd.DataFrame(mm.transform(data), columns=data.columns)

    print('normalized data')
    print(data.describe())

    X, y = get_response_features(data, feat_names, resp_names)
    print('normalized training data\n', X)

    print("y.shape after normilization", y.shape)
    #print('scaled y\n', y)
    #plot_data_columns(DataFrame(y, columns=resp_names))

    response_distribution_plot(out_prefix, y, resp_names, interactive_plots)
    
    # split data into training and test sets
    # TODO !!! do we rather use seed instead of 17?
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
    if True and not y_test is None:
        #print(y_test.head()); print(X_test.head())
        filter_test_samples = y_test[resp_names[0]] > 0.9
        y_test = y_test[filter_test_samples]; 
        #print('y_test with y_test > 0.9', y_test.shape); print(y_test.head())
        X_test = X_test[filter_test_samples]; 
        #print('X_test with y_test > 0.9', X_test.shape); print(X_test.head())
    
    print('Preparing data for training: end')
    return X, y, X_train, y_train, X_test, y_test, mm
    