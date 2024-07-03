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

from smlp_py.smlp_plots import response_distribution_plot
from smlp_py.smlp_utils import (np_JSONEncoder, list_intersection, str_to_bool, list_unique_unordered,
    lists_union_order_preserving_without_duplicates, get_response_type, cast_type, pd_df_col_is_numeric)
from smlp_py.smlp_mrmr import SmlpMrmr
#from smlp_py.smlp_spec import SmlpSpec
from smlp_py.smlp_constants import *
from smlp_py.smlp_discretize import SmlpDiscretize
#from smlp_py.smlp_correlations import SmlpCorrelations


# Methods for data processing, traing vs test splitting, handling responses vs features, and more.
class SmlpData:
    def __init__(self):
        #data_logger = logging.getLogger(__name__)
        self._data_logger = None # create_logger('data_logger', log_file, log_level, log_mode, log_time)
        # TODO !!!: currently we instantiate MrmrFeatures here within DataCommon calss because MRMR is only
        # used as part of data processing before modeling. In the furture MRMR will be used as part of feature
        # selection say in Rage Analysis (and model building might or might not be required in the overall flow)
        # and in that case we might need to instantiate MrmrFeatures at a top level script (like smslp_train.py)
        # along with instantiating DataCommon (and maybe ModelsCommon).
        self._mrmrInst = SmlpMrmr() 
        self._specInst = None # SmlpSpec()
        
        # default values of parameters related to dataset; used to generate args.
        self._DEF_TRAIN_FIRST = 0 # subseting first_n rows from training data
        self._DEF_TRAIN_RAND = 0  # sampling random_n rows from training data
        self._DEF_TRAIN_UNIF = 0  # sampling from training data to acheive uniform distribution
        self._DEF_SCALER = 'min_max'  # options: 'min_max', 'max-abs'
        self._DEF_SPLIT_TEST = 0.2 # ratio to split training data into training and validation subsets
        self._DEF_SAMPLE_WEIGHTS_INTERCEPT = 0 # intercept a of the power function a+bx**c that computes the
                                     # sample weights based on response values on these samples
        self._DEF_SAMPLE_WEIGHTS_COEFFICIENT = 0 # coefficient b of the power function a+bx**c that computes the
                                     # sample weights based on response values on these samples
        self._DEF_SAMPLE_WEIGHTS_EXPONENT = 0 # degree/exponent c of the power function a+bx**c that computes the
                                     # sample weights based on response values on these samples
        self._RESPONSE_TO_BOOL = None # list of conditions per response to convert numeric responses 
                                      # into binary 1/0 responses
        self._CONDITION_SEPARATOR = ';' # separator used in self._RESPONSE_TO_BOOL to seperate conditions 
        self._DEF_SCALE_FEATURES = True
        self._DEF_SCALE_RESPONSES = True
        self._DEF_IMPUTE_RESPONSES = False
        self._DEF_RESPONSE_PLOTS = True # should response values distribution plots be genrated?
        
        # SMLP default values of positive and negative samples in banary responses 
        self.SMLP_NEGATIVE_VALUE = int(0)
        self.SMLP_POSITIVE_VALUE = int(1)
        
        # features that must be used in training models (assuming they are within initially defined input features)
        self._DEF_KEEP_FEATURES = []
        
        # Dictionary with features (names) that have a missing values as dictionary keys and row indices of the 
        # missing values as dictionary values. It is computed just before imputing missing values in features. 
        # IMPORTANT: ensure rows are not dropped beyond this point or if they are, row indices are not reset so 
        # that missing value locations in features are identified using this dictionary at later stages of the flow.
        self._missing_vals_dict = None
        
        # will store respectively features and reponses scalres after fitting to features and 
        # responses in the training data
        self._mm_scaler_feat = None
        self._mm_scaler_resp = None
        
        # will store labled data before scaling for training, required for model exploration modes
        # like otimization, verification, querying model
        self._X_orig_scale = None
        self._y_orig_scale = None
        
        # Default values of dataset processing related parameters; used to generate args parser.
        # Here are some principles for data processing which give a more precise meaning to the 
        # options introduced below:
        # Consant columns and responses are dropped always in the training data, so all responses 
        # which will be considered for anlysis have at least two values (not so for test/new data).
        #
        # data preprocessing
        # Data preprocessing is applied to raw data immediately after loading, and its aim is to prepare
        # data to confirm to SMLP tool requirements many of which are described below. That is, this
        # stage of data processing is to make the tool user friendly, and perform some data 
        # transformations instead of the user having to do this. Thus, all the reports and visualization
        # of the results will use preprocessed data, and assume the data was passed to the tool in that
        # form. As an example, if some values in columns were replaced, say 'pass' wwas replaced by 0
        # and 'fail' was replaced by 1, the reports will use values 0 and 1 in that column. Some of the 
        # data preprocessing steps might include dropping some of the columns and/or rows in the loaded
        # data, replacing values in some of the columns with other values, casting types of columns, etc.
        #
        # Categorical responses / classification analyses:
        # For now, categorical responses are supported only if they have two values -- it is user 
        # responsibility to encode a categorical response with more than two levels (values) into
        # a number of binary responses (say through the one-hot encoding). 
        # A categorical response can be specified as a (a) 0/1 feature, (b) categorical feature with 
        # two levels; or (c) numeric feature with two values. In all cases, parameters specified through
        # options positive_value and negative_value determine which one of these two values in that
        # response define the positive samples and which ones define the negative ones -- both in 
        # training data and in test data if the latter is labeled -- has that response column. 
        # Then, as part of data preprocessing (which is only the first stage of data preparation),
        # the positive_value and the negative_value in the response will be replaced by 
        # STAT_POSITIVE_VALUE and STAT_POSITIVE_VALUE, respectively, which are equal to 1 and 0, 
        # following the convention in statistics that value integer 1 denotes positive and 0 negative. 
        # That is, the intention is that user has to provide categorical responses as 1/0 responses
        # where 1 denotes positive samples and 0 denotes negative samples, and to make the tool user
        # friendly the user has freedom to specify categorical responses in one of the three ways (a)-(c)
        # described above with the intention that the tool internally will convert such responses into
        # 1/0 columns and all the results will be reported and visualized using 1/0 values and not the
        # original values used to define positive and negative values in the responses in the raw input data.
        #
        # Numeric responses / regression analyses:
        # Float and int columns in input data can define numeric responses. Each such response with more than 
        # two values is treated as numeric (and we are dealing with a regression analyses). If a response has 
        # two values, than it can still be treated as a categorical/binary response, as described in case (c)
        # of specifying binary responses. Otherwise -- that is, when {positive_value, negative_value} is not
        # equal to the set of the two values in the response, the response is treated as numeric. 
        # Parameter values specified through options positive_value and negative_value have a different 
        # meaning for numeric responses: they are not used to replace values in the response as part of 
        # preprocessing. Instead, positive_value = STAT_POSITIVE_VALUE and negative_value = STAT_NEGATIVE_VALUE
        # (which is the default) specifies that the high values in the response are positive (undesirable) and
        # the low values are negative (desirable). The opposite assignment positive_value = STAT_NEGATIVE_VALUE 
        # and negative_value = STAT_POSITIVE_VALUE specifies that low values in the response are positive and
        # high values are negative. Other possibilities for the pair (positive_value, negative_value) are 
        # considered as incorrect specification and an error message is issued. 
        #
        # Multiple responses
        # Multiple responses can be treated in a single analysis only if all of them are identified as 
        # defining regression analysis or all of them are identifies as defining classification analysis.
        # if that is not the case, SMLP will abort with an error message clarifying the reason.
        #
        # Optimization problems:
        # If we are dealing with an optimization problem for a response or multiple responses, then combination
        # positive_value = STAT_POSITIVE_VALUE and negative_value = STAT_NEGATIVE_VALUE specifies that 
        # we want to maximize the response values (find regions in input space where the responses are
        # close to maximum / close to pareto optimal with respect to maximization; and conversely, combination
        # positive_value = STAT_NEGATIVE_VALUE and negative_value = STAT_POSITIVE_VALUE specifies that we
        # are looking at (pareto) optimization problem with respect to minimization.
        #
        # Data scaling / normalization
        # Data scaling is managed separately for the features and the responses. A particular mode of usage
        # (model training and prediction, feature selection or range analysis, pareto optimization, etc.)
        # can decide to scale features and or scale responses. The reports and visualization should use
        # features and responses in the original scale, thus unscaling must be performed. Normalization is 
        # done after data preprocessing, and is part of data processing (which is a second stage of data 
        # preparation, to which preprocessing is the first stage). SMLP optimization algorithms operate
        # with data in original scale (say domain spec describing input feature and response types and
        # ranges refer to preprocessed data, in the original scale, even if say models were trained with
        # scaled features and/or responses).
        #
        # Processing of features
        # Constant features (and responses) are dropped as part of preprocessing. Currently SMLP does
        # not have a need to make a direct usage of boolean type in features (or in responses) 
        # and they (the boolean features) are treated as categorical (by converting boolean values to
        # strings 'True' and 'False'). Therefore after preprocessing the only supported pandas column types
        # are 'int', 'float', or categorical, where, in turn, categorical features can have types 'object' 
        # (=string type of column values) or 'category'; the category type can be ordered or un-ordered.
        # Some of the algorithms prefer to use categorical features as is -- with string values: for example,
        # feature selection algorithms can use dedicated correlation measures for categorical features. Also,
        # some of the training algorithms, such as tree based, can deal with categorical features directly, 
        # while others, e.g., Neural Networks, polynomial models, and more, assume all inputs are numeric
        # integer or float). Therefore, depending on the analysis mode (feature selection, model training,
        # model exploration), categorical features might be encoded into integers (and treated as discrete 
        # domains -- and not continuous domains). Conversely, some algorithms (especially, correlations)
        # might prefer to discretize numeric features into categorical features, and discretization options
        # in SMLP support discretization of numeric features with target type 'object' and 'category', 
        # ordered or un-ordered, where the values in the resulting columns can be integers (as strings or
        # levels, e.g., 5) of other values (like 'bin5').
        self.data_params_dict = {
            'response': {'abbr':'resp', 'default':None, 'type':str,
                'help':'Names of response variables, must be provided [default None]'}, 
            'features': {'abbr':'feat', 'default':None, 'type':str,
                'help':'Names of input features (can be computed from data) [default None]'},
            'keep_features': {'abbr':'keep_feat', 'default':self._DEF_KEEP_FEATURES, 'type':str,
                'help':'Names of input features that should be used in model training: feature selection ' +
                    'or other heuristics for selecting features that will be used in model training ' +
                    'cannot drop these input features [default {}]'.format(str(self._DEF_KEEP_FEATURES))},
            'new_data': {'abbr':'new_data', 'default':None, 'type':str,
                'help':'Path excluding the .csv suffix to new data file [default: None]'},
            'data_scaler': {'abbr':'data_scaler', 'default':self._DEF_SCALER, 'type':str, 
                'help':'Should features and responses be scaled and with which scaling optionton? '+
                    'Value "none" implies no scaling; the only other supported option in "min_max" scaler ' +
                    '[default: {}]'.format(str(self._DEF_SCALER))},
            'scale_features': {'abbr':'scale_feat', 'default': self._DEF_SCALE_FEATURES, 'type':str_to_bool,
                'help': 'Should features be scaled using scaler specified through option "data_scaler"? ' +
                    '[default: ' + str(self._DEF_SCALE_FEATURES) + ']'},
            'scale_responses': {'abbr':'scale_resp', 'default': self._DEF_SCALE_RESPONSES, 'type':str_to_bool,
                'help': 'Should responses be scaled using scaler specified through option "data_scaler"? ' +
                    '[default: ' + str(self._DEF_SCALE_RESPONSES) + ']'},
            'impute_responses': {'abbr':'impute_resp', 'default': self._DEF_IMPUTE_RESPONSES, 'type':str_to_bool,
                'help': 'Should missing values in responses be imputed? Might make sense when there are ' +
                    'multiple responses and different responses have missing values in different samples: ' +
                    'this might be a better alternative compared to dropping rows where at least one response '
                    'has a missing value [default: ' + str(self._DEF_IMPUTE_RESPONSES) + ']'},
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
            'sample_weights_coef': {'abbr':'sw_coef', 'default':self._DEF_SAMPLE_WEIGHTS_COEFFICIENT, 'type':float,
                'help':'Coefficient in range ]-1, 1[ to compute sample weights for model training; ' +
                    'weights are defined as [sw_coef * (v - mid_range) + 1 for v in resp_vals] ' +
                    'where resp_vals is the response value vector or vector of mean values of all ' +
                    'responses per sample, and mid_range is the mid point of the range of resp_vals. ' +
                    'The value of sw_coef is chosen positive (resp. negative) when one wants to ' +
                    'assign higher weights to samples with high (resp. low) values in resp_vals. ' +
                    'As an example, sw_coef = 0.2 assigns weight=1.2 to samples with max(resp_vals) ' +
                    'and weight=0.8 to samples with min(resp_vals), and sw_coef = 0 implies weight=1 ' +
                    'for each sample [default: {}]'.format(self._DEF_SAMPLE_WEIGHTS_COEFFICIENT)},
            'sample_weights_exponent': {'abbr':'sw_exp', 'default':self._DEF_SAMPLE_WEIGHTS_EXPONENT, 'type':float,
                'help':'The Exponent to compute sample weights for model training; ' +
                    'weights are defined as  [sw_int + sw_coef *((v - mn)/(mx-mn))**sw_exp for v in resp_vals ] ' +
                    'where resp_vals is the response value vector or vector of mean values of all ' +
                    'responses per sample, and mn and mx are respectively the min and max of resp_vals. ' +
                    'The value of sw_coef is chosen non-negative to make sure all weights are non-negative ' +
                    '[default: {}]'.format(self._DEF_SAMPLE_WEIGHTS_EXPONENT)},
            'sample_weights_intercept': {'abbr':'sw_int', 'default':self._DEF_SAMPLE_WEIGHTS_INTERCEPT, 'type':float,
                'help':'The intercept to compute sample weights for model training; ' +
                    'weights are defined as [sw_int + sw_coef *((v - mn)/(mx-mn))**sw_exp for v in resp_vals ] ' +
                    'where resp_vals is the response value vector or vector of mean values of all ' +
                    'responses per sample, and mn and mx are respectively the min and max of resp_vals. ' +
                    'The value of sw_coef is chosen non-negative to make sure all weights are non-negative ' +
                    '[default: {}]'.format(self._DEF_SAMPLE_WEIGHTS_INTERCEPT)},
            'response_to_bool':{'abbr':'resp2b', 'default':self._RESPONSE_TO_BOOL, 'type':str,
                'help':'Semicolon seperated list of conditions to be applied to the responses in the '
                    'order the responses are specified, to convert them into binary responses ' +
                    'as part of data preprocessing. The conditions define when each response is positive. ' +
                    'Say a condition resp1 > 5 transforms response called resp1 into a binary 1/0 response ' + 
                    'that has value 1 for each data sample (row) where resp1 is greater than 5 and value 0 '
                    'for the remaining samples [default: {}]'.format(str(self._RESPONSE_TO_BOOL))},
            'positive_value': {'abbr':'pos_val', 'default':self.SMLP_POSITIVE_VALUE, 'type':str,
                'help':'Value that represents positive values in a binary categorical response ' +
                    'in the original input data (before any data processing has been applied) ' +
                    '[default: {}]'.format(str(self.SMLP_POSITIVE_VALUE))},
            'negative_value': {'abbr':'neg_val', 'default':self.SMLP_NEGATIVE_VALUE, 'type':str,
                'help':'Value that represents negative values in a binary categorical response ' +
                    'in the original input data (before any data processing has been applied) ' +
                    '[default: {}]'.format(str(self.SMLP_NEGATIVE_VALUE))},
            'response_plots': {'abbr':'resp_plots', 'default': self._DEF_RESPONSE_PLOTS, 'type':str_to_bool,
                'help': 'Should response value distribution plots be genrated during data processing? ' +
                    'A related option interactive_plots controls whether the generated plots should be ' +
                    'displayed interactively during runtime [default: ' + str(self._DEF_RESPONSE_PLOTS) + ']'}
            } | self._mrmrInst.mrmr_params_dict
        self.data_bounds_dict = None
        
    
    # set logger from a caller script
    def set_logger(self, logger):
        self._data_logger = logger 
        self._mrmrInst.set_logger(logger)
    
    # report_file_prefix is a string used as prefix in all report files of SMLP
    def set_report_file_prefix(self, report_file_prefix):
        self.report_file_prefix = report_file_prefix

    # model_file_prefix is a string used as prefix in all outut files of SMLP that are used to 
    # save a trained ML model and to re-run the model on new data (without need for re-training)
    def set_model_file_prefix(self, model_file_prefix):
        self.model_file_prefix = model_file_prefix
    
    def set_spec_inst(self, spec_inst):
        self._specInst = spec_inst
    
    @property
    def unscaled_training_features(self):
        return self._X_orig_scale
    
    @property
    def unscaled_training_responses(self):
        return self._y_orig_scale
    
    @property
    def commandline_condition_separator(self):
        return self._CONDITION_SEPARATOR
    
    # min/max info of all columns (features and responses) in input data
    # (labeled data used for training and testing ML models)
    @property
    def data_bounds_file(self):
        assert self.model_file_prefix is not None
        return self.model_file_prefix + '_data_bounds.json'

    # file to save/load missing values information of features in input data
    @property
    def missing_values_fname(self):
        assert self.report_file_prefix is not None
        return self.report_file_prefix + '_missing_values_dict.json'
    
    # saved min-max scaler filename for features
    @property
    def features_scaler_file(self):
        assert self.model_file_prefix is not None
        return self.model_file_prefix + '_features_scaler.pkl'
    
    # saved min-max scaler filename for responses
    @property
    def responses_scaler_file(self):
        assert self.model_file_prefix is not None
        return self.model_file_prefix + '_responses_scaler.pkl'
    
    # dictionary containing categorica feature names as keys and correponding levels as values.
    # Includes only categorical features that will be used in traing a model (a subset of original.
    # fetaures and features engineered from these features). This info is part of the saved model.
    @property
    def model_levels_dict_file(self):
        assert self.model_file_prefix is not None
        return self.model_file_prefix + '_model_levels_dict.json'
    
    @property
    def model_features_dict_file(self):
        assert self.model_file_prefix is not None
        return self.model_file_prefix + '_model_features_dict.json'
    
    def _get_data_scaler(self, scaler_type):
        if scaler_type == 'none':
            return None
        elif scaler_type == 'min_max':
            return MinMaxScaler()
        else:
            raise Exception('Unsupported scaler type ' + str(scaler_type))
                            
    # Encode levels in categorical features of dataframe df as integers.
    # When applied to labaled data (training, CV & test), argument levels_dict
    # is passed as None and is filled-in in this function, to re-use for new data.
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
    # For labeled training data, raises an exception if some of the responses specified by user are 
    # missing in training data. For new data, checks that it contains either all the user specified 
    # responses or none. Raises an exception if the latter is not the case. Otherwise reports new 
    # data as labelled or a unlabelled, respectively, using boolean return value new_labeled.
    def _sanity_check_responses(self, df, resp_names, is_training):
        if resp_names is None:
            self._data_logger.error('error: no response names were provided', file=sys.stderr)
            raise Exception('Response names must be provided')
        missing_responses = [rn for rn in resp_names if not rn in df.columns]
        if is_training:
            new_labeled = False
            if len(missing_responses) > 0:
                raise Exception('Responses ' + str(missing_responses) + ' do not occur in input data')
        else:
            if len(missing_responses) > 0 and len(missing_responses) != len(resp_names):
                raise Exception('The fillowing responses ' + str(missing_responses) + ' do not occur in input data')
            # does new data have response columns?
            new_labeled = len(missing_responses) == 0
        '''    TODO !!! move this check after _preprocess_responses()
        # check that all responses have the same type (numeric vs binary); otherise raise an exception.
        if is_training or new_labeled:
            resp_type = None
            for rn in resp_names:
                rn_type = get_response_type(df, rn)
                if resp_type is None:
                    resp_type = rn_type
                elif resp_type != rn_type:
                    raise Exception('All responses must require same classification vs regeression mode')
        '''
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
                ' rows where at least one response is NA have been dropped from ' + str(data_name) + ' data')
        return df
    
    # drop features with single value (applies both to categorical and numeric features),
    # both to features and to responses, in training data only (not new data)
    def _drop_constant_features(self, df:pd.DataFrame, keep_feat:list[str], data_name:str):
        constant_cols = []
        #print('df\n', df)
        for col in df.columns.tolist():
            #print('col', col, 'df[col]\n', df[col]); print('df[col].dropna()\n', df[col].dropna())
            unique_vals = df[col].dropna().unique(); #print('col', col, 'unique', unique_vals)
            # cover here also case when unique_vals is empty -- when all values are missing in a feature
            if len(unique_vals) <= 1: 
                constant_cols.append(col)
        #print('constant_cols', constant_cols, 'keep_feat', keep_feat)
        constant_cols_to_drop = [c for c in constant_cols if c not in keep_feat]
        #print('constant_cols_to_drop', constant_cols_to_drop); assert False
        df.drop(constant_cols_to_drop, axis=1, inplace=True)
        if len(constant_cols_to_drop) > 0:
            self._data_logger.info('The following constant features have been droped from ' + str(data_name) + ' data:')
            self._data_logger.info(str(constant_cols_to_drop))
        return df, constant_cols_to_drop
    
    # We do not make a direct usage of boolean type in data, convert such columns into object/string type
    def _cast_boolean_features(self, df:pd.DataFrame):
        col_types = df.dtypes
        for i, resp_name in enumerate(df.columns):
            if col_types[i] == bool:
                df[resp_name].map({True:'True', False:'False'})
        return df
    
    # This function should be called after rows with missing values in at least one of the responses  
    # have been dropped. Constant responses (with exactly one non-NaN value) are dropped here. 
    def _preprocess_responses(self, resp_df:pd.DataFrame, pos_value:int, neg_value:int, resp_to_bool:str, is_training:bool):
        resp_names = resp_df.columns.tolist()
        resp_types = resp_df.dtypes
        for i, resp_name in enumerate(resp_names):
            resp_type = resp_types[i]
            assert resp_type != bool # columns of boolean type were already converted to object type
            assert resp_type in [int, float, object] # the only expected / suported column types
            
            # values in the response of type object are strings -- hence the following line of code:
            resp_val_type = resp_type if resp_type in [int, float] else str
            pos_val = cast_type(pos_value, resp_val_type); #print('pos_value', pos_value, type(pos_value))
            neg_val = cast_type(neg_value, resp_val_type); #print('neg_value', neg_value, type(neg_value))
            resp_unique_vals = resp_df[resp_name].unique().tolist()
            
            if len(resp_unique_vals) < 2 and is_training:
                raise Exception('Response ' + str(resp_name) + ' has only one value')
                
            if set(resp_unique_vals) == {STAT_NEGATIVE_VALUE, STAT_POSITIVE_VALUE}:
                if pos_val == 0 and neg_val == 1:
                    # swap 0 and 1 in the response, to make sure that after processing positive value will be 1
                    resp_df[resp_name] = resp_df.replace({resp_name: {pos_val: STAT_POSITIVE_VALUE, 
                        neg_val: STAT_NEGATIVE_VALUE}})
                elif not (pos_val == STAT_POSITIVE_VALUE and neg_val == STAT_NEGATIVE_VALUE):
                    raise Exception('Options positive_value and negative value might not be specified correctly')
            elif len(resp_unique_vals) == 2:
                # this response can be treated a binary or numeric depending on pos_value and neg_values, as follows:
                if {neg_val, pos_val} == set(resp_unique_vals):
                    # adapt the response to treat is a categorical / binary
                    resp_df[resp_name].replace({pos_val: STAT_POSITIVE_VALUE, neg_val: STAT_NEGATIVE_VALUE}, inplace=True)
                elif not resp_type in [int, float]:
                    raise Exception('Response ' + str(resp_name) + ' has two string (categorical) values ' +
                        str(resp_unique_vals) + ' not matching set ({neg}, {pos})'.format(neg=STAT_NEGATIVE_VALUE, 
                            pos=STAT_POSITIVE_VALUE) + '; Please use options --positive_value and ' + 
                        '--negative_value to specify positive and nagative values in the responses')
            elif not resp_type in [int, float]:
                if resp_type == object:
                    if len(resp_unique_vals) > 2:
                        raise Exception('Response ' + str(resp_name) + 'has more then two string (categorical) ' +
                            'values; please encode it into a number of binary responses')
                    else:
                        raise Exception('Response ' + str(resp_name) + 'has unsupported type ' + str(resp_type))
        
        # Apply resp_to_bool condition to numeric responses, so that if high values are positive (which is 
        # the default) then for samples that satisfy resp_to_bool the generated response will have value
        # STAT_POSITIVE_VALUE (and the rest of the samples will have response value STAT_NEGATIVE_VALUE).
        # Since values of parameters pos_value and neg_value are also used to determine whether, in the
        # optimization mode, the aim is to mazimize or to minimize, we want to make sure that pos_value and
        # neg_value have their default values STAT_POSITIVE_VALUE and STAT_NEGATIVE_VALUE, respectively.
        # Hence when resp_to_bool is used for numeric responses, there is no need to specufy / change default 
        # values of pos_value and neg_value.
        if resp_type in [int, float] and resp_to_bool is not None:
            assert pos_value == STAT_POSITIVE_VALUE and neg_value == STAT_NEGATIVE_VALUE
            #print('resp_to_bool', resp_to_bool, type(resp_to_bool)); print(resp_to_bool is None)
            df_resp_cond = resp_to_bool; #print('resp_cond', resp_to_bool)
            for resp_name in resp_names:
                df_resp_cond = df_resp_cond.replace('{}'.format(resp_name), 'resp_df[resp_name]')
            #print('df_resp_cond', df_resp_cond)

            resp_conds = df_resp_cond.split(self._CONDITION_SEPARATOR); #print('resp_conds', resp_conds)
            for resp_name, resp_cond in zip(resp_names, resp_conds):
                #print(resp_name, resp_cond)
                # because eval() below is called with global environemnt {} and local environment  
                # {'resp_name':resp_name, 'resp_df':resp_df}. we limit the variables and functions 
                # that the evaluated code can access and this way make call to eval() safer.
                resp_new = eval(resp_cond, {}, {'resp_name':resp_name, 'resp_df':resp_df})
                resp_df[resp_name] = [int(e) for e in resp_new]
        #print('processed resp_df\n', resp_df)
        return resp_df
        
    # Save feature names to be used in model training per response as a dictionary
    # with response names as keys and the correonding model features as values.
    # Currently all responses use the same features for training, this will change say
    # if model features per response will be selected based on MRMR algorithm or if
    # model features will be synthesised from original features per response.
    def _save_model_features(self, resp_names, feat_names, model_features_dict_file):
        model_features_dict = {}
        for resp in resp_names:
            model_features_dict[resp] = feat_names
        with open(model_features_dict_file, 'w') as f:
            json.dump(model_features_dict, f, indent='\t', cls=np_JSONEncoder)
    
    # Load feature names dictionary with response names as keys and the correponding features used 
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
    
    # this function is intended to be applied on training data (features and responses) where
    # all features are numeric; thie latter is required when training a model in SMLP model 
    # exploration modes like optimization, etc.
    def _data_bounds(self, df):
        assert all([pd_df_col_is_numeric(df, col) for col in df.columns])
        return { col:{'min' : df[col].min(), 'max' : df[col].max() } for col in df.columns }
    
    # saving the column min/max info into json file to be able to scale model prediction
    # results back to the original scale of the responses. The information in this file
    # is essetially the same as that avilable within mm_scaler but is easier to consume.
    def _save_data_bounds(self, df, feat_names, resp_names, data_bounds_file, mm_scaler_feat, mm_scaler_resp):
        if mm_scaler_feat is None and mm_scaler_resp is None:
            data_bounds_dict = self._data_bounds(df)    
        else:
            feat_bounds_dict = { col: { 'min': mm_scaler_feat.data_min_[i], 'max': mm_scaler_feat.data_max_[i] }
                for i,col in enumerate(feat_names) } if mm_scaler_feat is not None else {}
            resp_bounds_dict = {col: { 'min': mm_scaler_resp.data_min_[i], 'max': mm_scaler_resp.data_max_[i] }
                for i,col in enumerate(resp_names) } if mm_scaler_resp is not None else {}
            data_bounds_dict = feat_bounds_dict | resp_bounds_dict

        if data_bounds_dict != {}:
            self._data_logger.info('Saving data bounds into file:' + str(data_bounds_file))
            self._data_logger.info(str(data_bounds_dict))
            self.data_bounds_dict = data_bounds_dict
            with open(data_bounds_file, 'w') as f:
                json.dump(data_bounds_dict, f, indent='\t', cls=np_JSONEncoder)
        
    def _save_data_scaler(self, scale_feat:bool, scale_resp:bool, mm_scaler_feat, mm_scaler_resp, 
            features_scaler_file:str, responses_scaler_file:str):
        #print('features_scaler_file to save into', features_scaler_file); 
        #print('responses_scaler_file to save into', responses_scaler_file)
        if scale_feat:
            #print('mm_scaler_feat as saved', mm_scaler_feat.scale_, mm_scaler_feat.feature_names_in_);
            pickle.dump(mm_scaler_feat, open(features_scaler_file, 'wb'))
        if scale_resp:
            #print('mm_scaler_resp as saved', mm_scaler_resp.scale_, mm_scaler_resp.feature_names_in_); 
            pickle.dump(mm_scaler_resp, open(responses_scaler_file, 'wb'))

    def _load_data_scaler(self, scale_features:bool, scale_responses:bool, features_scaler_file, 
            responses_scaler_file):
        #print('features_scaler_file to load from', features_scaler_file); 
        #print('responses_scaler_file to load from', responses_scaler_file)
        if scale_features:
            mm_scaler_feat = pickle.load(open(features_scaler_file, 'rb'))
            #print('mm_scaler_resp as loaded', mm_scaler_resp.scale_, mm_scaler_resp.feature_names_in_); 
        else:
            mm_scaler_feat = None
        
        if scale_responses:
            mm_scaler_resp = pickle.load(open(responses_scaler_file, 'rb'))
            #print('mm_scaler_resp as loaded', mm_scaler_resp.scale_, mm_scaler_resp.feature_names_in_); 
        else:
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
                uniform_n_i = round(uniform_n / y.shape[1]); #print('uniform_n_i', uniform_n_i)
                uniform_n_y_i = np.random.uniform(low=y[y_i].min(), high=y[y_i].max(), size=uniform_n_i) 
                filter_samples = [(y[y_i] - v).abs().idxmin() for v in uniform_n_y_i]
                #filter_samples = [(y.mean(axis=1) - v).abs().idxmin() for v in uniform_n_y_i]
                # takes nearly the same time: filter_samples = list(map(select_closest_row, np.array(uniform_n_y_i)))
                #print('y', y.shape)
                # .loc[] is required to sample exactly len(filter_samples) with replacement
                # cannot use .iloc[] because the indices are not continuous from 0 to k -- [0:k].
                # cannot use .isin() because it will not perform selection with replacement.
                #print('y_i', y_i, '\n', y.loc[filter_samples])
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

    # Compute locations (indices) of missing values in each column and create a 
    # dictionary with features that have at least one missing value as keys and 
    # the list of the respective missing value indices as the values. 
    # Write out this dictionary as a json file.
    def _compute_missing_values_dict(self, df):
        missing_vals_rows_array,  missing_vals_cols_array = np.where(df.isna()); 
        #print(missing_vals_rows_array); print(missing_vals_cols_array)
        if len(missing_vals_rows_array) == 0:
            # there are no missing values in df
            return 
        
        # replace column indices with column names in missing_vals_cols_array
        missing_vals_cols_array = [df.columns[i] for i in missing_vals_cols_array];
        
        # initialize missing_vals_dict:
        missing_vals_dict = {}
        for col in missing_vals_cols_array:
            missing_vals_dict[col] = []
            
        # fill in the missing value row indices for each col that has a missing value
        for ind, col in zip(missing_vals_rows_array, missing_vals_cols_array):
            missing_vals_dict[col].append(ind)
        #print('missing_values_dict', missing_vals_dict)
        
        # write out missing_vals_dict as json file
        with open(self.missing_values_fname, 'w') as f:
            json.dump(missing_vals_dict, f, indent='\t', cls=np_JSONEncoder)
        # save missing_vals_dict in self. TODOD !!! this is not required in all analyis modes
        self._missing_vals_dict = missing_vals_dict
        
        return
        
    def preprocess_data(self, data_file:str, feat_names:list[str], resp_names:list[str], feat_names_dict:dict, 
            keep_feat:list[str], impute_resp:bool, data_version_str:str, pos_value:int, neg_value:int, resp_to_bool):
        self._data_logger.info('loading ' + data_version_str + ' data')
        data = pd.read_csv(data_file)
        self._data_logger.info('data summary\n' + str(data.describe()))
        #plot_data_columns(data)
        self._data_logger.info(data_version_str + ' data\n' + str(data))

        # sanity-check the response names aginst input data
        is_training = data_version_str == 'training'
        new_labeled = self._sanity_check_responses(data, resp_names, is_training)
        
        # if feature names are not provided, we assume all features in the data besides
        # the responses should be used in the analysis as input features.
        if feat_names is None and is_training:
            feat_names = [col for col in data.columns.tolist() if not col in resp_names]
        elif feat_names is None: # not is_training: infer features from feat_names_dict
            assert feat_names_dict is not None
            #print('feat_names_dict', feat_names_dict)
            feat_names = lists_union_order_preserving_without_duplicates(list(feat_names_dict.values()))
          
        if is_training:
            feat_names_dict = {}
            for rn in resp_names:
                feat_names_dict[rn] = feat_names
            #print('feat_names_dict used for feat_names', feat_names_dict)
        
        #print('data\n', data, '\n', 'feat_names', feat_names, 'resp_names', resp_names)
        # extract the required columns in data -- features and responses
        if is_training or new_labeled:
            data = data[feat_names + resp_names]
        else:
            data = data[feat_names]
        #print('data 0\n', data)
        
        # in training data, drop all rows where at least one response has a missing value
        if is_training:
            if not impute_resp:
                data = self._drop_rows_with_na_in_responses(data, resp_names, 'training'); #print('data 1\n', data)
            data, constant_feat = self._drop_constant_features(data, keep_feat, 'training'); #print('constant_feat', constant_feat); print('data 2\n', data)
            resp_names = [rn for rn in resp_names if not rn in constant_feat]; #print('resp_names', resp_names)
            feat_names = [fn for fn in feat_names if not fn in constant_feat]; #print('feat_names', feat_names)
            for rn in feat_names_dict.keys():
                if not rn in resp_names:
                    del feat_names_dict[rn]
                else:
                    for fn in feat_names_dict[rn]:
                        if fn not in feat_names:
                            feat_names_dict[rn].remove(fn)
            #print('feat_names_dict', feat_names_dict)
        # impute missing values; before doing that, save the missing values location information in 
        # self._missing_values_dict and write it out as json file.
        self._compute_missing_values_dict(data)
        imp = SimpleImputer(strategy="most_frequent")
        data[ : ] = imp.fit_transform(data)
        self._data_logger.info(data_version_str + ' data after imputing missing values\n' + str(data))

        # convert columns (feature and responses) of type bool, if any, to object/string type
        data = self._cast_boolean_features(data)
        
        # seprate features and responses, process them separately
        X, y = self._get_response_features(data, feat_names, resp_names, is_training, new_labeled)
        if is_training or new_labeled:
            y = self._preprocess_responses(y, pos_value, neg_value, resp_to_bool, is_training)
        if not y is None:
            assert set(resp_names) == set(y.columns.tolist())
            
        return X, y, feat_names, resp_names, feat_names_dict
    

    # Feature and response scaling (optional) will be done only on responses and features that occur
    # in model_features_dict. For the optimization task, X and y befoee scling are saved in self.
    def _scale_data(self, X, y, scale_features, scale_responses, scaler_type, mm_scaler_feat, mm_scaler_resp, 
            model_features_dict, is_training, data_version_str):
        # compute whether scaling is required
        scale = self._get_data_scaler(scaler_type) is not None
        scale_feat = scale_features and scale
        scale_resp = scale_responses and scale
        if is_training:
            # save unscaled training data, relevant for model exploration modes
            self._X_orig_scale = X
            self._y_orig_scale = y
            if scale_feat:
                mm_scaler_feat = self._get_data_scaler(scaler_type) #MinMaxScaler()
                mm_scaler_feat.fit(X)
                self._mm_scaler_feat = mm_scaler_feat
            if scale_resp and y is not None:
                mm_scaler_resp = self._get_data_scaler(scaler_type) #MinMaxScaler()
                mm_scaler_resp.fit(y)
                self._mm_scaler_resp = mm_scaler_resp
                #print('mm_scaler_feat as computed', mm_scaler_feat.scale_, mm_scaler_feat.feature_names_in_);
                #print('mm_scaler_resp as computed', mm_scaler_resp.scale_, mm_scaler_resp.feature_names_in_);
        #print('mm_scaler_feat', mm_scaler_feat); print('mm_scaler_resp', mm_scaler_resp)        
        if scale_feat:
            X = pd.DataFrame(mm_scaler_feat.transform(X), columns= X.columns)
        else:
            assert mm_scaler_feat is None
        if scale_resp and y is not None:
            y = pd.DataFrame(mm_scaler_resp.transform(y), columns=y.columns)
        elif is_training:
            assert mm_scaler_resp is None
        
        # verbosity messages
        if scale_feat and scale_resp:   
            self._data_logger.info(data_version_str + ' data after scaling (normalizing) features and responses\n' + 
                                   str(pd.concat([X,y], axis=1)))        
        elif scale_feat:
            self._data_logger.info(data_version_str + ' data after scaling (normalizing) features\n' + 
                                   str(pd.concat([X,y], axis=1)))
        elif scale_resp and y is not None:
            self._data_logger.info(data_version_str + ' data after scaling (normalizing) responses\n' + 
                                   str(pd.concat([X,y], axis=1)))
            
        #if scale:
        #    print('mm_scaler_feat as returned', mm_scaler_feat.scale_, mm_scaler_feat.feature_names_in_)
        #    print('mm_scaler_resp as returned', mm_scaler_resp.scale_, mm_scaler_resp.feature_names_in_)
        
        return X, y, mm_scaler_feat, mm_scaler_resp, #feat_names, resp_names, model_features_dict
    
    
    # Split data into training and test sets. TODO !!! do we rather use seed instead of 17?
    # Another (maybe more robust) to make the split deterministic without using seed is using :
    # hashing https://engineeringfordatascience.com/posts/ml_repeatable_splitting_using_hashing/.
    # The argument split_test defines 
    # In some cases it is more convenient to use only subset of training data, and arguments 
    # train_first_n, train_random_n, train_uniform_n are used to further subset samples for training
    # from training data separated from the entire labeled data available for training using the value
    # of split_test.
    def _split_data_for_training(self, X, y, split_test, train_first_n, train_random_n, train_uniform_n):
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
        if False and y_test is not None:
            #print(y_test.head()); print(X_test.head())
            filter_test_samples = y_test[resp_names[0]] > 0.9
            y_test = y_test[filter_test_samples]; 
            #print('y_test with y_test > 0.9', y_test.shape); print(y_test.head())
            X_test = X_test[filter_test_samples]; 
            #print('X_test with y_test > 0.9', X_test.shape); print(X_test.head())

        return X_train, y_train, X_test, y_test
    
    # load data, scale using sklearn MinMaxScaler(), then split into training and test 
    # subsets with ratio given by split_test. Optionally (based on arguments train_first_n, 
    # train_random_n, train_uniform_n), subsample/resample training data using function
    # sample_first_random_unifirm() explained above. Optionally (hard coded), subset also
    # the test data to see how the model performs on a subset of interest in test data 
    # (say samples where the responses have high values). 
    # Select from data only relevant features -- the responses that must be provided, and
    # required input features which are either specified using feat_names or are computed
    # from the data (features that are not responses are used in analysis as input features).
    # Sanity check on response names resp_names and feat_names are also performed.
    # Besides training and test subsets, the function returns also the MinMaxScaler object 
    # used for data normalization, to be reused for applying the model to unseen datasets
    # and also to rescale back the prediction results to the original scale where needed.
    def _prepare_data_for_modeling(self, data_file:str, is_training:bool, split_test:float, 
            feat_names:list[str], resp_names:list[str], keep_feat:list[str], out_prefix:str, 
            train_first_n:int, train_random_n:int, train_uniform_n:int, interactive_plots:bool, 
            response_plots:bool, mrmr_features_n:int, pos_value:int, neg_value:int, resp_to_bool:str, 
            scaler_type:str, scale_features:bool, scale_responses:bool, impute_responses:bool, 
            mm_scaler_feat=None, mm_scaler_resp=None, levels_dict=None, model_features_dict=None):
        data_version_str = 'training' if is_training else 'new'
        self._data_logger.info('Preparing ' + data_version_str + ' data for modeling: start')

        # sanity check that the function is called correctly (as intended)
        if is_training:
            assert levels_dict==None
            assert mm_scaler_feat==None
            assert mm_scaler_resp==None

        # basic pre-processing, including inferring feature and response names, dropping rows 
        # where a response value is missing; imputing missing values in features.
        X, y, feat_names, resp_names, model_features_dict = self.preprocess_data(data_file, 
            feat_names, resp_names, model_features_dict, keep_feat, impute_responses, data_version_str, 
            pos_value, neg_value, resp_to_bool)
        if not y is None:
            assert set(resp_names) == set(y.columns.tolist())
        
        
        # TMP !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! start
        try_correlations = False
        if try_correlations: # use test 46
            instCorr = SmlpCorrelations()
            instCorr.mean_encode_categorical_feature_to_ordered(X['WAFER_ID'], y['PF'], reuse_levels=True)
            instCorr.mean_encode_categorical_feature_to_numeric(X['WAFER_ID'], y['PF'])
            instCorr.resp_mi_corr_feat(X, y[resp_names[0]], resp_names[0], float, X.dtypes, 'pearson', 'features', 
                'uniform')
            assert False
        # TMP !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! end
        
        # Feature selection / MRMR go here, will refine model_features_dict
        if is_training:
            #keep_feat = keep_feat + self._specInst.get_spec_constraint_vars(); #print('keep_feat', keep_feat)
            #print('features before mrmr', feat_names)
            for rn in resp_names:
                mrmr_feat = self._mrmrInst.mrmr_regres(X, y[rn], rn, mrmr_features_n); #print('mrmr_feat', mrmr_feat)
                model_feat = [ft for ft in feat_names if (ft in mrmr_feat or ft in keep_feat)]; #print(model_feat); 
                model_features_dict[rn] = model_feat #mrmr_feat
            feat_names = [ft for ft in feat_names if ft in 
                lists_union_order_preserving_without_duplicates(list(model_features_dict.values()))]
            X = X[feat_names]
            #print('features after mrmr', feat_names); print('model_features_dict after MRMR', model_features_dict)
        
        # encode levels of categorical features as integers for model training (in feature selection tasks 
        # it is best to use the original categorical features). 
        X, levels_dict = self._encode_categorical(X, levels_dict)
        self._data_logger.info(data_version_str + ' data after encoding levels of categorical features with integers\n' + 
                               str(pd.concat([X,y], axis=1)))
        
        #print('X\n', X); print('y\n', y); 
        # feature and response scaling (optional) will be done only on responses and features that occur
        # in model_features_dict. For the optimization task, X and y are saved in self before scaling
        # X, y, mm_scaler_feat, mm_scaler_resp, feat_names, resp_names, model_features_dict = \
        X, y, mm_scaler_feat, mm_scaler_resp = self._scale_data(X, y, scale_features, scale_responses, 
            scaler_type, mm_scaler_feat, mm_scaler_resp, model_features_dict, is_training, data_version_str)
        
        # plot responses
        if y is not None and response_plots:
            response_distribution_plot(out_prefix, y, resp_names, interactive_plots)

        # split data into training and test sets. TODO !!! do we rather use seed instead of 17?
        # another (maybe more robust) to make the split deterministic without using seed is using
        # hashing https://engineeringfordatascience.com/posts/ml_repeatable_splitting_using_hashing/
        if is_training:
            X_train, y_train, X_test, y_test = self._split_data_for_training(X, y, split_test, 
                train_first_n, train_random_n, train_uniform_n)
            res = X, y, X_train, y_train, X_test, y_test, mm_scaler_feat, mm_scaler_resp, \
                feat_names, resp_names, levels_dict, model_features_dict
        else:
            res = X, y

        self._data_logger.info('Preparing ' + data_version_str + ' data for modeling: end')
        return res    
                
    # Process data to prepare components required for training models and prediction, and reporting results in
    # original scale. Supports also prediction and results reporting in origibal scale from saved model
    def process_data(self, report_file_prefix:str, data_file:str, new_data_file:str, is_training:bool, split_test, 
            feat_names:list[str], resp_names:list[str], keep_feat:list[str],  
            train_first_n:int, train_random_n:int, train_uniform_n:int, interactive_plots:bool, response_plots:bool,
            scaler_type:str, scale_features:bool, scale_responses:bool, impute_responses:bool, mrmr_features_n:int, 
            pos_value, neg_value, resp_to_bool, save_model:bool, use_model:bool):
        
        #scale = not self._get_data_scaler(scaler_type) is None
        keep_feat = keep_feat + self._specInst.get_spec_constraint_vars(); #print('keep_feat', keep_feat)
        if data_file is not None:
            split_test = self._DEF_SPLIT_TEST if split_test is None else split_test
            X, y, X_train, y_train, X_test, y_test, mm_scaler_feat, mm_scaler_resp, \
            feat_names, resp_names, levels_dict, model_features_dict = self._prepare_data_for_modeling(
                data_file, True, split_test, feat_names, resp_names, keep_feat, report_file_prefix, 
                train_first_n, train_random_n, train_uniform_n, interactive_plots, response_plots,
                mrmr_features_n, pos_value, neg_value, resp_to_bool, scaler_type, 
                scale_features, scale_responses, impute_responses, None, None, None, None)
            
            # santy check that: mm_scaler_feat is not None --> scaler_type != 'none'
            assert not scaler_type == 'none' or mm_scaler_feat is None
            assert not scaler_type == 'none' or mm_scaler_resp is None
            
            self._save_model_levels(levels_dict, self.model_levels_dict_file)
            self._save_model_features(resp_names, feat_names, self.model_features_dict_file)
            self._save_data_scaler(scale_features, scale_responses, mm_scaler_feat, mm_scaler_resp, 
                self.features_scaler_file, self.responses_scaler_file)
            self._save_data_bounds(pd.concat([X,y], axis=1), feat_names, resp_names, self.data_bounds_file, 
                                   mm_scaler_feat, mm_scaler_resp)
        else:
            assert use_model
            #print('features_scaler_file before use', self.features_scaler_file); 
            #print('responses_scaler_file before use', self.responses_scaler_file)
            mm_scaler_feat, mm_scaler_resp = self._load_data_scaler(scale_features, scale_responses, 
                self.features_scaler_file, self.responses_scaler_file)
            levels_dict = self._load_model_levels(self.model_levels_dict_file)
            model_features_dict = self._load_model_features(self.model_features_dict_file)
            feat_names = lists_union_order_preserving_without_duplicates(list(model_features_dict.values()))
            #print('model_features_dict loaded', model_features_dict); print('feat_names loaded', feat_names)
            X, y, X_train, y_train, X_test, y_test =  None, None, None, None, None, None 
        
        if new_data_file is not None:
            X_new, y_new = self._prepare_data_for_modeling(
                new_data_file, False, None, feat_names, resp_names, keep_feat, report_file_prefix,  None, None, None, 
                interactive_plots, response_plots, mrmr_features_n, pos_value, neg_value, resp_to_bool, scaler_type,
                scale_features, scale_responses, impute_responses, mm_scaler_feat, mm_scaler_resp, levels_dict, model_features_dict)
        else:
            X_new, y_new = None, None

        # make sure data and new data have the same features to be used in the models
        if not data_file is None and not new_data_file is None:
            common_features = list_intersection(X.columns.tolist(), X_new.columns.tolist())
            X = X[common_features]
            X_train = X_train[common_features]
            X_test = X_test[common_features]
            X_new = X_new[common_features]
        
        return X, y, X_train, y_train, X_test, y_test, X_new, y_new, mm_scaler_feat, mm_scaler_resp, \
            levels_dict, model_features_dict, feat_names, resp_names