# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

from mrmr import mrmr_classif, mrmr_regression
import pandas as pd
from smlp_py.smlp_utils import (pd_series_is_binary_int, pd_series_is_binary_categorical, pd_series_is_numeric, pd_series_is_int)

class SmlpMrmr:
    def __init__(self):
        self._mrmr_logger = None
        self._MRMR_FEATURES_PRED = 15
        self._MRMR_FEATURES_CORR = 15
        self.mrmr_params_dict = {
            'mrmr_feat_count_for_prediction': {'abbr':'mrmr_pred', 'default':self._MRMR_FEATURES_PRED, 'type':int,
                'help':'Count of features selected by MRMR algorithm for predictive models '  +
                    '[default: {}]'.format(str(self._MRMR_FEATURES_PRED))},
            'mrmr_feat_count_for_correlation': {'abbr':'mrmr_corr', 'default':self._MRMR_FEATURES_CORR, 'type':int,
                'help':'Count of features selected by MRMR algorithm for correlation analysis '  +
                    '[default: {}]'.format(str(self._MRMR_FEATURES_CORR))}
        }
        
    # set logger from a caller script
    def set_logger(self, logger):
        self._mrmr_logger = logger
        
    # select_datatypes recognizes as 'categorical' features that were declared as 'category' 
    # during creation of the datatframe (or maybe later, after creation as well?)
    # Say DataFrame({'A' : Series(range(3)).astype('category'), ...| declares column 'A' as 'category'
    def _get_df_categorical_feat_names(self, df):
        return df.select_dtypes(include=['category', 'object']).columns.tolist()
    
    # Extract the computed scores from MRMR algo result mrmr_res and return it as a dataframe.
    # MRMR result mrmrm_res is a list of three elements where the first element is the ranked
    # list of selected features, the second element contains the feature scores; and the third
    # element is a matrix of mutual correlation scores between all pairs of input features. 
    def _mrmr_res_to_scores_df(self, mrmr_res):
        mrmr_scores_df = pd.DataFrame(mrmr_res[1])
        mrmr_scores_df = mrmr_scores_df[mrmr_scores_df.index.isin(mrmr_res[0])]
        mrmr_scores_df.index.name = 'Feature'
        mrmr_scores_df.reset_index(inplace=True)
        mrmr_scores_df.columns = ['Feature', 'Score']
        mrmr_scores_df = mrmr_scores_df.sort_values('Score', ascending=False)
        #print(mrmr_scores_df)
        return mrmr_scores_df
    
    # mrmr feature selection using mrmr-feature package, where y is a numeric variable (pandas.Series)
    def _mrmr_regres(self, X:pd.DataFrame, y:pd.Series, K:int, relevance='f', redundancy='c', denominator='mean',
            cat_encoding='leave_one_out', only_same_domain=False, return_scores=True, n_jobs=-1, show_progress=False):
        if K == 0: # or X.shape[1] <= 1: #K:
            if K > 0:
                self._mrmr_logger.info('Skipping MRMR feature selection for response ' + y.name)
            return X.columns.tolist(), None
        
        self._mrmr_logger.info('MRMR feature selection for response ' + y.name + ' : start')
        
        ctg_features = self._get_df_categorical_feat_names(X); #print('ctg_features', ctg_features)
        mrmr_res = mrmr_regression(X, y, K, relevance, redundancy, denominator,
            ctg_features, cat_encoding, only_same_domain, return_scores, n_jobs, show_progress)
        
        # log the selected features and their scores as a dataframe
        mrmr_scores_df = self._mrmr_res_to_scores_df(mrmr_res)
        self._mrmr_logger.info('MRMR selected feature scores (in the ranked order) for response ' + \
                               str(y.name) + ' :\n'+ str(mrmr_scores_df))
        
        self._mrmr_logger.info('MRMR feature selection for response ' + y.name + ' : end') 
        return mrmr_res[0], mrmr_scores_df

    # mrmr feature selection using mrmr-feature package, where y is a categorical variable (pandas.Series)
    # TODO !!!: not tested
    def _mrmr_class(self, X:pd.DataFrame, y:pd.Series, K:int, relevance='f', 
            redundancy='c', denominator='mean', cat_encoding='leave_one_out', only_same_domain=False,
            return_scores=True, n_jobs=-1, show_progress=False):
        if K == 0: # or X.shape[1] <= 1: #K:
            if K > 0:
                self._mrmr_logger.info('Skipping MRMR feature selection for response ' + y.name)
            return X.columns.tolist(), None
        
        self._mrmr_logger.info('MRMR feature selection for response ' + y.name + ' : start')
        
        ctg_features = self._get_df_categorical_feat_names(X); #print('ctg_features', ctg_features)
        mrmr_res = mrmr_classif(X, y, K, relevance, redundancy, denominator,
            ctg_features, cat_encoding, only_same_domain, return_scores, n_jobs, show_progress)
        
        # log the selected features and their scores as a dataframe
        mrmr_scores_df = self._mrmr_res_to_scores_df(mrmr_res)
        self._mrmr_logger.info('MRMR selected feature scores (in the ranked order) for response ' + \
                               y.name + ' :\n'+ str(mrmr_scores_df))
        
        self._mrmr_logger.info('MRMR feature selection for response ' + y.name + ' : end')
        return mrmr_res[0], mrmr_scores_df
    
    def smlp_mrmr(self, X:pd.DataFrame, y:pd.Series, #resp_type:str, #"numeric", 
            feat_cnt:int):
        #print('smlp_mrmr: X\n', X, '\ny\n', y, 'feat_cnt', feat_cnt)
        if pd_series_is_binary_int(y) or pd_series_is_binary_categorical(y):
            mrmr_res_pair = self._mrmr_class(X, y, feat_cnt, relevance='f', redundancy='c', 
                denominator='mean', cat_encoding='leave_one_out', only_same_domain=False,
                return_scores=True, n_jobs=-1, show_progress=False)
        elif pd_series_is_numeric(y) or (pd_series_is_int(y) and not pd_series_is_binary_int(y)):
            mrmr_res_pair = self._mrmr_regres(X, y, feat_cnt, relevance='f', redundancy='c', 
                denominator='mean', cat_encoding='leave_one_out', only_same_domain=False, 
                return_scores=True, n_jobs=-1, show_progress=False)
        else:
            raise Exception('Response of unsupported type ' + y.dtype.name + ' in function smlp_mrmr')
        #print('mrmr_res_pair\n', mrmr_res_pair)
        return mrmr_res_pair
            
