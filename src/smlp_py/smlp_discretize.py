# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

import pandas as pd
import numpy as np
import jenkspy
from sklearn.preprocessing import KBinsDiscretizer

from smlp_py.smlp_utils import cast_type, list_unique_ordered, str_to_bool


# useful links for discretization in Python used to build class SmlpDiscretize
# https://pypi.org/project/jenkspy/
# https://pbpython.com/natural-breaks.html
# https://pbpython.com/pandas-qcut-cut.html
# https://machinelearningmastery.com/discretization-transforms-for-machine-learning/ 

class SmlpDiscretize:
    def __init__(self):
        self._discr_logger = None
        
        # supported discretization algorithms
        self._DISCRETIZATION_ALGO_UNIFORM = 'uniform'
        self._DISCRETIZATION_ALGO_QUANTILE = 'quantile'
        self._DISCRETIZATION_ALGO_KMEANS = 'kmeans'
        self._DISCRETIZATION_ALGO_JENKS = 'jenks'
        self._DISCRETIZATION_ALGO_ORDINALS = 'ordinals'
        self._DISCRETIZATION_ALGO_RANKS = 'ranks'
        self._DEF_DISCRETIZATION_ALGO = self._DISCRETIZATION_ALGO_UNIFORM
        self._sklearn_algs = [self._DISCRETIZATION_ALGO_UNIFORM, self._DISCRETIZATION_ALGO_QUANTILE, 
            self._DISCRETIZATION_ALGO_KMEANS]
        self._discretization_algs = self._sklearn_algs + \
            [self._DISCRETIZATION_ALGO_JENKS, self._DISCRETIZATION_ALGO_ORDINALS, 
             self._DISCRETIZATION_ALGO_RANKS]
        
        # number of required bins after discretization
        self._DEF_DISCRETIZATION_BINS = 10
        
        # Whether to use integers as names of levels/values in the resulting discretized features
        # or to use string like Bin1, Bin2, ... The option value True means the latter, and string
        # self._label_prefix is used as prefix to integer values of the bins.
        self._DEF_DISCRETIZATION_LABELS = True
        self._label_prefix = 'Bin'
        
        # required data type of the resulting discretized features. In case the required data type 
        # is 'integer', the feature will be of type int (not of one of the types object, category or
        # irdered category used in pandas for categorical features).
        self._DISCRETIZATION_TYPE_CATEGORY = 'category'
        self._DISCRETIZATION_TYPE_ORDERED = 'ordered'
        self._DISCRETIZATION_TYPE_OBJECT = 'object'
        self._DISCRETIZATION_TYPE_INTEGER = 'integer'
        self._DEF_DISCRETIZATION_TYPE = self._DISCRETIZATION_TYPE_CATEGORY
        self._DISCRETIZATION_TYPES = [self._DISCRETIZATION_TYPE_CATEGORY, self._DISCRETIZATION_TYPE_ORDERED,
            self._DISCRETIZATION_TYPE_OBJECT, self._DISCRETIZATION_TYPE_INTEGER]
        
        # control parameters for discretization
        self.discr_params_dict = {
            'discretization_algo': {'abbr':'discr_algo', 'default':self._DEF_DISCRETIZATION_ALGO, 'type':str,
                'help':'Discretization algorithm to use. The possible options are: ' +
                    ' * "uniform": constracts constant-width bins; ' +
                    ' * "quantile": uses the quantiles values to have equally populated bins in each feature; ' +
                    ' * "kmeans": defines bins based on a k-means clustering performed on each feature independently; ' +
                    ' * "jenks": implements the Fisher-Jenks Natural Breaks algorithm; ' +
                    ' * "ordinals": converts the feature values into ordinals correponding to the location of the ' +
                    '   correponding value in the ascending sorted list of unique values in that feature. ' +
                    ' * "ranks": converts the feature values into ranks (ranks used in Spearman\'s rank correlation) ' +
                    '[default {}]'.format(str(self._DEF_DISCRETIZATION_ALGO))}, 
            'discretization_bins': {'abbr':'discr_bins', 'default':self._DEF_DISCRETIZATION_BINS, 'type':int,
                'help':'Number of required bins in a discretization algorithm ' + 
                    '[default {}]'.format(str(self._DEF_DISCRETIZATION_BINS))},
            'discretization_labels': {'abbr':'discr_labels', 'default':self._DEF_DISCRETIZATION_LABELS, 'type':str_to_bool,
                'help':'If true, string labels (e.g., "Bin2") will be used to denote levels of the categorical feature ' + 
                    'resulting from discretization; othewise integers (e.g., 2) will be used to represent the levels ' +
                    '[default {}]'.format(str(self._DEF_DISCRETIZATION_LABELS))},
            'discretization_type': {'abbr':'discr_type', 'default':self._DEF_DISCRETIZATION_TYPE, 'type':str,
                'help':'The type of the categorical feature resulting from discretization. Possible values are: ' +
                    ' * "object": the feature will be of type "object" -- with strings as values; ' +
                    ' * "category": the feature will be of pandas type "category" -- with levels unordered; ' +
                    '   these correspond to factors in statistics (and in R lamguage terminology) ' +
                    ' * "ordered": the feature will be of pandas type "category" -- with levels ordered; ' +
                    '   these correspond to ordered factors in statistics (and in R language terminology) ' +
                    ' * "integer": The feature will be of type int, its values will be the resulting bin ' +
                    '   numbers when enumerating the bins from left to right. ' + 
                    '[default {}]'.format(str(self._DEF_DISCRETIZATION_TYPE))},
            }
        
    # set logger from a caller script / module
    def set_logger(self, logger):
        self._discr_logger = logger 

    def _report_discretization_table(self, feat_df_discr, result_type):
        cat_dtypes = feat_df_discr.dtypes; #print('cat_dtypes', type(cat_dtypes), '\n', cat_dtypes)
        cat_dtypes_dict = dict(cat_dtypes); #print('cat_dtypes_dict\n', cat_dtypes_dict)
        cat_types_dict = {}
        for k, v in cat_dtypes_dict.items():
            #print('===============', k, 'type', type(v), type(v).name, feat_df_discr[k].dtype)
            #print('feat_df_discr[k].dtype', feat_df_discr[k].dtype, 'name', feat_df_discr[k].dtype.name)
            if feat_df_discr[k].dtype == 'category':
                assert isinstance(v, pd.core.dtypes.dtypes.CategoricalDtype)
                assert result_type in [self._DISCRETIZATION_TYPE_ORDERED, self._DISCRETIZATION_TYPE_CATEGORY]
                if v.ordered:
                    assert feat_df_discr[k].cat.ordered
                    cat_types_dict[k] = self._DISCRETIZATION_TYPE_ORDERED
                else:
                    cat_types_dict[k] = self._DISCRETIZATION_TYPE_CATEGORY
            elif feat_df_discr[k].dtype == object: #isinstance(v, np.dtype):
                cat_types_dict[k] = self._DISCRETIZATION_TYPE_OBJECT
            elif feat_df_discr[k].dtype == int: #isinstance(v, np.dtype):
                cat_types_dict[k] = self._DISCRETIZATION_TYPE_INTEGER
            else:
                raise Exception('Unexpected data type ' + str(feat_df_discr[k].dtype) + ' in discretized column')
        #print('cat_types_dict', cat_types_dict)
        cat_types_series = pd.Series(cat_types_dict)
        cat_types_df = cat_types_series.to_frame(name='type')
        #print('cat_types_df\n', cat_types_df); 
        
        def col_levels(col):
            if feat_df_discr[col].dtype == 'category':
                return str(list(feat_df_discr[col].cat.categories))
            else:
                return sorted(feat_df_discr[col].unique())
        cat_types_df['levels'] = [col_levels(col) for col in feat_df_discr.columns ]
        #print('cat_types_df\n', cat_types_df); 
        self._discr_logger.info('data after discretization\n' + str(feat_df_discr)) 
        self._discr_logger.info('feature data types\n' + str(cat_types_df))  
    
    # This function is a special case of smlp_discretize_df() in that it takes pandas Series
    # feat as an argument instead of a data frame feat_df and returns pandas Series result. 
    # The other arguments are similar to that of smlp_discretize_df() -- see the discreption below.
    # This function is called from smlp_discretize_df() when discretization is performed
    # per feature. In case of a single column data frame (which has an equivalent representation 
    # as pandas.Series), discretization results for uniform and quantile algorithms might be
    # slightly different different for smlp_discretize_df() and smlp_discretize_feature()
    # since the former used pd.cut() and pd.qcut() functions for discretizing pandas series and 
    # the latter uses KBinsDiscretizer from sklearn for the same purpose.
    # TODO: how to use default values self._DEF_DISCRETIZATION_ALGO and self_.DISCRETIZATION_BINS to set
    # the defaults in function smlp_discretize_df? Usage of say self._DEF_DISCRETIZATION_ALGO as default 
    # value in the function causes error "NameError: name 'self' is not defined"
    def smlp_discretize_feature(self, feat, algo='uniform', bins=10, labels=True, result_type='object'):
        #print('labels', labels)
        assert isinstance(feat, pd.Series)
        assert result_type in self._DISCRETIZATION_TYPES
        
        # when discretization target type is integer, labels cannot be used (thus labels should be False)
        if labels and result_type == self._DISCRETIZATION_TYPE_INTEGER:
            self._discr_logger.warning('setting discretization_labels to false since discretization_type is "integer"') 
            labels = False
        
        bins_count = min(bins, len(feat.unique()))
        if algo in self._sklearn_algs:
            feat_discr_df = pd.DataFrame(KBinsDiscretizer(n_bins=bins_count, encode='ordinal', 
                strategy=algo).fit_transform(feat.to_frame(name=feat.name)), columns=[feat.name])
            feat_discr = feat_discr_df[feat.name]
            feat_discr = feat_discr.astype(int)
            # define levels (names of values) within discretized feature
            if result_type in [self._DISCRETIZATION_TYPE_ORDERED, self._DISCRETIZATION_TYPE_CATEGORY]:
                levels = sorted(feat_discr.unique().tolist(), reverse=False)
            elif result_type == self._DISCRETIZATION_TYPE_OBJECT:
                feat_discr = feat_discr.astype(str)
            if labels:
                feat_discr = self._label_prefix + feat_discr.astype(int).astype(str)
                if result_type in [self._DISCRETIZATION_TYPE_ORDERED, self._DISCRETIZATION_TYPE_CATEGORY]:
                    levels = [self._label_prefix + str(l) for l in levels]
        elif algo == self._DISCRETIZATION_ALGO_JENKS:
            breaks = jenkspy.jenks_breaks(feat, n_classes=bins_count); #print('breaks', breaks)
            # Changing the samllest break point to -inf as in some cases this break point can 
            # be equal to next one, and pd.cut() does not accept break points with repitition.
            # changing the highest great point to inf is not strictly necessary if these
            # break points are not used to discretize the feature in new data.
            breaks[0] = float(-np.inf); breaks[-1] = float(np.inf); #print('breaks', breaks)
            if not labels: # or result_type == self._DISCRETIZATION_TYPE_INTEGER:
                pd_cut_labels = False
            else: 
                pd_cut_labels = [self._label_prefix+str(n) for n in range(len(breaks)-1)]; 
                #print('pd_cut_labels', pd_cut_labels)
            feat_discr = pd.cut(feat, bins=list_unique_ordered(breaks), labels=pd_cut_labels); 
            #print('jenks feat_discr', feat_discr.name, type(feat_discr))
            if result_type in [self._DISCRETIZATION_TYPE_ORDERED, self._DISCRETIZATION_TYPE_CATEGORY]:
                levels = range(len(breaks)-1)
                if labels:
                    levels = [self._label_prefix + str(l) for l in levels]
        elif algo == self._DISCRETIZATION_ALGO_ORDINALS:
            #print('feat.unique()', feat.unique())
            sorted_unique_vals = sorted(feat.unique()); #print('sorted_unique_vals', sorted_unique_vals)
            target_vals = range(len(feat.unique())); #print('target_vals', target_vals)
            feat_discr = feat.replace(to_replace=sorted_unique_vals, value=range(len(sorted_unique_vals)),
                inplace=False).astype(int); 
            if labels:
                feat_discr = self._label_prefix + feat_discr.astype(str)
            if result_type in [self._DISCRETIZATION_TYPE_ORDERED, self._DISCRETIZATION_TYPE_CATEGORY]:
                levels = target_vals
                if labels:
                    levels = [self._label_prefix + str(l) for l in levels]
            #feat_discr = feat_discr.to_frame(name=feat.name)
        elif algo == self._DISCRETIZATION_ALGO_RANKS:
            feat_df = feat.to_frame(name=feat.name); #print('feat_df\n', feat_df)
            feat_discr = feat_df.rank(axis=0, method='max', numeric_only=True, na_option='keep', 
                ascending=True, pct=False)[feat.name].astype(int)
            if labels:
                feat_discr = self._label_prefix + feat_discr.astype(str)
            if result_type in [self._DISCRETIZATION_TYPE_ORDERED, self._DISCRETIZATION_TYPE_CATEGORY]:
                levels = sorted(feat_discr.unique(), reverse=False)
        else:
            raise Exception('Unsupported discretization algorithm ' + str(algo))
        
        #print('feat_discr as object\n', feat_discr); print('type = ', type(feat_discr))
        assert isinstance(feat_discr, pd.Series)

        if result_type in [self._DISCRETIZATION_TYPE_ORDERED, self._DISCRETIZATION_TYPE_CATEGORY]:
            #print('levels---------', levels)
            category_type = pd.api.types.CategoricalDtype(categories=levels, ordered=result_type=='ordered')
            #print('category_type', category_type)
            feat_discr = feat_discr.astype(category_type)
        elif result_type == self._DISCRETIZATION_TYPE_OBJECT: 
            feat_discr = feat_discr.astype(str)
        elif result_type == self._DISCRETIZATION_TYPE_INTEGER:
            feat_discr = feat_discr.astype(int)
        #print('feat_discr as category\n', feat_discr)
        assert isinstance(feat_discr, pd.Series)
        return feat_discr

    # This function discretizes a dataframe of numeric features into categorical 
    # features if the argument ordered is False, else into an integer feature.
    # The supported discretization algorithms are:
    # * 'uniform' -- supported by sklearn KBinsDiscretizer and pandas pd.cut().
    #    Divides the feature range into k equal length intervals or based on user 
    #    supplied cut points, using a spec of the form [0.5, 2, 10, 20]. Such
    #    specs are currently not supported in SMLP (they are not exposed to user).
    #    Each interval is then treated as a level in the resulting categorical
    #    feature, or if ordered is True, these levels are represented as integers
    #    from 0 to k-1 (in the order that these levels first occur in the feature)
    # * 'quantile' -- supported by sklearn KBinsDiscretizer and pandas pd.qcut().
    #    Divides the feature range into intervals with equal sample counts or based
    #    on user supplied sample distribution spec of the form [0, 0.2, 0.5, 0.7, 1].
    #    Such specs are currently not supported in SMLP (are not exposed to user).
    # * 'kmeans' -- supported by sklearn KBinsDiscretizer. 
    #    Divides the feature range into intervals using the k-means clustering algo.
    # * 'jenks' -- supported in jenkspy package.
    #    Divides the feature range into k intervals using the Fisher-Jenks Natural Breaks 
    #    algorithm.
    # * 'ordinal' -- does not use any external package.
    #   Simply converts the feature values into ordinals correponding to the location of the
    #   correponding value in the ascending sorted list of unique values in that feature.
    # * 'ranks' -- uses pandas ranking implimenetation pd.DataFrame.rank().
    #   converts the feature values into ranks (ranks used in Spearman\'s rank correlation) '
    #
    #   Depending on the values of option discretization_labels, these ordinal values can be 
    #   prefixed with a string (e.g., a string like "Bin").
    #
    # If there are fewer values, say m, in the feature than the requested count bins 
    # of bins, bins is set to m.
    # TODO: how to use default values self._DEF_DISCRETIZATION_ALGO and self._DEF_DISCRETIZATION_BINS to set
    # the defaults in function smlp_discretize_df? Usage of say self._DEF_DISCRETIZATION_ALGO as default value
    # in the function causes error "NameError: name 'self' is not defined"
    def smlp_discretize_df(self, feat_df, algo='uniform', bins=10, labels=True, result_type='object'):
        numeric_cols = feat_df.select_dtypes(include=['int', 'float']).columns.tolist(); #print('numeric_cols', numeric_cols)
        feat_df_discr = []
        for col in feat_df.columns:
            if not col in numeric_cols:
                feat_df_discr.append(feat_df[col])
            else:
                feat_discr = self.smlp_discretize_feature(feat_df[col], algo, bins, labels, result_type)
                feat_df_discr.append(feat_discr)
                    
        feat_df_discr = pd.concat(feat_df_discr, axis=1)
        #print('feat_df_discr\n', feat_df_discr.dtypes, '\n', feat_df_discr)

        # log discretization info
        self._report_discretization_table(feat_df_discr, result_type)
        return feat_df_discr
    
    
