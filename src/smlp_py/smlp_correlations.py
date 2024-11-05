# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

# This module is experiemnal
import pandas as pd
import numpy as np
#from scipy.stats import chi2_contingency
from scipy.stats._stats import _kendall_dis
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.feature_selection import mutual_info_classif
import scipy
from concurrent.futures import ProcessPoolExecutor, as_completed

from smlp_py.smlp_utils import (pd_series_is_numeric, pd_series_is_categorical, pd_series_is_int, pd_series_is_binary_int, \
                                pd_series_is_binary_categorical, pd_series_is_binary_numeric, pd_df_split_numeric_categorical, \
                                pd_series_is_object, pd_series_is_category, \
                                pd_df_convert_numeric_to_categorical, pd_series_is_ordered_category, pd_df_is_empty, \
                                cast_type, cast_series_type, str_to_bool, str_to_str_list)
from smlp_py.smlp_discretize import SmlpDiscretize
from smlp_py.smlp_mrmr import SmlpMrmr

PEARSON = "pearson"
SPEARMAN = "spearman"
KENDALL = "kendall"
FREQUENCY = "frequency"
SOMERSD = "somersd"  # somers2() correlation, [Dxy] feild
CRAMERSV = "cramersv" # mRMRe::correlate() with method="cramersv" and mRMRe::mim() for ordered categorical features
CINDEX = "cindex" # mRMRe::mim() and mRMRe::correlation() for numeric vs orderted categorical
RANDINDEX = "randindex"
ADJ_RAND_IND = "adj_rand_ind"
#ICC = "icc" # use when either feature or response is categorical and the other one is numeric
SOMERSD_CRAMERSV = "somersd_cramersv"
SOMERSD_RAND = "somersd_rand"
SOMERSD_ADJRAND = "somersd_adjRand"
PEARSON_SOMERSD = "pearson_somersd"
SPEARMAN_SOMERSD = "spearman_somersd"
KENDALL_SOMERSD = "kendall_somersd"
FREQUENCY_SOMERSD = "frequency_somersd"
PEARSON_CINDEX = "pearson_cindex"
SPEARMAN_CINDEX = "spearman_cindex"
KENDALL_CINDEX = "kendall_cindex"
FREQUENCY_CINDEX = "frequency_cindex"
PEARSON_CRAMERSV = "pearson_cramersv"
SPEARMAN_CRAMERSV = "spearman_cramersv"
KENDALL_CRAMERSV = "kendall_cramersv"
FREQUENCY_CRAMERSV = "frequency_cramersv"
MRMR_CONTINUOUS_ESTIMATORS = [PEARSON, SPEARMAN, KENDALL, FREQUENCY]

PEARSON_CODE = "Prs"
SPEARMAN_CODE = "Spr"
KENDALL_CODE = "Kdl"
FREQUENCY_CODE = "Freq"
CRAMERSV_CODE = "Cra"
SOMERSD_CODE = "Som"
CINDEX_CODE = "Cix"
SOMERSD_CRAMERSV_CODE = "DoV"
SOMERSD_RAND_CODE = "DoR"
SOMERSD_ADJRAND_CODE = "DaR"
PEARSON_SOMERSD_CODE = "PoD"
SPEARMAN_SOMERSD_CODE = "SoD"
KENDALL_SOMERSD_CODE = "KoD"
FREQUENCY_SOMERSD_CODE = "FoD"
PEARSON_CINDEX_CODE = "PoC"
SPEARMAN_CINDEX_CODE = "SoC"
KENDALL_CINDEX_CODE = "KoC"
FREQUENCY_CINDEX_CODE = "FoC"
PEARSON_CRAMERSV_CODE = "PoV"
SPEARMAN_CRAMERSV_CODE = "SoV"
KENDALL_CRAMERSV_CODE = "KoV"
FREQUENCY_CRAMERSV_CODE = "FoV"
RANDINDEX_CODE = "Rand"
ADJ_RAND_IND_CODE = "AdjRand"


ORIGIN_NUM = "numeric"
ORIGIN_CTG = "factor"
ORIGIN_BIN = "binary"

TYPE_NUM = "numeric"
TYPE_CTG = "factor"

REGRESSION = "regression"
CLASSIFICATION = "classification"


class SmlpCorrelations:
    def __init__(self):
        self._instDiscret = SmlpDiscretize()
        self._instMrmr = SmlpMrmr()
        self.params_dict = {}
    
        self._MUTUAL_INFORMATION_METHOD = 'normalized'
        self._CORRELATION_AND_MUTUAL_INFORMATION = True
        self._DISCRETIZE_NUMERIC_FEATURES = None
        self._CONTINUOS_CORRELATION_ESTIMATORS = ','.join([PEARSON,SPEARMAN])
        self.corr_params_dict = {
            'mutual_information_method': {'abbr':'mi_method', 'default':self._MUTUAL_INFORMATION_METHOD, 'type':str,
                'help':'The mutual information method to be used when computing feature correlation scores with responses. ' + 
                    'Supported options are "shannon", "normalized", and "adjusted", for Shannon\'s mutual information, ' +
                    'for the normalized mutual information, and the adjusted mutual information, respectively; in addition, ' +
                    'with the option value "correlation", the mutual information is computed from a correlation coefficient corr ' +
                    'between the feature and response using equation mi = -0.5 * log(1 - (corr**2)), which is primarily useful for '
                    'computing mutual information for (preferably) normally distributed continuous random variables ' +
                    '[default: {}]'.format(str(self._MUTUAL_INFORMATION_METHOD))},
            'correlations_and_mutual_information': {'abbr':'corr_and_mi', 'default':self._CORRELATION_AND_MUTUAL_INFORMATION, 'type':str_to_bool,
                'help':'Should correlation and mutual information between the features and the response(s) be computed when ' +
                    'computing scores for feature selection and ranking [default: {}]'.format(str(self._CORRELATION_AND_MUTUAL_INFORMATION))},
            'discretize_numeric_features': {'abbr':'discret_num', 'default':self._DISCRETIZE_NUMERIC_FEATURES, 'type':str_to_bool,
                'help':'The mutual information method to be used for discretizing numeric features, when computing ' + 
                    'feature correlation scores with responses [default: {}]'.format(str(self._DISCRETIZE_NUMERIC_FEATURES))},
            'continuous_correlation_estimators': {'abbr':'cont_est', 'default':self._CONTINUOS_CORRELATION_ESTIMATORS, 'type':str_to_str_list,
                'help':'Correlation estimators for continuous features, to be used in correlation, mutual information and ' +
                    'MRMR feature selection algorithms. The options are pearson, spearman, kendall and frequency, and any subset ' +
                    'of these specified thru a comma-separated string. In addition, the value "all" indicates that all the options ' +
                    'should be used and value "none" indicates that no options should be used. ' + 
                    '[default: {}]'.format(str(self._CONTINUOS_CORRELATION_ESTIMATORS))}
        }
        
        # abbriviated names for correlation concepts (estimators), to be used in log files
        self._estimator_code_dict = {
            PEARSON: PEARSON_CODE,
            SPEARMAN: SPEARMAN_CODE,
            KENDALL: KENDALL_CODE,
            FREQUENCY: FREQUENCY_CODE,
            SOMERSD: SOMERSD_CODE,
            CRAMERSV: CRAMERSV_CODE,
            CINDEX: CINDEX_CODE,
            SOMERSD_CRAMERSV: SOMERSD_CRAMERSV_CODE,
            PEARSON_SOMERSD: PEARSON_SOMERSD_CODE,
            SPEARMAN_SOMERSD: SPEARMAN_SOMERSD_CODE,
            KENDALL_SOMERSD: KENDALL_SOMERSD_CODE,
            FREQUENCY_SOMERSD: FREQUENCY_SOMERSD_CODE,
            SOMERSD_CRAMERSV: SOMERSD_CRAMERSV_CODE,
            PEARSON_CINDEX: PEARSON_CINDEX_CODE,
            SPEARMAN_CINDEX: SPEARMAN_CINDEX_CODE,
            KENDALL_CINDEX: KENDALL_CINDEX_CODE,
            FREQUENCY_CINDEX: FREQUENCY_CINDEX_CODE,
            PEARSON_CRAMERSV: PEARSON_CRAMERSV_CODE,
            SPEARMAN_CRAMERSV: SPEARMAN_CRAMERSV_CODE,
            KENDALL_CRAMERSV: KENDALL_CRAMERSV_CODE,
            FREQUENCY_CRAMERSV: FREQUENCY_CRAMERSV_CODE,
            RANDINDEX: RANDINDEX_CODE,
            ADJ_RAND_IND: ADJ_RAND_IND_CODE,
            SOMERSD_RAND: SOMERSD_RAND_CODE,
            SOMERSD_ADJRAND: SOMERSD_ADJRAND_CODE
        }
        
        # integer codes for correlation concepts (estimetors)
        self._estimator_int_code_dict = {
            PEARSON: 1,
            SPEARMAN: 2,
            KENDALL: 3,
            FREQUENCY: 4,
            SOMERSD: 5,
            CRAMERSV: 6,
            CINDEX: 7,
            SOMERSD_CRAMERSV: 8,
            PEARSON_SOMERSD: 9,
            SPEARMAN_SOMERSD: 10,
            KENDALL_SOMERSD: 11,
            FREQUENCY_SOMERSD: 12,
            PEARSON_CRAMERSV: 13,
            SPEARMAN_CRAMERSV: 14,
            KENDALL_CRAMERSV: 15,
            FREQUENCY_CRAMERSV: 16,
            RANDINDEX: 17,
            ADJ_RAND_IND: 18,
            SOMERSD_RAND: 19,
            SOMERSD_ADJRAND: 20
        }
        
        # convert integer code of correlations concepts (estimators) back to their names
        self._int_code_to_estimator_dict = {
            1: PEARSON,
            2: SPEARMAN,
            3: KENDALL,
            4: FREQUENCY,
            5: SOMERSD,   
            6: CRAMERSV, 
            7: CINDEX,    
            8: SOMERSD_CRAMERSV,
            9: PEARSON_SOMERSD,
            10: SPEARMAN_SOMERSD,
            11: KENDALL_SOMERSD,
            12: FREQUENCY_SOMERSD,
            13: PEARSON_CRAMERSV,
            14: SPEARMAN_CRAMERSV,
            15: KENDALL_CRAMERSV,
            16: FREQUENCY_CRAMERSV,
            17: RANDINDEX,
            18: ADJ_RAND_IND,
            19: SOMERSD_RAND,
            20: SOMERSD_ADJRAND
        }
    
    # set logger from a caller script / module
    def set_logger(self, logger):
        self._corr_logger = logger 
        self._instDiscret.set_logger(logger)
        self._instMrmr.set_logger(logger)
        
    def set_report_file_prefix(self, report_file_prefix):
        self.report_file_prefix = report_file_prefix
    
    def estimator_vector_to_estimator(self, est_vec):
        if len(est_vec) == 1:
            if est_vec[0] in MRMR_CONTINUOUS_ESTIMATORS + [SOMERSD, CINDEX, CRAMERSV]:
                est = est_vec[0]
            else:
                raise EstimatorVectorError(f"Unknown estimator vector {est_vec} in function estimator_vector_to_estimator")
        elif len(est_vec) == 2:
            est_vec_set = set(est_vec)
            if est_vec_set == {PEARSON, SOMERSD}:
                est = PEARSON_SOMERSD
            elif est_vec_set == {SPEARMAN, SOMERSD}:
                est = SPEARMAN_SOMERSD
            elif est_vec_set == {KENDALL, SOMERSD}:
                est = KENDALL_SOMERSD
            elif est_vec_set == {FREQUENCY, SOMERSD}:
                est = FREQUENCY_SOMERSD
            elif est_vec_set == {CRAMERSV, SOMERSD}:
                est = SOMERSD_CRAMERSV
            elif est_vec_set == {PEARSON, CINDEX}:
                est = PEARSON_CINDEX
            elif est_vec_set == {SPEARMAN, CINDEX}:
                est = SPEARMAN_CINDEX
            elif est_vec_set == {KENDALL, CINDEX}:
                est = KENDALL_CINDEX
            elif est_vec_set == {FREQUENCY, CINDEX}:
                est = FREQUENCY_CINDEX
            elif est_vec_set == {PEARSON, CRAMERSV}:
                est = PEARSON_CRAMERSV
            elif est_vec_set == {SPEARMAN, CRAMERSV}:
                est = SPEARMAN_CRAMERSV
            elif est_vec_set == {KENDALL, CRAMERSV}:
                est = KENDALL_CRAMERSV
            elif est_vec_set == {FREQUENCY, CRAMERSV}:
                est = FREQUENCY_CCRAMERSV
            elif est_vec_set == {RANDINDEX, SOMERSD}:
                est = SOMERSD_RAND
            elif est_vec_set == {ADJ_RAND_IND, SOMERSD}:
                est = SOMERSD_ADJRAND
            else:
                raise EstimatorVectorError(f"Unknown estimator vector {est_vec} in function estimator_vector_to_estimator")
        else:
            raise EstimatorVectorError(f"Unknown estimator vector {est_vec} in function estimator_vector_to_estimator")

        return est

    # compute the name of an estimator column in feature importance tables
    # estimtor is one of the continuous estimtors ("pearson", etc.)
    # method can be "mi" for mutual information, or "corr" or "corr2" for correlations
    def est_colname(self, estimator, mutual_information_method, method, resp_type):
        if method == "mi":
            if mutual_information_method == "normalized": #cluster
                est = "norm"
            elif mutual_information_method == "adjusted": #entropy
                est = "adjast"
            elif mutual_information_method == "shannon": 
                est = "shannon"
            elif mutual_information_method == "correlation":
                est = self._estimator_code_dict[estimator]
            else:
                raise ValueError(f"Unexpected MI method {mutual_information_method} in function est_colname")
        else:
            est = self._estimator_code_dict[estimator]

        if resp_type == "numeric":
            tp = "num"
        elif resp_type == "integer":
            tp = "int"
        elif resp_type == "binary":
            tp = "bin"
        elif resp_type == "ordered":
            tp = "ord"
        elif resp_type == "factor":
            tp = "ctg"
        else:
            tp = resp_type

        return f"{est}_{method}_{tp}"
    
    # function to check that the correlations that are actually run in mrmr_resp_mi_corr_feat match
    # the correlation names of correlations reported in the txt log file
    def method_to_name_check(self, method_vec, method_name, corr_vs_mim):
        #print('method_to_name_check: method_vec', method_vec, 'method_name', method_name, 'corr_vs_mim', corr_vs_mim)
        if len(method_vec) == 1:
            failed = (# TODO !!!! fails on tests 226,227 (method_vec[0] == CINDEX and method_name != SOMERSD) or
                      (not corr_vs_mim and method_vec[0] == PEARSON and method_name == PEARSON_SOMERSD) or
                      (not corr_vs_mim and method_vec[0] == SPEARMAN and method_name == SPEARMAN_SOMERSD) or
                      (not corr_vs_mim and method_vec[0] == KENDALL and method_name == KENDALL_SOMERSD) or
                      (not corr_vs_mim and method_vec[0] == FREQUENCY and method_name == FREQUENCY_SOMERSD))
        elif len(method_vec) == 2:
            failed = ((set(method_vec) == {SOMERSD, CRAMERSV} and method_name != SOMERSD_CRAMERSV and corr_vs_mim) or
                      (set(method_vec) == {SOMERSD, RANDINDEX} and method_name != SOMERSD_RAND and not corr_vs_mim) or
                      (set(method_vec) == {SOMERSD, ADJ_RAND_IND} and method_name != SOMERSD_ADJRAND and not corr_vs_mim) or
                      (set(method_vec) == {PEARSON, SOMERSD} and method_name != PEARSON_SOMERSD) or
                      (set(method_vec) == {SPEARMAN, SOMERSD} and method_name != SPEARMAN_SOMERSD) or
                      (set(method_vec) == {KENDALL, SOMERSD} and method_name != KENDALL_SOMERSD) or
                      (set(method_vec) == {FREQUENCY, SOMERSD} and method_name != FREQUENCY_SOMERSD))
        else:
            raise ValueError("Implementation error in function method_to_name_check")

        if failed:
            raise ValueError(f"method_to_name_check failed for method_vec {method_vec} and method_name {method_name}")

        return failed
    
    # choose which correlation method to use depending on types of feature and respsonse 
    # (which we want to correlate). cont_est is method for correlation between continuous feature:
    # pearson, spearman, or kendall.
    # TODO !!! adapt to not use resp_type as the resp.dtype and not as numeric, ordered, ...
    def mim_estimator(self, feat_type, resp_type, cont_est):
        #print('mim_estimator: feat_type', feat_type, 'resp_type', resp_type, 'cont_est', cont_est)
        ri_for_cv = False # use Rand Index instead of CramersV for binary ordered response and categorical
        # old feat_type_is_numeric = feat_type == "numeric"
        # old feat_type_is_ordered = feat_type == "ordered"
        resp_type_is_numeric = resp_type == "numeric"
        resp_type_is_ordered = resp_type == "ordered"
        feat_type_is_numeric = feat_type in ['int16', 'int32', 'int64', 'float32', 'float64']
        feat_type_is_ordered = feat_type in ['object', 'category']
        #print('feat_type_is_numeric', feat_type_is_numeric, 'feat_type_is_ordered', feat_type_is_ordered)
        if (feat_type_is_numeric and resp_type_is_numeric):
            est = cont_est
        elif feat_type_is_numeric and resp_type_is_ordered:
            est = CINDEX
        elif feat_type_is_ordered and resp_type_is_numeric:
            est = CINDEX
        elif feat_type_is_ordered and resp_type_is_ordered:
            if (ri_for_cv):
                  est = RANDINDEX #ADJ_RAND_IND
            else:
                  est = CRAMERSV
        else:
            assert False
        return est

    # https://www.kaggle.com/code/chrisbss1/cramer-s-v-correlation-matrix/notebook
    # This function works the same even if numeric features are passed to it -- they are treated as
    # categoorical, and does not matter with which type fatures are passed as categrical: object, 
    # category or ordered category
    def cramers_V(self, var1, var2):
        crosstab = np.array(pd.crosstab(var1, var2, rownames=None, colnames=None)) # Cross table building
        stat = scipy.stats.chi2_contingency(crosstab)[0] # Keeping of the test statistic of the Chi2 test
        obs = np.sum(crosstab) # Number of observations
        mini = min(crosstab.shape)-1 # Take the minimum value between the columns and the rows of the cross table
        return (stat/(obs*mini))
    
    def somers_D(self, var1, var2):
        crosstab = np.array(pd.crosstab(var1, var2, rownames=None, colnames=None)) # Cross table building
        stat = scipy.stats.somersed(crosstab); #print('somersed', stat)
        stat2 = scipy.stats.somersed(var1, var2)
        assert stat.statistic == stat2.statistic
        return stat.statistic
    
    def pearson_R(self, var1, var2):
        stat = scipy.stats.pearsonr(var1, var2); #print('pearson', stat, stat.statistic)
        return stat.statistic
    
    def spearman_R(self, var1, var2):
        stat = scipy.stats.spearmanr(var1, var2); #print('spearman', stat, stat.statistic)
        return stat.statistic
    
    def kendall_T(self, var1, var2):
        stat = scipy.stats.kendalltau(var1, var2); #print('spearman', stat, stat.statistic)
        return stat.statistic
    
    # This implementation of somersd_via_ktau is borrowed from 
    # https://stackoverflow.com/questions/59442544/is-there-an-efficient-python-implementation-for-somersd-for-ungrouped-variables
    # It is supposed to be much faster than scipy.stats.somersed() and an analog of R's implementation of Hmisc::somers2()
    def tau_a(self, x, y):
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        
        def count_rank_tie(ranks):
            cnt = np.bincount(ranks).astype('int64', copy=False)
            cnt = cnt[cnt > 1]
            # Python ints to avoid overflow down the line
            return (
              int((cnt * (cnt - 1) // 2).sum()),
              int((cnt * (cnt - 1.) * (cnt - 2)).sum()),
              int((cnt * (cnt - 1.) * (2*cnt + 5)).sum())
              )

        size = x.size
        perm = np.argsort(y)  # sort on y and convert y to dense ranks
        x, y = x[perm], y[perm]
        y = np.r_[True, y[1:] != y[:-1]].cumsum(dtype=np.intp)

        # stable sort on x and convert x to dense ranks
        perm = np.argsort(x, kind='mergesort')
        x, y = x[perm], y[perm]
        x = np.r_[True, x[1:] != x[:-1]].cumsum(dtype=np.intp)

        dis = scipy.stats._stats._kendall_dis(x, y)  # discordant pairs

        obs = np.r_[True, (x[1:] != x[:-1]) | (y[1:] != y[:-1]), True]
        cnt = np.diff(np.nonzero(obs)[0]).astype('int64', copy=False)

        ntie = int((cnt * (cnt - 1) // 2).sum())  # joint ties
        xtie, x0, x1 = count_rank_tie(x)     # ties in x, stats
        ytie, y0, y1 = count_rank_tie(y)     # ties in y, stats

        tot = (size * (size - 1)) // 2

        if xtie == tot or ytie == tot:
            return np.nan

        # Note that tot = con + dis + (xtie - ntie) + (ytie - ntie) + ntie
        #               = con + dis + xtie + ytie - ntie
        con_minus_dis = tot - xtie - ytie + ntie - 2 * dis
        n0 = (size * (size - 1)) / 2

        tau_a = con_minus_dis/n0

        return tau_a

    # counterpart of Hmisc::somers2() in R
    def somersd_via_ktau(self, x,y): 
        return (self.tau_a(x, y)/self.tau_a(x,x))
    
    # concordance correlation coefficient 
    # (not same as c-index which is for survival analysis, or time-to-event analysis).
    # code: https://github.com/stylianos-kampakis/supervisedPCA-Python/blob/master/Untitled.py
    # wiki: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient 
    def concordance_correlation_coefficient(self, y_true, y_pred,
                       sample_weight=None,
                       multioutput='uniform_average'):
        """Concordance correlation coefficient.
        The concordance correlation coefficient is a measure of inter-rater agreement.
        It measures the deviation of the relationship between predicted and true values
        from the 45 degree angle.
        Read more: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
        Original paper: Lawrence, I., and Kuei Lin. "A concordance correlation coefficient to evaluate reproducibility." Biometrics (1989): 255-268.  
        Parameters
        ----------
        y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
            Ground truth (correct) target values.
        y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
            Estimated target values.
        Returns
        -------
        loss : A float in the range [-1,1]. A value of 1 indicates perfect agreement
        between the true and the predicted values.
        """
        #print('y_pred, resp', y_pred.name); print(y_pred); print('y_true, feat', y_true.name); print(y_true)
        if pd_series_is_categorical(y_pred):
            y_pred = pd.DataFrame(y_pred).codes; #print(y_pred)
        if pd_series_is_categorical(y_true):
            y_true =  pd.Categorical(y_true).codes; #print('codes'); print(y_true); print('after')
        #assert False
        cor=np.corrcoef(y_true,y_pred)[0][1]

        mean_true=np.mean(y_true)
        mean_pred=np.mean(y_pred)

        var_true=np.var(y_true)
        var_pred=np.var(y_pred)

        sd_true=np.std(y_true)
        sd_pred=np.std(y_pred)

        numerator=2*cor*sd_true*sd_pred

        denominator=var_true+var_pred+(mean_true-mean_pred)**2

        return numerator/denominator
    '''     
    def somersd(score,target):
        score, target = (list(t) for t in zip(*sorted(zip(score, target))))
        ttl_num=len(score)
        bin=20;
        n=ttl_num/20;
        sum_target=sum(target);
        sum_notarget=ttl_num-sum_target;
        pct_target=[];
        pct_notarget=[];
        pct_target.append(0.0);
        pct_notarget.append(0.0);
        for i in range(1,bin):
        if (i!=bin):
        pct_target.append((sum(target[0:(i*n-1)])+0.0)/sum_target);
        pct_notarget.append((i*n-sum(target[0:(i*n-1)])+0.0)/sum_notarget)

        pct_target.append(1.0);
        pct_notarget.append(1.0);
        sd=[]
        for i in range(1,bin+1):
        sd.append((pct_target[i]+pct_target[i-1])*(pct_notarget[i]-pct_notarget[i-1]));
        somersd=1-sum(sd);
        return(somersd);
        loc="F:\WebSiteProjects\CART" #File location
        f=file(loc+"\pred_result_logit.csv");
        score=[];
        target=[];
        while True:
        line=f.readline()
        line=line[0:-1]
        if len(line)==0:
        break
        if len(line)!=0:
        sep=line.find(',')
        score.append(float(line[0:sep]))
        target.append(int(line[sep+1:]))
 
    somersd=somersd(score=score,target=target);
    '''
    # feat is supposed to be object, categorical or ordered categorical, and resp is supposed to be numeric.
    # This function then returns the response based mean encoding of feat as numeric feature whose value
    # on a given sample is the mean value of the response on samples having the same level as the given sample
    def mean_encode_categorical_feature_to_numeric(self, feat:pd.Series, resp:pd.Series):
        # sanity check that feat is categorical
        if not pd_series_is_categorical(feat):
            raise Exception('Incorrect call to function mean_encode_categorical_feature_to_numeric()')
        
        # the following is required for the conversion below to numeric feature to work properly
        if feat.dtype.name != 'object':
            feat = feat.astype(str)
        
        # convert feat to numeric type
        feat_resp_df = pd.concat([feat, resp], axis=1); #print('feat_resp_df\n', feat_resp_df)
        lookup = feat_resp_df.groupby(feat.name).mean(); #print('lookup\n', lookup); print(dict(lookup[resp.name]))
        mean_encoded_feat = feat.replace(dict(lookup[resp.name])); #print('mean_encoded_feat categorical\n', mean_encoded_feat)
        assert pd_series_is_numeric(mean_encoded_feat)
        return mean_encoded_feat

    
    # orders levels of categorical feature based on the mean values of the response on samples on each level.
    # argument reuse_levels=T implies that when turning categorical features into ordered, use same levels 
    # (instead of using numeric as levels which is a little easier for implementation)
    def mean_encode_categorical_feature_to_ordered(self, feat:pd.Series, resp:pd.Series, reuse_levels=True):
        mean_encoded_feat = self.mean_encode_categorical_feature_to_numeric(feat, resp)
        #print('mean_encoded_feat input\n', mean_encoded_feat)
        
        if not reuse_levels:
            # use the mean values of response on original levels as the levels in the constructed ordered ctg feature
            ordered_type = pd.api.types.CategoricalDtype(categories=sorted(mean_encoded_feat.unique().tolist(), 
                reverse=False), ordered=True) # ascending order
            mean_encoded_feat = mean_encoded_feat.astype(ordered_type)
        else:
            # keep the same levels as in the original feature feat but order based on mean_encoded_feat
            mean_encoded_feat.name = feat.name + '_original'
            feat_mean_feat_df = pd.concat([feat, mean_encoded_feat], axis=1); 
            #print('feat_mean_feat_df before sort\n', feat_mean_feat_df)
            feat_mean_feat_df = feat_mean_feat_df.sort_values(by=[mean_encoded_feat.name], ascending=True); 
            #print('feat_mean_feat_df after sort\n', feat_mean_feat_df)
            ordered_type = pd.api.types.CategoricalDtype(categories=feat_mean_feat_df[feat.name].unique().tolist(), ordered=True)
            mean_encoded_feat = feat.astype(ordered_type)
        #print('mean_encoded_feat output', mean_encoded_feat.tolist());
        return mean_encoded_feat

    # This function is only used to determine the names correlation concepts/types when reporting correlation scores.
    # resp_type = "factor" correponds to the case of 3-valued reponse like good/bad/notsure
    # resp_type = "ordered" correponds to binary response casted into ordered factor 
    # estimator is a continuous estimator to use (if appropriate), such as Pearson, Spearman, etc.
    def mim_correlation_type(self, resp_type, feat_types, estimator, is_bin_resp):
        #print('mim_correlation_type: resp_type', resp_type, 'feat_types\n', feat_types, '\nestimator', estimator, 'is_bin_resp', is_bin_resp)
        ri_for_cv = False # use Rand Index instead of CramersV for binary ordered response and categorical
        mim_cind_swap_fix = True # for numeric binary response and catagorical feature run CramersV instead of mim().
        assert feat_types is not None
        feat_types = [feat_types[0]]
        if resp_type in ["ordered", "factor"]:
            if feat_types == ["numeric"]:
                method_name = SOMERSD
            elif feat_types == ["ordered"]:
                if ri_for_cv:
                    method_name = RANDINDEX  # or ADJ_RAND_IND
                else:
                    method_name = CRAMERSV
            else:
                if ri_for_cv:
                    method_name = SOMERSD_RAND  # or SOMERSD_ADJRAND
                else:
                    method_name = SOMERSD_CRAMERSV
        elif resp_type == "numeric":
            if feat_types == ["numeric"]:
                if is_bin_resp:
                    method_name = SOMERSD
                else:
                    method_name = estimator
            elif feat_types == ["ordered"]:
                if is_bin_resp:  # tmp_cra_vs_som
                    method_name = CRAMERSV
                else:
                    method_name = SOMERSD
            elif is_bin_resp:
                method_name = SOMERSD_CRAMERSV
            elif estimator == PEARSON:
                if mim_cind_swap_fix and is_bin_resp:
                    method_name = PEARSON_CRAMERSV
                else:
                    method_name = PEARSON_SOMERSD
            elif estimator == SPEARMAN:
                if mim_cind_swap_fix and is_bin_resp:
                    method_name = SPEARMAN_CRAMERSV
                else:
                    method_name = SPEARMAN_SOMERSD
            elif estimator == KENDALL:
                if mim_cind_swap_fix and is_bin_resp:
                    method_name = KENDALL_CRAMERSV
                else:
                    method_name = KENDALL_SOMERSD
            elif estimator == FREQUENCY:
                if mim_cind_swap_fix and is_bin_resp:
                    method_name = FREQUENCY_CRAMERSV
                else:
                    method_name = FREQUENCY_SOMERSD
            else:
                raise ValueError("Implementation error in response type in function mim_correlation_type")
        else:
            raise ValueError("Implementation error in response type in function mim_correlation_type")

        return method_name

    def correlate_feat_resp(self, feat, resp, corr_method):
        #print('correlate_feat_resp'); print(feat.name, 'of type', feat.dtype); print(resp.name, 'of type', resp.dtype); print('corr_method', corr_method)
        if corr_method == PEARSON:
            assert pd_series_is_numeric(feat)
            assert pd_series_is_numeric(resp)
            corr = self.pearson_R(feat, resp)
        elif corr_method == SPEARMAN:
            assert pd_series_is_numeric(feat)
            assert pd_series_is_numeric(resp)
            corr = self.spearman_R(feat, resp)
        elif corr_method == KENDALL:
            assert pd_series_is_numeric(feat)
            assert pd_series_is_numeric(resp)
            corr = self.kendall_T(feat, resp)
        elif corr_method == CRAMERSV:
            corr = self.cramers_V(feat, resp)
        elif corr_method == SOMERSD:
            corr = self.somers_D(feat, resp)
        elif corr_method == CINDEX:
            corr = self.concordance_correlation_coefficient(feat, resp)
        else:
            raise Exception('Uexpected correlation ' + str(corr_method) + ' in function correlate_feat_resp')
        return corr
    
    
    # The next two methods for computing joint entropies (pairwise) and mutual information matrix are borrowed
    # from https://medium.com/latinxinai/computing-mutual-information-matrix-with-python-6ced9169bcb1.
    # That page also has a good exlanations of how these methods were implemented (with the aim to make them fast).
    def joint_entropies(self, data, nbins=None):
        n_variables = data.shape[-1]
        n_samples = data.shape[0]
        if nbins == None:
            nbins = int((n_samples/5)**.5)
        histograms2d = np.zeros((n_variables, n_variables, nbins, nbins))
        for i in range(n_variables):
            for j in range(n_variables):
                histograms2d[i,j] = np.histogram2d(data[:,i], data[:,j], bins=nbins)[0]
        probs = histograms2d / len(data) + 1e-100
        joint_entropies = -(probs * np.log2(probs)).sum((2,3))
        return joint_entropies

    # computing mutual information matrix -- counterpart of R's mim() function, by approximating
    # mutual probabilites of pairs of variables with repective joint frequences within histograms
    # that approximate distributions of each variable.
    def mutual_info_matrix(self, df, nbins=None, normalized=True):
        data = df.to_numpy(); #print('mutual_info_matrix: data\n', data)
        n_variables = data.shape[-1]; #print('n_variables', n_variables)
        j_entropies = self.joint_entropies(data, nbins); #print('j_entropies\n', j_entropies)
        entropies = j_entropies.diagonal(); #print('entropies', entropies)
        entropies_tile = np.tile(entropies, (n_variables, 1)); #print('entropies_tile\n', entropies_tile)
        sum_entropies = entropies_tile + entropies_tile.T; #print('sum_entropies', sum_entropies)
        mi_matrix = sum_entropies - j_entropies; #print('mi_matrix', mi_matrix)
        # TODO !!!!!! when normalized is True and sum_entropies has 0 values, 
        # NaN-s are produced, something likely is wrong in how normalization is done
        if normalized:
            mi_matrix = mi_matrix * 2 / sum_entropies
        return pd.DataFrame(mi_matrix, index=df.columns, columns=df.columns) 
    
    # check whether all features in data frame X are numeric, all are factor (categorical), 
    # all are ordered factor (ordered categorical), or alse are of mixed types
    def compute_feature_supertype(self, X:pd.DataFrame):
        # Determine feature types
        all_numeric = all(X.apply(lambda col: pd_series_is_numeric(col)))
        all_factor = all(X.apply(lambda col: pd_series_is_categorical(col)))
        #print('all_factor', all_factor); print(X)
        if all_factor:
            all_factor = all(X.apply(lambda col: not pd_series_is_ordered_category(col)))
            all_ordered = all(X.apply(lambda col: pd_series_is_ordered_category(col)))
        else:
            all_factor = all_ordered = False
        feat_type = "numeric" if all_numeric else "ordered" if all_ordered else "factor" if all_factor else "mixed"
        
        if feat_type == "integer":
            raise Exception("Implementation error in function compute_feature_supertype")
        return feat_type
    
    # compute correlation between feature feat and response resp using MRMR package.
    # not using currently because there is no control on which correlation concept to apply
    # (say even for continuous features, which one of pearson, slearman, kendall is applied?)
    def mrmr_correlate(self, feat:pd.Series, resp:pd.Series):
        _, mrmr_scores_df = self._instMrmr.smlp_mrmr(pd.DataFrame(feat), resp, 1)
        corr = mrmr_scores_df.iloc[0, 1]; #print('corr', corr)
        assert corr == mrmr_scores_df['Score'][0]
        return corr
    
    def sklearn_mutual_info_categorical(self, resp:pd.Series, feat:pd.Series, mi_method):
        if mi_method == 'normalized':
            mi_score = normalized_mutual_info_score(resp, feat)
        elif mi_method == 'adjusted':
            mi_score = adjusted_mutual_info_score(resp, feat)
        elif mi_method == 'shannon':
            mi_score = mutual_info_score(resp, feat)
        #print('mi_score', mi_score)
        return mi_score
    
    # numeric vs. numeric Pearson, Spearman, Kendall or concordance index 
    # numeric vs. ordered factor concordance index (Somers' Dxy) 
    # numeric vs. survival data concordance index (Somers' Dxy) 
    # ordered factor vs. ordered factor Cramer's V 
    # ordered factor vs. survival data concordance index (Somers' Dxy) 
    # survival data vs. survival data concordance index (Somers' Dxy)
    #        min() function from MRMR package:
    # Compute mutual information and correlation of selected features with response.
    # The mim method computes and returns a mutual information matrix when method="mi". 
    # and the correlations matrix when method="cor". Correlation between continuous 
    # features is estimated using an estimator specified thru continuous_estimator; 
    # currently: pearson, spearman, kendall, frequency are supported. 
    # The estimator for discrete features is Cramer's V
    # and for all other combinations, concordance index (also called cindex).
    # Usage of correlations in this function is as folows:
    # (a) the " corr" part
    # 1. numeric response vs numeric feature: continuous correlations
    # 2. binary ordered (ctg) resp vs numeric feat (binary or not): SomersD implimented by somers2()
    # 3. binary ordered (ctg) resp vs ctg feat (binary or not): CramersV implimented by somers2()
    # 4. numeric response (int or num) with ctg feature: correlate() with continuous estimator.
    #     corr_fix_numeric_resp fixes that to instead run correlate() with CINDEX
    # (b) the mim_cor part:
    # The following form equivalence classes wrt mi and correlations:
    # pearson_corr2_numeric pearson_corr_numeric 
    # spearman_corr2_numeric spearman_corr_numeric spearman_corr2_ordered 
    # spearman_mi_ordered pearson_mi_ordered kendall_mi_ordered frequency_mi_ordered
    # spearman_corr_ordered pearson_corr_ordered kendall_corr_ordered frequency_corr_ordered
    # The function computes mi, corr and corr2 correlations, which form ensemble Feature Selection.
    # TODO !!!! get rid of usage of resp_type, use instead resp.dtype (which is used to define orig_resp_time)
    def mrmr_resp_mi_corr_feat(self, df_feat:pd.DataFrame, resp:pd.Series, resp_type, orig_feat_types:list, 
            estimator:str, mutual_information_method, mode, discretization_algo:str, discretization_bins:int, 
            discretization_labels:bool, discretization_type:str): #, discret_algo:str
        #print('== resp_mi_corr_feat: resp ', resp.name, type(resp)); print(resp)
        #print('orig_feat_types\n', orig_feat_types); print('resp_type', resp_type)
        
        colnms = df_feat.columns.tolist(); #print(colnms)
        if df_feat.shape[1] == 0: # there are no features, only the response
            assert False  # TODO 
            return pd.DataFrame(colnms)

        # resp_type is defined in ensemble_features_single_response() and can be numeric, factor or ordered 
        # but not int or any other type)
        if resp_type == int:
            raise Exception('binary integer response sanity check failed')
        
        orig_resp_type = resp.dtype; #print('orig_resp_type', orig_resp_type); print(pd_series_is_int(resp))
        orig_resp_is_int = pd_series_is_int(resp); #print('orig_resp_is_int', orig_resp_is_int)
        is_bin_resp = pd_series_is_binary_int(resp); #print('is_bin_resp', is_bin_resp)
        
        if is_bin_resp and not pd_series_is_int(resp):
            raise Exception("Sanity check for binary responses has failed in function mrmr_resp_mi_corr_feat")
        
    
        # "corr_method" is defined as correlation concept that will actually be used to compute "corr" with
        # correlate() function; it could be different from estimators for numeric features like Pearson, 
        # Spearman, etc. And similarly, "mim_method" is the actual correlation concept used to compute 
        # "mim_cor" using function mim(). Say when response is numeric, 
        # then the actual method used depends on the type of the input feature: for categorical features 
        # cindex or SomersD will be used. In this case for "method_name" we will use PrsSom, SprSom, etc.
        # When response is ordered, depending on whether the feature is numeric or catagorical, method
        # will be SomersD of cramersv (Rand Index); and in this case we use SOMERSD_CRAMERSV 
        # (now SOMERSD_RAND) as method_name.
        corr_method = estimator
        mim_method = estimator
        
        # TODO !!! use compute_feature_supertype(), will require to slightly adapt mim_correlation_type() as well.
        # just t make code modular.
        unique_feat_types = []
        for feat in df_feat.columns.tolist():
            feat_series = df_feat[feat]
            #print('feat',  feat, type(feat_series), 'all', df_feat.columns.tolist())
            if pd_series_is_numeric(feat_series):
                unique_feat_types.append('numeric')
            elif pd_series_is_object(feat_series) or pd_series_is_category(feat_series):
                unique_feat_types.append('factor')
            elif pd_series_is_ordered_category(feat_series):
                 unique_feat_types.append('ordered')
            else:
                raise Exception('Unexpected response type in ensemble_features_single_response')
        unique_feat_types = list(set(unique_feat_types)); #print('unique_feat_types', unique_feat_types)
            
        method_name = self.mim_correlation_type(resp_type, unique_feat_types, estimator, is_bin_resp); #print(method_name)
        
        # required in somers2() and for computing concordance/discordance with concordance_pairs() as well as for MI
        resp_orig = resp; #print('resp\n', resp)
        # TODO !!!!!! casting is not required 
        #print('casting resp of type', resp.dtype, 'to type', resp_type)
        #resp = cast_series_type(resp, resp_type); print('casted\n', resp)
        def corr_mi_cor(feat):
            nm = feat.name; #print('feature name nm', nm)
            feat_type = feat.dtype.name; #print('feat_type', feat_type)
            cat_inp = feat_type in ["category", "CategoricalDtype", "object"] #"ordered"
            assert cat_inp == pd_series_is_categorical(feat)

            # initialize mim_mi
            mim_mi = None

            # for numeric response, on categorical features that are not obtained by discretizing numeric
            # features, we apply some transformation to improve accuracy of corelations to the response in 
            # case the response is integer/numeric
            feat_use_mean_corr = cat_inp and (pd_series_is_numeric(resp)) and not pd_series_is_numeric(feat)
            
            ################### correlate()
            # ordered binary response (classification when we treat response as ordered 0 < 1)
            # In this case we use CRAMERSV or SOMERSD (implementation somers2)
            # TODO !!!! fix condition orig_feat_types[ nm ] in ["numeric", "integer"] as it will never hapen
            if pd_series_is_numeric(resp) and is_bin_resp and cat_inp and pd_series_is_numeric(feat):
                #print('corr_mi_cor, case 1')
                #som2_res = Hmisc::somers2(as.numeric(feat), resp_orig)
                corr_method = CRAMERSV
                corr = self.correlate_feat_resp(feat, resp, corr_method)
                #corr = self.mrmr_correlate(feat, resp)
            elif resp.nunique() > 2:
                #print('corr_mi_cor, case 2')
                # pass method that would be used by mim() and run mim()
                corr_method = self.mim_estimator(feat_type, resp_type, estimator)
                #print(c("call 4 to correlate with", corr_method))
                corr = self.correlate_feat_resp(feat, resp, corr_method); #print('corr 1', corr)
                #corr = self.mrmr_correlate(feat, resp); print('corr 2', corr); assert False
                #if (corr_method == SOMERSD) corr = corr[1]  #CINDEX
            else:
                #print('corr_mi_cor, case 3')
                # TODO: fix likely required here as we call continuous estimator w/o first checking
                # which correlation is most appropriate; 
                #print(c("call 5 to correlate with", estimator))
                corr_method = estimator
                if feat_use_mean_corr:
                    #print('corr_mi_cor, case 3.1')
                    mean_encoded_feat = self.mean_encode_categorical_feature_to_numeric(feat, resp); #print('mean_encoded_feat', mean_encoded_feat.tolist())
                    # na_corr_fix <<- T  pearson correlation formula gives NA when one of thefeatures is constant
                    # TODO !!! check for the perason corr implementation used here; also -- why is 0 a good choice?
                    #if len(unique(mean_encoded_feat)) == 1: #na_corr_fix
                    if mean_encoded_feat.nunique() == 1:
                        corr = 0
                    else:
                        #corr = unlist(mRMRe::correlate(mean_encoded_feat, resp, method=estimator))
                        corr = self.correlate_feat_resp(mean_encoded_feat, resp, estimator)
                        #corr = self.mrmr_correlate(feat, resp)
                    # TODO !!! if the condition below holds, then corr computed a few lines above is redundant????
                    if is_bin_resp and estimator==PEARSON:
                        #corr = unlist(mRMRe::correlate(feat, cast_type(resp, "ordered"), method=CRAMERSV))
                        # TODO !!! need casting?
                        corr = self.correlate_feat_resp(feat, resp, CRAMERSV)
                else:
                        #print('corr_mi_cor, case 3.2')
                        corr = self.correlate_feat_resp(feat, resp, estimator)                
                
            # use corr_ver as correlation from which MI is inferred, and use somers2 as SemersD correlation
            # currently the inverse is done, is as good perhaps.
            # This way both versions of somers D (real one and somers2 which is more like c-index defined in slides)
            # will be represented, and MI will be defined from correlation ranging beyween -1 and 1, as usual
            # say for pearson or spearman.
            # Next steps would be (a) use other known to be useful correlations like point biserial, Lambda
            # and (b) instead of runing response as ordered add respective correlations when considering 
            # ressponse as binary integer (c) with binary ordered response and binary integer feature, turn 
            # the features as ordered and run cramers'v

            # ###################### mim()    
            #names(corr) = NULL
            # TODO !!!  definition in next line is not used
            curr_imp_df = df_feat[nm]; #print(curr_imp_df)
            if pd_series_is_numeric(resp) and is_bin_resp and not cat_inp: 
                #print('mim case 1')
                self.somersd_via_ktau(feat, resp)
                #som2_res = Hmisc::somers2(feat, resp_orig)
                #mim_cor = som2_res['Dxy']
                mim_cor = self.somersd_via_ktau(feat, resp_orig); #print('mim_cor', mim_cor)
                mim_method = SOMERSD
                # TODO: compare corr and corr_tmp
                #corr_res_tmp = unlist(correlate(df_feat[ , nm ], resp, method=CINDEX))
                #corr_tmp = corr_res_tmp["estimate"]
            elif pd_series_is_numeric(resp) and cat_inp and is_bin_resp:
                #print('mim case 2')
                #mim_cor = unlist(mRMRe::correlate(feat, cast_type(resp, "ordered"), method=CRAMERSV)); 
                # TODO !!! need casting?
                mim_cor = self.correlate_feat_resp(feat, resp, CRAMERSV)
                mim_method = CRAMERSV
            else:
                #print('mim case 3')
                # case covered here: 
                # (a) non-binary numeric response (with any type of feature)
                # (a1) for categorical feature, use its mean encoding like for feat_use_mean_corr, treat 
                # it as numeric and use MI for numeric features?; or use mean encoding of the feature to 
                # encode it to numeric feature (the mean values) rather than categorical?
                # (a2) for numeric feature, use MI for numeric features
                # (b) non-binary categorical response (levels mode) with any type of feature
                # (b1) if the feature is categorical compute MI fpr categorical features
                # (b2) if the feature is numeric ?????
                # (c) binary numeric response and numeric feature: compute MI for numeric features
                if feat_use_mean_corr:
                    #print('case feat_use_mean_corr')
                    if is_bin_resp: # cra_mean_prs and 
                        mim_cor = corr
                    else:
                        mean_encoded_feat = self.mean_encode_categorical_feature_to_ordered(feat, resp)
                        mean_encoded_feat_df = pd.DataFrame(mean_encoded_feat, dtype='category'); #print(mean_encoded_feat_df); print(nm)
                        assert mean_encoded_feat_df.columns.tolist() == [nm]
                        curr_imp_df = pd.concat([resp, mean_encoded_feat_df], axis=1)
                        #mim_cor_df = mim(curr_mrmr_data, 
                        #    # estimator -> NULL to be sure the cont estimator is not used
                        #    method="cor", continuous_estimator=NULL, bootstrap_count=0, prior_weight=0)
                        ###### mim_cor_df = self.mutual_info_matrix(curr_imp_df, nbins=None, normalized=True)
                        ######### mim_cor_df = mutual_info_classif(mean_encoded_feat_df, resp); 
                        #print(mim_cor_df)
                        #mim_cor = mim_cor_df[nm, resp.name];
                        mim_cor = self.sklearn_mutual_info_categorical(mean_encoded_feat, resp, 'normalized')
                        if is_bin_resp and not (mim_cor == corr).all().all():
                            #print(nm, cor, mim_cor) 
                            raise Exception('sanity check for binary responses in function corr_mi_cor has failed')
                else:
                    #print('case NOT feat_use_mean_corr')
                    #print(resp); print(curr_imp_df)
                    curr_imp_df = pd.concat([resp, curr_imp_df], axis=1)
                    if estimator==SPEARMAN and type(resp)=="numeric" and type(feat)=="numeric":
                        mim_cor = corr
                    else:
                        #mim_cor_df = mim(curr_mrmr_data,
                        #    method="cor", continuous_estimator=estimator, bootstrap_count=0, prior_weight=0)
                        mim_cor_df = self.mutual_info_matrix(curr_imp_df, nbins=None, normalized=False) #TODO !!!! was True
                        mim_cor = mim_cor_df.loc[nm, resp.name];
                mim_method = self.mim_estimator(feat_type, resp_type, estimator); #print('mim_method', mim_method)
            
            ################### computation of MI or normalized MI
            # Applying mutual information formula to continuous features does not work as one might expect, see
            # https://math.stackexchange.com/questions/2809880/mutual-information-of-continuous-variables
            # We therefore discretize feat and resp before applying Shannon's MI formula, where needed, and
            # in case mutual_information_method is "corr", we derive it from a correponding correlation score
            # mim_cor between feat nad response using formula mim_mi = -0.5 * np.log(1 - (mim_cor**2)).
            if (mutual_information_method in ["normalized",  "adjusted", "shannon"]):
                #print('MI case 1')
                #is_bin_feat = length(unique(feat)) == 2
                is_bin_feat = feat.nunique() == 2
                # both categorical; we run two different versions for normalized MI for 
                # resp == "ordered" and resp == "numeric" -- this was an arbitrary 
                # choice which one of these NMI versions to run in which of the two cases
                #print('is_bin_resp', is_bin_resp, 'orig_resp_type', orig_resp_type, 'cat_inp', cat_inp, 'is_bin_feat', is_bin_feat, 'resp_type', resp_type, 'feat_type', feat_type)
                if ((is_bin_resp or orig_resp_type in ["category", "CategoricalDtype", "object"]) and (cat_inp or is_bin_feat)):
                    #print("(is_bin_resp | orig_resp_type == factor) & (cat_inp | is_bin_feat)")
                    # for "ordered" binary response, use normalized MI from entropy package
                    if pd_series_is_ordered_category(resp): # resp has already been casted into one of the categorical types (object, category, ordered)
                        #print("ordered")
                        #mim_mi = nmi_ctg(feat, resp, mutual_information_method, mutual_information_algo)
                        mim_mi = self.sklearn_mutual_info_categorical(feat, resp, mutual_information_method)
                    # for numeric binary response, use aricode::AMI -- adjusted MI
                    elif pd_series_is_numeric(resp):
                        # aricode treats feat and resp as categorical: "a vector containing the labels of the classification"
                        #print("numeric")
                        #mim_mi = nmi_ctg(feat, resp, "cluster", "ami")
                        mim_mi = self.sklearn_mutual_info_categorical(feat, resp_orig, mutual_information_method)
                    else:
                        assert False
                elif pd_series_is_numeric(resp_orig) and (cat_inp or is_bin_feat):
                    resp_disc = self._instDiscret.smlp_discretize_feature(resp_orig, algo=discretization_algo, 
                        bins=discretization_bins, labels=discretization_labels, result_type=discretization_type)
                    #mim_mi = nmi_ctg(feat, resp_disc, mutual_information_method, mutual_information_algo)
                    mim_mi = self.sklearn_mutual_info_categorical(feat, resp_disc, mutual_information_method)
                elif orig_resp_type in ["category", "CategoricalDtype", "object"] and not cat_inp: # == "factor"
                    # feat can only be a propurly numeric (not binary 0/1 and not two-valued with other values?)
                    #print("orig_resp_type == factor & !cat_inp")
                    feat_disc = self._instDiscret.smlp_discretize_feature(feat, algo=discretization_algo, 
                        bins=discretization_bins, labels=discretization_labels, result_type=discretization_type)
                    mim_mi = nmi_ctg(feat_disc, resp_orig, mutual_information_method, mutual_information_algo)
                    mim_mi = self.sklearn_mutual_info_categorical(feat_disc, resp_orig, mutual_information_method)
                elif pd_series_is_numeric(resp_orig) and pd_series_is_numeric(feat) and not is_bin_feat:
                    # here feat can be both proper numeric and binary numeric and response is proper numeric
                    #print("orig_resp_type == numeric & feat_type == numeric")
                    # binary or numeric response and numeric feature
                    #print("entropy_ctg")
                    feat_disc = self._instDiscret.smlp_discretize_feature(feat, algo=discretization_algo, 
                        bins=discretization_bins, labels=discretization_labels, result_type=discretization_type)
                    resp_disc = self._instDiscret.smlp_discretize_feature(resp_orig, algo=discretization_algo, 
                        bins=discretization_bins, labels=discretization_labels, result_type=discretization_type)
                    #print('feat_disc\n', feat_disc);  print('resp_disc\n', resp_disc)
                    #mim_mi = nmi_ctg(feat_disc, resp_disc, mutual_information_method, mutual_information_algo)
                    mim_mi = self.sklearn_mutual_info_categorical(feat_disc, resp_disc, mutual_information_method)
                elif is_bin_resp and feat_type == "numeric":
                    # here feat can only be proper numeric (not 0/1)
                    #print("is_bin_resp & feat_type == numeric")
                    feat_disc = smlp_discretize(feat, discret_algo)
                    feat_disc = self._instDiscret.smlp_discretize_feature(feat, algo=discretization_algo, 
                        bins=discretization_bins, labels=discretization_labels, result_type=discretization_type)
                    mim_mi = nmi_ctg(feat_disc, resp_orig, mutual_information_method, mutual_information_method)
                    mim_mi = self.sklearn_mutual_info_categorical(feat_disc, resp_orig, mutual_information_method)
                else:
                    raise Exception("Missing case for MI computation")
                #print('mim_mi', mim_mi)
                
                # TODO !!! double check sanity checks below
                if mutual_information_method == "normalized":
                    if mim_mi > 1 or mim_mi < 0:
                        raise Exception("mim_mi test for [0,1] has failed for the normalized mutual information")
                elif mutual_information_method == "adjusted":
                    # it is legal (according to its formula) that mim_mi can be negative, 
                    # and this can be interpreted as 0 mutual info
                    if mim_mi < 0:
                        mim_mi == 0
                    elif mim_mi > 1:
                        raise Exception("mim_mi test for [-inf,1] has failed for the adjusted mutual information")
                elif mutual_information_method == "shannon":
                    if mim_mi < 0:
                        raise Exception("mim_mi test for [0,inf] has failed for (the regular) mutual information")
            elif mutual_information_method == 'correlation':
                #print('MI case 2')
                # computation of mim_cor is not deterministic because of R's implementation of operations on doubles.
                # say 2/49*49 does not equal 2.
                # as a result, doubles that are printed as 1 are not really equal to 1. still the difference is smaller 
                # than .Machine$double.eps^0.5 -- thus the following code to make deterministic MI computation from 
                # correlations that are equal to 1 under tolorance smaller than .Machine$double.eps^0.5.
                # this makes regression tests 234,241,243,245 and many more stable
                def machineEpsilon(func=float):
                    #print('func', func, '\nval', func(1))
                    machine_epsilon = machine_epsilon_last = func(1)
                    while func(1)+func(machine_epsilon) != func(1):
                        machine_epsilon_last = machine_epsilon
                        machine_epsilon = func(machine_epsilon) / func(2)
                    return machine_epsilon_last
                #print('mim_cor', mim_cor);print('mim_mi', mim_mi)
                if mim_cor is None:
                    mim_mi = np.nan
                elif abs(1-mim_cor) < (machineEpsilon(float))**0.5: #  .Machine$double.eps^0.5): 
                    mim_mi = np.inf 
                else: 
                    mim_mi = -0.5 * np.log(1 - (mim_cor**2))
                #earlier version, non-deterministic: mim_mi = -0.5 * log(1 - (mim_cor^2));
            else:
                raise Exception('Unexpected mutual_information_method ' + str(mutual_information_method))
                      
            #print([corr_method, mim_method])
            #print([mim_mi, mim_cor, corr, self._estimator_int_code_dict[corr_method], self._estimator_int_code_dict[mim_method]])
            return([mim_mi, mim_cor, corr, self._estimator_int_code_dict[corr_method], self._estimator_int_code_dict[mim_method]])
            # end of corr_mi_cor
                      
        # We'll use a DataFrame.apply() in Python to apply 'corr_mi_cor' to each column
        #print('df_feat\n', df_feat)
        importance_tbl = df_feat.apply(corr_mi_cor).T
        #print("1"); print(importance_tbl)

        # Convert the index to a column and reset the index
        importance_tbl = importance_tbl.reset_index()
        importance_tbl.columns = ['names'] + importance_tbl.columns[1:].tolist()
        #print("2"); print(importance_tbl)

        # the last column is the vector of correlations used
        # Extract the last column as a Series and apply 'int_code_to_estimator'
        corr_methods_codes_vec = importance_tbl.iloc[:, 4]
        corr_methods_vec = corr_methods_codes_vec.apply(lambda code: self._int_code_to_estimator_dict[code])
        corr_method_name = self.estimator_vector_to_estimator(corr_methods_vec.unique())
        #print('corr_methods_vec', corr_methods_vec.tolist()); print('corr_method_name', corr_method_name)

        # Extract the next-to-last column as a Series and apply 'int_code_to_estimator'
        mim_methods_codes_vec = importance_tbl.iloc[:, 5]
        mim_methods_vec = mim_methods_codes_vec.apply(lambda code: self._int_code_to_estimator_dict[code])
        #print('mim_methods_vec', mim_methods_vec.tolist())

        # Drop the last two columns from the DataFrame
        importance_tbl = importance_tbl.drop(importance_tbl.columns[[-2, -1]], axis=1)
        #print("3"); print(importance_tbl); 
        
        if is_bin_resp: #update_corr_names
            corr_name = self.est_colname(method_name, mutual_information_method, "corr", "bin")
            corr2_name = self.est_colname(corr_method_name, mutual_information_method, "corr2", "bin")
            mi_name = self.est_colname(method_name, mutual_information_method, "mi", "bin")
        else:
            corr_name = self.est_colname(method_name, mutual_information_method, "corr", resp_type)
            corr2_name = self.est_colname(corr_method_name, mutual_information_method, "corr2", resp_type)
            mi_name = self.est_colname(method_name, mutual_information_method, "mi", resp_type)
        
        dummy = self.method_to_name_check(list(set(corr_methods_vec)), corr_method_name, True); #print(dummy)
        dummy = self.method_to_name_check(list(set(mim_methods_vec)), method_name, False); #print(dummy)
                                          
        #print(corr_name, corr2_name, mi_name)
        importance_tbl = importance_tbl.set_axis(["important_features", mi_name, corr_name, corr2_name], axis='columns');
        #print("4"); print(importance_tbl);
                
        # Return the correlation result
        return importance_tbl

    # Finds features strongly correlated to the response using function mrmr_resp_mi_corr_feat.
    # Returns feature selection summary table importance_tbl. 
    # TODO !!! discretize_num_feat is not used
    def mrmre_resp_feat_corr(self, feat_df:pd.DataFrame, resp:pd.Series, resp_type, orig_feat_types, estimator:str, 
            mutual_information_method, corr_and_mi, discretization_algo:str, discretization_bins:int, discretization_labels:bool, 
            discretization_type:str, discretize_num_feat:bool, mrmr_tbl_incr_feat, mode):
        
        mrmr_selection = True # TODO !!!! need command line option
        range_pair_analysis = False # TODO !!!! 
        if (not mrmr_selection and not range_pair_analysis and not corr_and_mi):
            #print("exit 1 -- skip")
            return None
                
        # here we cbind the response as the last column, for feature numbers (as column numbers)
        # to stay consistent with feature numbers in the separate features data frames
        #df = pd.concat([feat_df, resp], axis=1)
        if pd_series_is_numeric(resp):
            classic_corr_feat_nms = mrmr_tbl_incr_feat['mrmr_incr_feat_num']
            mrmr_tbl = mrmr_tbl_incr_feat['mrmr_tbl_num']
        elif pd_series_is_ordered_category(resp):
            classic_corr_feat_nms = mrmr_tbl_incr_feat['mrmr_incr_feat_ord']
            mrmr_tbl = mrmr_tbl_incr_feat['mrmr_tbl_ord']
        else:
            raise Exception('Implementation error in functon mrmre_resp_feat_corr')

        if mode in ['train', 'predict', 'novelty']:
            importance_df = self.mrmr_resp_mi_corr_feat(feat_df[classic_corr_feat_nms], 
            resp, resp_type, orig_feat_types, estimator, mutual_information_method, mode, 
            discretization_algo, discretization_bins, discretization_labels, discretization_type)
        elif mode in ['features', 'levels']:
            #print('classic_corr_feat_nms', classic_corr_feat_nms)
            importance_df = self.mrmr_resp_mi_corr_feat(feat_df, 
                resp, resp_type, orig_feat_types, estimator, mutual_information_method, mode, 
                discretization_algo, discretization_bins, discretization_labels, discretization_type)
        else:
            raise Exception('Unexpected mode ' + str(mode) + ' in function mrmre_resp_feat_corr')
        return importance_df

    
    def mrmre_resp_feat_corr_multi_estimators(self, feat_df:pd.DataFrame, resp:pd.Series, resp_type:str, 
            orig_feat_types, continuous_estimators, mutual_information_method, corr_and_mi:bool,
            discretization_algo:str, discretization_bins:int, discretization_labels:bool, discretization_type:str,
            discretize_num_feat, fs_summary_df, mrmr_tbl_incr_feat, mode):
        #print("Function mrmre_resp_feat_corr_multi_estimators starts")
        #TODO !!!! not checked when True, seems to work but there is a non-determinism in the order in which 
        # the results for particular continuous estimators are combined into single table
        do_parallel_foreach_mrmr = False  # Set this to True if parallel execution is desired. 
        full_mrmr_list = []

        if do_parallel_foreach_mrmr and len(continuous_estimators) > 1:
            #print("[-v-] Running MRMR heuristics in parallel\n")
            with ProcessPoolExecutor() as executor:
                futures = {executor.submit(self.mrmre_resp_feat_corr, feat_df, resp, resp_type, 
                                           orig_feat_types, estimator, mutual_information_method, corr_and_mi,
                                           discretization_algo, discretization_bins, discretization_labels, discretization_type,
                                           discretize_num_feat, mrmr_tbl_incr_feat, mode): estimator 
                           for estimator in continuous_estimators}
                for future in as_completed(futures):
                    full_mrmr_list.append(future.result())
        else:
            for estimator in continuous_estimators:
                full_mrmr_list.append(self.mrmre_resp_feat_corr(feat_df, resp, resp_type, 
                    orig_feat_types, estimator, mutual_information_method, corr_and_mi, 
                    discretization_algo, discretization_bins, discretization_labels, discretization_type,
                    discretize_num_feat, mrmr_tbl_incr_feat, mode))
        
        # Helper function to append one file to another
        # def file_append(destination, source):
        #     try:
        #         with open(destination, 'ab') as outfile, open(source, 'rb') as infile:
        #             outfile.write(infile.read())
        #         return True
        #     except IOError:
        #         return False
        
        for i, importance_df in enumerate(full_mrmr_list):
            #print('i', i, 'importance_df\n', importance_df)
            if mode in ["features"] and not pd_df_is_empty(importance_df):
                #print('fs_summary_df before merge\n', fs_summary_df)
                fs_summary_df = pd.merge(fs_summary_df, importance_df, on="important_features", 
                                        how='outer', suffixes=('', '.x'+str(i)))
                #print('fs_summary_df after merge\n', fs_summary_df); print('merged cols', fs_summary_df.columns.tolist())
            # if do_parallel_foreach_mrmr and len(continuous_estimators) > 1:
            #     output_file = self.report_file_prefix + '_correlations'
            #     estimator = continuous_estimators[i]
            #     output_file_i = f"{output_file}_{self.estimator_code(estimator)}"
            #     success = file_append(output_file, output_file_i)
            #     if not success:
            #         raise Exception("File append failed in mrmre_resp_feat_corr_multi_estimators")
            #     os.remove(output_file_i)
        
        #print("Function mrmre_resp_feat_corr_multi_estimators exits")
        return fs_summary_df

    

    # combine mrmr features computed with response as numeric and response as ordered.
    # When limit_model_features is F, we just take the unique features in the union;
    # otherwise, we take the intersection and half of the remaining required amount
    # mrmr_feat_count we select from the firt set mrmr_incr_feat_1 and the other half 
    # from the second set mrmr_incr_feat_2
    def combine_model_features(self, mrmr_incr_feat_1, mrmr_incr_feat_2, mrmr_feat_count):
        #print("Function combine_model_features starts")
        #print('mrmr_incr_feat_1, mrmr_incr_feat_2', mrmr_incr_feat_1, mrmr_incr_feat_2)
        # TODO !!!!! limit_model_features is hard-coded
        limit_model_features = False
        if not limit_model_features:
            #print("Function combine_model_features exits")
            return list(set(mrmr_incr_feat_1).union(set(mrmr_incr_feat_2)))

        if not mrmr_incr_feat_1:
            return mrmr_incr_feat_2
        if not mrmr_incr_feat_2:
            return mrmr_incr_feat_1

        common_feat = list(set(mrmr_incr_feat_1).intersection(set(mrmr_incr_feat_2)))
        all_feat = list(set(mrmr_incr_feat_1).union(set(mrmr_incr_feat_2)))
        target_feat_count = min(mrmr_feat_count, max(len(mrmr_incr_feat_1), len(mrmr_incr_feat_2)))
        other_feat_count_each = int(np.ceil((target_feat_count - len(common_feat)) / 2))

        mrmr_incr_diff_feat_1 = list(set(mrmr_incr_feat_1).difference(set(common_feat)))
        mrmr_incr_diff_feat_2 = list(set(mrmr_incr_feat_2).difference(set(common_feat)))

        mrmr_incr_feat_1_other = mrmr_incr_diff_feat_1[:min(len(mrmr_incr_diff_feat_1), other_feat_count_each)]
        mrmr_incr_feat_2_other = mrmr_incr_diff_feat_2[:min(len(mrmr_incr_diff_feat_2), other_feat_count_each)]

        #print("Function combine_model_features exits")
        return common_feat + mrmr_incr_feat_1_other + mrmr_incr_feat_2_other

    def mrmr_classic(self, X:pd.DataFrame, y:pd.Series, y_type, mrmr_feat_count, mode):
        # make sure mrmr_feature_count is not greater than the entire feature count
        feat_cnt = min(X.shape[1], mrmr_feat_count)

        # mrmr_classic should not be called with feat_cnt <= 1
        if feat_cnt <= 1:
            raise Exception("Implementation error in function mrmr_classic")
    
        feature_types = X.dtypes
        feature_types_counts = X.dtypes.apply(lambda dtype: dtype.name).value_counts().to_dict()
        #print('Running mRMR.classic with response type', str(y_type), ", features of type", str(feature_types_counts))
        #print('Casting ' + y.name + ' from type ' + y.dtype.name + ' to type ' + str(y_type))
        y_casted = cast_series_type(y, y_type); #print('y\n', y, '\nresp_casted\n', y_casted)
        # TODO !!!!!!!!!!!!!!!! need to use y_casted instead of y in next line
        mrmr_res, mrmr_scores_df = self._instMrmr.smlp_mrmr(X, y, feat_cnt)
        #mrmr_res, mrmr_scores_df = self._instMrmr.smlp_mrmr(X, y_casted, feat_cnt)
        #print('mrmr_res\n', mrmr_res); print('mrmr_res type', str(type(mrmr_res)), 'length', len(mrmr_res))
        return mrmr_res, mrmr_scores_df
    
    
    # This functions runs smlp_mrmr() separately on categorical features and on numeric features, then combines the results.
    def mrmr_split_combine(self, resp:pd.Series, resp_type:str, labeled_features_mrmr:pd.DataFrame, mrmr_feat_count:int, 
            mode:str, discretize_num_feat:bool, for_prediction=False):
        # number of features to select and rank
        feat_cnt = min(labeled_features_mrmr.shape[1], mrmr_feat_count);
        if feat_cnt <= 1:
            if feat_cnt == 0:
                self._corr_logger.info("Skipping MRMR-classic feature selection since mrmr_feat_count == 0; performing importance ranking.")
            else:
                self._corr_logger.info("Skipping MRMR-classic feature selection as there are only " + str(feat_cnt) + 
       " features to select from")
                mrmr_incr_feat_num = mrmr_tbl_num = None
                mrmr_incr_feat_ord = mrmr_tbl_ord = None
        else:
            cond_adapt_types = resp.nunique() <= 2

            if cond_adapt_types:
                # since here we have binary response we can decide to treat it as numeric or categorical based
                # on the features' types: (1) if all numeric features are binary, turn them into catgorical and
                # join with categorical features, and treat the response as categorical; (2) if all categorical 
                # features are binary, turn them into numeric, join to numeric features and treat response as 
                # numeric. (3) Otherwise, run MRMR with response as numeric on all numeric features, and also run
                # MRMR with response as categorocal on all categorical features and binary numeric features, then
                # combine (take a union or interleave to select the required number of features) 
                mrmr_incr_feat_num = mrmr_incr_feat_ord = []
                mrmr_tbl_num = mrmr_tbl_ord = None
                num_labeled_features_mrmr, ctg_labeled_features_mrmr = pd_df_split_numeric_categorical(labeled_features_mrmr)

                bin_num_ind = [pd_series_is_binary_numeric(num_labeled_features_mrmr[col]) for col in num_labeled_features_mrmr.columns.tolist()]
                if ctg_labeled_features_mrmr.shape[1] > 0 and num_labeled_features_mrmr.shape[1] > 0:
                    if all(bin_num_ind):
                        #print("all num features are binary -- join them to ctg_labeled_features_mrmr")
                        ord_labeled_features_mrmr = pd_df_convert_numeric_to_categorical(pd_series_is_binary_numeric)
                        ctg_labeled_features_mrmr = pc.concat([ord_labeled_features_mrmr, ctg_labeled_features_mrmr], axis=1)
                        del ord_labeled_features_mrmr
                        num_labeled_features_mrmr = num_labeled_features_mrmr.drop([num_labeled_features_mrmr.columns.tolist()])
                    else:
                        #print("NOT all num features are binary -- join them to ctg_labeled_features_mrmr")
                        if all([pd_series_is_binary_categorical(ctg_labeled_features_mrmr[col]) for col in ctg_labeled_features_mrmr.columns.tolist()]):
                            #print("all ctg features are binary -- join them to num_labeled_features_mrmr")
                            # all ctg features are binary -- join them to num_labeled_features_mrmr
                            num_ctg_labeled_features_mrmr = pd_df_convert_numeric_to_categorical(ctg_labeled_features_mrmr)
                            num_labeled_features_mrmr = pd.concat([num_labeled_features_mrmr, num_ctg_labeled_features_mrmr], axis=1)
                            del num_ctg_labeled_features_mrmr
                            ctg_labeled_features_mrmr = ctg_labeled_features_mrmr[[]]
                        elif any(bin_num_ind):
                            #print("join to ctg features also binary numeric ones")
                            # join to ctg features also binary numeric ones; otherwise original categorical
                            # features are selected even if their levels have been selected to and are more
                            # important than the original categorical features
                            num_bin_labeled_features_mrmr = num_labeled_features_mrmr[bin_num_ind];
                            #num_bin_labeled_features_mrmr = as.data.frame(apply(num_bin_labeled_features_mrmr, 2, as.factor), stringsAsFactors=T); 
                            # num_bin_labeled_features_mrmr = pd_df_convert_numeric_to_categorical(pd_series_is_binary_numeric)
                            # num_bin_labeled_features_mrmr = 
                            #         colwise(function (col) convert_col_to_mrmr_type(col)$column)(num_bin_labeled_features_mrmr); 
                            # num_bin_labeled_features_mrmr = as.data.frame(num_bin_labeled_features_mrmr, stringsAsFactors=T);
                            ctg_labeled_features_mrmr = pd.concat([num_bin_labeled_features_mrmr, ctg_labeled_features_mrmr], axis=1)
                            del num_bin_labeled_features_mrmr
                #print('num_labeled_features_mrmr\n', num_labeled_features_mrmr);
                #print('ctg_labeled_features_mrmr\n', ctg_labeled_features_mrmr)

                if num_labeled_features_mrmr.shape[1] > 1:
                    #print("run mrmr_classic on mumetic features")
                    mrmr_incr_feat_num, mrmr_tbl_num = self.mrmr_classic(num_labeled_features_mrmr, resp, "numeric", 
                           feat_cnt, mode)
                    if for_prediction:
                        self._corr_logger.info(mrmr_tbl_num)
                elif num_labeled_features_mrmr.shape[1] == 1:
                    self._corr_logger.info("Skipping MRMR-classic feature selection for numeric features as there is only one feature to select from")
                    mrmr_incr_feat_num = num_labeled_features_mrmr.columns.tolist()
                del num_labeled_features_mrmr 

                if ctg_labeled_features_mrmr.shape[1] > 1:
                    #print("run mrmr_classic on categorical features")
                    mrmr_incr_feat_ord, mrmr_tbl_ord = self.mrmr_classic(ctg_labeled_features_mrmr, resp, "ordered", 
                           feat_cnt, mode)
                    if for_prediction:
                        self._corr_logger.info(mrmr_tbl_ord)
                elif len(ctg_labeled_features_mrmr) == 1:
                    self._corr_logger.info("Skipping MRMR-classic feature selection for categorical features as there is only one feature to select from")
                    mrmr_incr_feat_ord = colnames(ctg_labeled_features_mrmr)
                del ctg_labeled_features_mrmr
            else: # not cond_adapt_types -- response has more than two values
                # response as numeric 
                # TO DO: should not run response as numeric if the response is factor?
                # responses with three character values are not supported in prediction but are supported in features and levels modes.
                # TO DO: should we add condition that the response is not factor and also that mode should not be features?
                # No need now as factor responses are not supported for training and prediction modes and the flow should not reach here
                response_as_numeric_cond = not discretize_num_feat

                if response_as_numeric_cond:
                    mrmr_incr_feat_num, mrmr_tbl_num = self.mrmr_classic(labeled_features_mrmr, resp, "numeric", feat_cnt, mode)
                    if for_prediction:
                        self._corr_logger.info(mrmr_tbl_num)
                else:
                    mrmr_incr_feat_num = []
                    mrmr_tbl_num = None

                # response as ordered
                if discretize_num_feat:
                    response_as_ordered_cond = True
                elif pd_series_is_numeric(resp):
                    response_as_ordered_cond = False
                elif for_prediction:
                    raise Exception("Implementation error with mrmr_response_as_ordered_pred")
                else:
                   # we get here for 3-valued categorical responses without feature discretization (e.g., 322-323); 
                   # could be be feature selection or trainingprediction modes. Cannot be levels mode as it passes
                   # to feature selection a binary response even if the original response has three values good, bad, not-sure
                   response_as_ordered_cond = mrmr_resp_as_ord #mrmr_response_as_ordered
                #print('response_as_ordered_cond', response_as_ordered_cond)

                if response_as_ordered_cond:
                    mrmr_incr_feat_ord, mrmr_tbl_ord = self.mrmr_classic(labeled_features_mrmr, resp, "ordered",
                        feat_cnt, mode)
                    if for_prediction:
                        self._corr_logger.info(mrmr_tbl_ord)
                else:
                    mrmr_incr_feat_ord = []
                    mrmr_tbl_ord = None

        mrmr_incr_feat = self.combine_model_features(mrmr_incr_feat_num, mrmr_incr_feat_ord, feat_cnt)
        mrmr_tbl_incr_feat = {'mrmr_incr_feat_num': mrmr_incr_feat_num, 'mrmr_incr_feat_ord': mrmr_incr_feat_ord,
            'mrmr_tbl_num': mrmr_tbl_num, 'mrmr_tbl_ord': mrmr_tbl_ord}; #print(mrmr_tbl_incr_feat)
        
        #if mode in ['train', 'predict']:
        #    if mrmr_incr_feat is None:
        #        mrmr_incr_feat = labeled_features_mrmr.columns.tolist()
        #if for_prediction:
        #    return fs_summary_df, mrmr_incr_feat
        return mrmr_tbl_incr_feat, mrmr_incr_feat
    
    # TODO !!! move function mrmr_split_combine() to smlp_mermr module, say for usage instead/alomg with smlp_mrmr()
    # This function computes fs_summary_df -- correlations summary netween features feat_df and response resp
    def correlate_filter_methods(self, feat_df:pd.DataFrame, resp:pd.Series, resp_type, continuous_estimators, mrmr_feat_count, 
            fs_summary_df, mode, for_prediction, discretization_algo:str, discretization_bins:int, discretization_labels:bool, discretization_type:str, 
            discretize_numeric_features:bool, mutual_information_method:str, corr_and_mi:bool, non_range_feat):
        #print("Function correlate_filter_methods starts")
        mrmr_selection = True # TODO !!!! need command line option
        
        # TODO !!! drop resp_type argument and define it here rather than passing from ensemble_features_single_response()
        assert (resp_type == 'numeric') == pd_series_is_numeric(resp)
        assert (resp_type == 'factor')  == (pd_series_is_object(resp) or pd_series_is_category(resp))
        assert (resp_type == 'ordered') == pd_series_is_ordered_category(resp)

        if not (mrmr_selection or corr_and_mi):
            #print("\nSkipping mRMRe feature selection\n")
            if mode in [PREDICTION, TRAINING, NOVELTY]:  # novelty_fix
                mrmr_incr_feat = feat_df.columns.tolist()
            else:
                mrmr_incr_feat = None
            #print("Function correlate_filter_methods exits")
            return fs_summary_df, mrmr_incr_feat

        # feature selection is meaningless for a single value response
        if resp.nunique() == 1:
            raise Exception("Single value response was passed to correlate_filter_methods")

        orig_feat_types = feat_df.dtypes; #print('orig_feat_types\n', orig_feat_types)
        #labeled_features_mrmr = feat_df # TODO !!!! changed the implementation
        #del feat_df # TODO !!! does not make sense to introduce labeled_features_mrmr, will just use feat_df
        
        if discretize_numeric_features is None:  # Corresponds to its (default) command line value 'auto'.
            # We want to discretize numeric features when the response is categorical, or is a binary
            # 0/1 response, which corresponds to classification problems, and this response might be
            # result of pre-processing catgorical responses into one or more 0/1 responses.
            # Please refer to description of data pre-processing and processing steps in smlp_data.py
            # module, within definition of class SmlpData. From that description it is also clear that
            # after data preparation steps (data pre-processing nad processing), a response cannot
            # be categical -- it can only be numeric or binary 0/1 response. Hence the sanity chekcs
            # (assertions / exceptions) below.
            discretize_num_feat = pd_series_is_binary_int(resp) or pd_series_is_categorical(resp)
            # reponses here are as they have resulted after data preparation steps -- not (yet) discretized of  
            # casted to other types for the sake of improving accuracy of correlations computed in this module.
            assert not pd_series_is_categorical(resp)
            if not discretize_num_feat and not pd_series_is_numeric(resp):
                raise Exception("Implementation error with response type in correlate_filter_methods")
        else:
            discretize_num_feat = discretize_numeric_features

        expand_factors_into_level_features = False # TODO !!! this is related to range analysis
        mrmr_response_as_ordered = True # command line option in previous implementation
        if discretize_num_feat or expand_factors_into_level_features:
            mrmr_resp_as_ord = False
        else:
            mrmr_resp_as_ord = mrmr_response_as_ordered

        # Discretize numeric features if needed
        if discretize_num_feat:
            self._corr_logger.info('Discretizing numeric features for feature selection using method ' + str(discretization_algo))
            #print("Treating the response as categorical for MRMR selection")
            #print(discretization_algo, discretization_bins, discretization_labels, discretization_type); 
            feat_df = self._instDiscret.smlp_discretize_df(feat_df, algo=discretization_algo, 
                bins=discretization_bins, labels=discretization_labels, result_type=discretization_type)

        # Determine feature types
        # TODO !!! just use compute_feature_supertype(), no need to compute feat_type twice
        is_numeric_feat = all(feat_df.apply(lambda col: pd_series_is_numeric(col)))
        is_factor_feat = all(feat_df.apply(lambda col: pd_series_is_categorical(col)))
        #print('is_factor_feat', is_factor_feat); print(feat_df)
        if is_factor_feat:
            is_factor_feat = all(feat_df.apply(lambda col: not pd_series_is_ordered_category(col)))
            is_ordered_feat = all(feat_df.apply(lambda col: pd_series_is_ordered_category(col)))
        else:
            is_factor_feat = is_ordered_feat = False
        feat_type = "numeric" if is_numeric_feat else "ordered" if is_ordered_feat else "factor" if is_factor_feat else "mixed"
        
        if feat_type == "integer":
            raise Exception("Implementation error in function correlate_filter_methods")
        assert feat_type == self.compute_feature_supertype(feat_df)
        factor_tp = "ordered" #TODO !!! not used currently


        mrmr_tbl_incr_feat, mrmr_incr_feat = self.mrmr_split_combine(resp, resp_type, feat_df, mrmr_feat_count, 
            mode, discretize_num_feat, for_prediction=False)
        # for factor response, pearson, spearman, kendall and freaquency estimators work the same ???
        estimators = ["pearson"]
        
        if pd_series_is_categorical(resp):
            # We have three options here
            if feat_type == "numeric":
                #print("case feat_type == numeric")
                # TODO !!! not clear why we need to convert the columns to numeric as they should alsready be numeric?
                #fs_summary_df = self.mrmre_resp_feat_corr_multi_estimators(colwise(as.numeric)(feat_df), 
                #    resp, resp_name, "numeric", orig_feat_types, estimators, mutual_information_method, discretize_num_feat, fs_summary_df, mrmr_tbl_incr_feat, mode, output_file)
                fs_summary_df = self.mrmre_resp_feat_corr_multi_estimators(feat_df, resp, 
                    "numeric", orig_feat_types, estimators, mutual_information_method, corr_and_mi,
                    discretization_algo, discretization_bins, discretization_labels, discretization_type,
                    discretize_num_feat, fs_summary_df, mrmr_tbl_incr_feat, mode)
            elif feat_type == "factor" or feat_type == "ordered" or feat_type == "mixed":
                #print("case feat_type in [factor, ordered, mixed], sub-case with numeric")
                fs_summary_df = self.mrmre_resp_feat_corr_multi_estimators(feat_df, resp, 
                    "numeric", orig_feat_types, estimators, mutual_information_method, corr_and_mi,
                    discretization_algo, discretization_bins, discretization_labels, discretization_type,                
                    discretize_num_feat, fs_summary_df, mrmr_tbl_incr_feat, mode)
                
                if mrmr_resp_as_ord:
                    #print('sub-case with ordered')
                    fs_summary_df = self.mrmre_resp_feat_corr_multi_estimators(feat_df, resp, 
                        "ordered", orig_feat_types, estimators, mutual_information_method, corr_and_mi, 
                        discretization_algo, discretization_bins, discretization_labels, discretization_type,
                        discretize_num_feat, fs_summary_df, mrmr_tbl_incr_feat, mode)

            # now run with response as factor (need to be ordered)
            # in the past, here we used if-condition (feat_type == "numeric" | feat_type == factor_tp) # feat_type == "factor"
            # this change fixes regression test 323
            if feat_type == "numeric":
                #print("case (feat_type == numeric | feat_type == factor)")
                if mrmr_resp_as_ord:
                    fs_summary_df = self.mrmre_resp_feat_corr_multi_estimators(feat_df, resp, 
                        "ordered", orig_feat_types, continuous_estimators, mutual_information_method, corr_and_mi, 
                        discretization_algo, discretization_bins, discretization_labels, discretization_type,          
                        discretize_num_feat, fs_summary_df, mrmr_tbl_incr_feat, mode)
        elif pd_series_is_numeric(resp) and not pd_series_is_int(resp): #resp_type == 'numeric': 
            #print("case (resp_type == numeric) and not int")
            fs_summary_df = self.mrmre_resp_feat_corr_multi_estimators(feat_df, resp,
                "numeric", orig_feat_types, continuous_estimators, mutual_information_method, corr_and_mi, 
                discretization_algo, discretization_bins, discretization_labels, discretization_type,                                                      
                discretize_num_feat, fs_summary_df, mrmr_tbl_incr_feat, mode); #print(fs_summary_df)
        elif pd_series_is_int(resp): #resp_type == 'integer': 
            #print("case !!!(resp_type == integer)");
            fs_summary_df = self.mrmre_resp_feat_corr_multi_estimators(feat_df, resp, 
                "numeric", orig_feat_types, continuous_estimators, mutual_information_method, corr_and_mi, 
                discretization_algo, discretization_bins, discretization_labels, discretization_type,                                                       
                discretize_num_feat, fs_summary_df, mrmr_tbl_incr_feat, mode)
        else:
            raise Exception("Unsupported response type " + str(resp_type) + "in feature selection.")
        
        #print('fs_summary_df\n', fs_summary_df); print('mrmr_incr_feat\n', mrmr_incr_feat)
        #print("Function correlate_filter_methods exits")
        return fs_summary_df, mrmr_incr_feat

    
    def drop_identical_features(df, log, with_colnames, check_colnames):
        if df is None or df.shape[1] < 2 or df.shape[0] < 2:
            return df

        colnms = df.columns
        indices_to_remove = []

        for i in range(1, df.shape[1]):
            for j in range(i):
                ith = df.iloc[:, i]
                jth = df.iloc[:, j]
                if ith.equals(jth):
                    if not with_colnames or colnms[i] == colnms[j]:
                        if duplicated_columns_fix:
                            colnm_i = colnms[i]
                            colnm_j = colnms[j]
                            to_drop = i if not (colnm_i.endswith('.x') or colnm_i.endswith('.y')) or \
                                             (colnm_j.endswith('.x') or colnm_j.endswith('.y')) else j
                            indices_to_remove.append(to_drop)
                        else:
                            indices_to_remove.append(i)
                elif check_colnames and colnms[i] == colnms[j]:
                    raise Exception("Implementation error in function drop_identical_features")

        unique_indices_to_remove = list(set(indices_to_remove))
        if unique_indices_to_remove:
            if log:
                self._corr_logger.info(f"Dropping {len(unique_indices_to_remove)} features identical to another feature\n")
                dropped_columns = [colnms[i] for i in unique_indices_to_remove]
                self._corr_logger.info('\t' + '\n\t'.join(dropped_columns))
            df = df.drop(columns=df.columns[unique_indices_to_remove])

        return df


    def fs_ensemble_summary(self, fs_summary_df, drop_identical, log):
        scale_for_ranking = True # TODO !!! hard coded
        ranking_na_fix = False # for ensemble ranking, ignore NA in mean value computation for rows
        do_not_normalize = False # compute ensenble scores and ranking in tex file w/o normalizing
        sort_ties = True # make sure sorting df works properly;
        CORRELATION_DECIMAL_PRECISION = 15
        
        if fs_summary_df is None:
            return fs_summary_df

        def scale_element(e, abs_max, abs_min, abs_rng, keep_sign):
            abs_e = abs_max if np.isinf(e) else 0 if pd.isna(e) else abs(e)
            if scale_for_ranking:
                r = abs_e / abs_max
            elif abs_rng == 0:
                r = abs_e
            else:
                r = (abs_e - abs_min) / abs_rng
            return -r if e < 0 and keep_sign else r
            if drop_identical:
                fs_summary_df = drop_identical_features(fs_summary_df, False, False, False)
        
        def scale01sign(col, keep_sign):
            #print('scale01sign: col\n', col)
            if col.dtype not in ['int64', 'float64']: # TODO -- more types like int16
                return col
            col = col.fillna(0)
            col_abs = col.abs()
            abs_min = col_abs.min()
            abs_max = col_abs.max()
            if abs_max == 0:
                return col_abs
            if np.isinf(abs_max):
                col_non_inf_vals = col_abs[~np.isinf(col_abs)]
                fin_max = 1 if col_non_inf_vals.empty else col_non_inf_vals.max()
                abs_max = 10 * fin_max
            abs_rng = abs_max - abs_min
            col_scaled = col.apply(lambda e: scale_element(e, abs_max, abs_min, abs_rng, keep_sign))
            return col_scaled

        if not ranking_na_fix:
            fs_summary_df = fs_summary_df.fillna(0)
        else:
            na_saved = fs_summary_df.isna()

        if not do_not_normalize:
            for colnm in fs_summary_df.columns:
                fs_summary_df[colnm] = scale01sign(fs_summary_df[colnm], False)

        if not pd_df_is_empty(fs_summary_df):
            if fs_summary_df.shape[1] > 1:
                if ranking_na_fix:
                    fs_summary_df[na_saved] = np.nan
                    ranking = fs_summary_df.iloc[:, 1:].mean(axis=1, skipna=True)
                else:
                    ranking = fs_summary_df.iloc[:, 1:].mean(axis=1, skipna=False)
                fs_summary_df.insert(1, 'ranking', ranking)
            else:
                ranking = np.ones(fs_summary_df.shape[0])
                fs_summary_df.insert(1, 'ranking', ranking)

            if sort_ties:
                fs_summary_df = fs_summary_df.sort_values(by='ranking', ascending=False, kind='mergesort')
            else:
                fs_summary_df = fs_summary_df.sort_values(by='ranking', ascending=False)

        if log:
            self._corr_logger.info('Ensemble feature selection (with all scores individually scaled to [0.1]):\n'+str(fs_summary_df))

        return fs_summary_df

    
    def ensemble_features_single_response(self, X:pd.DataFrame, y:pd.Series, feat_names:list[str], feat_names_dict:dict, mode:str,
            discretization_algo:str, discretization_bins:int, discretization_labels:bool, discretization_type:str, 
            discretize_numeric_features:bool, mutual_information_method:str, corr_and_mi:bool, mrmr_feat_count:int, 
            continuous_estimators:list[str]):
        for_prediction = False # TODO !!!
        non_range_feat = None # TODO !!!
        
        if pd_series_is_numeric(y):
            y_type = 'numeric'
        elif pd_series_is_object(y) or pd_series_is_category(y):
            y_type = 'factor'
        elif pd_series_is_ordered_category(y):
             y_type = 'ordered'
        else:
            raise Exception('Unexpected response type in ensemble_features_single_response')
                
        fs_summary_df = pd.DataFrame({'important_features': X.columns.tolist()})
        fs_summary_df, mrmr_incr_feat = self.correlate_filter_methods(X, y, y_type, continuous_estimators, mrmr_feat_count, 
            fs_summary_df, mode, for_prediction, discretization_algo, discretization_bins, discretization_labels, discretization_type, 
            discretize_numeric_features, mutual_information_method, corr_and_mi, non_range_feat)
        
        # gard coded: drop identical columns in ensemble summary tables in function fs_ensemble_summary
        fs_summary_df = self.fs_ensemble_summary(fs_summary_df, False, True)

        return fs_summary_df, mrmr_incr_feat
        
        
    # generates feature ranking fro individual responses and then combines into joint ranking of features based on the 
    # mean ranking score of ranking scores fro individual responses (per features). The jpint ramking is reported into
    # file with suffix "_features_summary.csv".
    def smlp_correlate(self, X:pd.DataFrame, y:pd.DataFrame, feat_names:list[str], resp_names:list[str], feat_names_dict:dict, 
            discretization_algo:str, discretization_bins, discretization_labels, discretization_type, 
            discretize_numeric_features, continuous_estimators, mutual_information_method, corr_and_mi:bool, mrmr_feat_count:int):
        mode = "features" # TODO !!!
        
        assert set(feat_names) == set(X.columns.tolist())
        assert set(resp_names) == set(y.columns.tolist())
        
        fs_summary_df = pd.DataFrame({'important_features': X.columns.tolist()})
        resp_ranking_cols = []                              
        for resp_name in y.columns.tolist():
            fs_summary_df_curr, mrmr_incr_feat_curr = self.ensemble_features_single_response(X, y[resp_name], feat_names, feat_names_dict, mode,
                discretization_algo, discretization_bins, discretization_labels, discretization_type, discretize_numeric_features, mutual_information_method,
                corr_and_mi, mrmr_feat_count, continuous_estimators)
            #fs_summary_df_curr.to_csv(self.report_file_prefix + '_' + resp_name + '_features_summary.csv', index=False)
            for col in fs_summary_df_curr.columns:
                if col != 'important_features':
                    fs_summary_df_curr.rename(columns={col: col + '_' + resp_name}, inplace=True)
                    if col.startswith('ranking'):
                        resp_ranking_cols.append(col + '_' + resp_name)
            fs_summary_df =  pd.merge(fs_summary_df, fs_summary_df_curr, on="important_features", how="outer")
        
        # Calculate the mean of the specified columns
        fs_summary_df.insert(1, 'ranking', fs_summary_df[resp_ranking_cols].mean(axis=1))
        # Sort the DataFrame in descending order with respect to the new column
        fs_summary_df = fs_summary_df.sort_values(by='ranking', ascending=False)
        
        fs_summary_df.to_csv(self.report_file_prefix + '_features_summary.csv', index=False)
        self._corr_logger.info('Ensemble feature selection for multiple responses (with all scores individually scaled to [0.1]):\n'+str(fs_summary_df))
                
        
                