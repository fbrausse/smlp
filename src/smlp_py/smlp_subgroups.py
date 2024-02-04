import pysubgroup as ps
import pandas as pd
import numpy as np

from smlp_py.smlp_utils import (pd_df_col_is_numeric, pd_df_col_is_categorical, list_unique_unordered, 
    list_unique_ordered, param_dict_with_algo_name, get_response_type, rows_dict_to_df)
from smlp_py.smlp_precisions import PrecisionMeasures
from smlp_py.smlp_constants import *
try:
    from smlp_py.range_plots import RangePlots
    RangePlots_are_missing = False
except ImportError:
    # Subgroup discovery can run but plots (visualization) of the slected ranges must be disabled
    RangePlots_are_missing = True


class SubgroupDiscovery:
    def __init__(self):
        self._psg_logger = None
        self._psg_positive_value = 'fail'
        self._psg_negative_value = 'pass'
        
        # type of feature from which a single range feature is built
        self._ORIGIN_NUM = "numeric"
        self._ORIGIN_CTG = "factor"
        self._ORIGIN_BIN = "binary"
        
        # columns in feature range ranking report fs_ranking_df
        self._MAX_BINS_1 = 'max_bins_1'
        self._MAX_BINS_2 = 'max_bins_2'
        self._MAX_BINS_3 = 'max_bins_3'
        
        self._FEATURE_1 = 'feature_1'
        self._FEATURE_2 = 'feature_2'
        self._FEATURE_3 = 'feature_3'
        
        self._RANGE_1 = 'range_1'
        self._RANGE_2 = 'range_2'
        self._RANGE_3 = 'range_3'
        
        self._BINS_1 = 'bins_1'
        self._BINS_2 = 'bins_2'
        self._BINS_3 = 'bins_3'
        
        # separator to concatinate feature name with level name to produce a unique reference name
        self._FEAT_LEVEL_SEP = '__' 
        self._RANGES_SEP = '__' 
        
        # 
        self.MAX_DIMENSION = 3
        self.TOP_RANKED = 15
        
        if not RangePlots_are_missing:
            self.instRangePlots = RangePlots()
        
        self.instPrecisions = PrecisionMeasures()
        self.STAT_NEGATIVE_VALUE = STAT_NEGATIVE_VALUE #int(0)
        self.STAT_POSITIVE_VALUE = STAT_POSITIVE_VALUE #int(1)
        
        # report file names
        #self.model_levels_dict_file = self.model_file_prefix + '_model_levels_dict.json'
        
        
        # subgroup search hyper parameters
        self._subgroup_params_dict = {
            'quality_target': {'abbr':'quality', 'default':LIFT, 'type':str,
                'help':'Quality function (quality target/measure) used for defining the ' +
                    'importance criterion for range selction. The supported options ' +
                    '(both for numeric as well as binary responses) are {} and {} ' +
                    '[default {}]'.format(LIFT, WR_ACC, LIFT)},
            'max_dimension' : {'abbr':'dim', 'default':self.MAX_DIMENSION, 'type':int,
                'help':'Maximal dimension of selected range tuples (feature-range tuples) ' +
                    '[default {}]'.format(self.MAX_DIMENSION)},
            'top_ranked' : {'abbr':'top', 'default':self.TOP_RANKED, 'type':int,
                'help':'Required count of selected range tuples (feature-range tuples) ' +
                    '[default {}]'.format(self.TOP_RANKED)}
        }
    
    # set logger from a caller script
    def set_logger(self, logger):
        self._psg_logger = logger 
    
    def set_report_file_prefix(self, report_file_prefix):
        self.report_file_prefix = report_file_prefix
    
    # report file and plot dir names
    def get_features_ranking_file(self):
        return self.report_file_prefix + '_features_ranking.csv'
    def get_ranking_resp_feat_file(self):
        return self.report_file_prefix + '_ranking_resp_feat.csv'
    def get_range_plots_dir(self):
        return self.report_file_prefix + '_plots'
    
    # Getter function to pass subgroup params dictionary to a caller function, after modifying 
    # the default params dictionary self.subgroup_params_dict by adding prefix 'psg_' to the 
    # names of full and abbriviated option names in self._subgroup_params_dict.
    def get_subgroup_hparam_default_dict(self):
        subgroup_hyperparam_dict = param_dict_with_algo_name(self._subgroup_params_dict, 'psg')
        return subgroup_hyperparam_dict
    
    # generate a unique neme for a level in a categorical feature, to serve as the name of 
    # a correponding range feature
    def level_to_unique_level(self, feat, level):
        return self._FEAT_LEVEL_SEP.join([feat,level])
    
    # given a unique name feat_level generated from a level in a categorical feature using function 
    # level_to_unique_level, return the level name (this is the same as dropping from feat_level an 
    # initial prefic consisting of the feature name concatenated with self._FEAT_LEVEL_SEP = '__'
    def unique_level_to_level(self, feat_level, ranges_map):
        return ranges_map[feat_level]['lo']
    
    # extract the name of the level of a categorical featre from a correponding range tuple 
    # (this level name is just one of the feilds of the range tuple). This function is a version
    # of unique_level_to_level() which extracts ramge_tuple from ranges_map and then accesses 
    # the level name just like unique_level_to_level()
    def unique_level_to_level(self, range_tuple):
        return range_tuple['lo']
    
    def isSupportedQF(self, bqf):
        return(bqf in [WR_ACC, LIFT])

    def evaluateSupportedQF(self, qf, isBoolean):
        #print(qf); print(isBoolean)
        if not self.isSupportedQF(qf):
            raise BaseException("quality function specified correctly; the supported options are WRAcc and Lift")
        if isBoolean:
            if qf==WR_ACC:
                return(ps.WRAccQF()) # same as ps.WRAccQF(1) ?
            elif qf==LIFT:
                return(ps.LiftQF()) # no need for arg 0
            else:
                raise BaseException('the specified quality function is currently not supported')
        else:
            if qf==WR_ACC:
                    return(ps.StandardQFNumeric(1))
            elif qf==LIFT:
                    return(ps.StandardQFNumeric(0))
            else:
                raise BaseException('the specified quality function is currently not supported')

    # Generates name of a single range features from feilds of a range tuple (which is s structure
    # (dictionery) containing information on how the single range features was produced from an o
    # riginal feature in the data. Say a range tuple cal look like
    # {'name': 'p3', 'lo': 8.0, 'hi': 8.0, 'binLo': 0, 'binHi': 0, 'origin': 'numeric'} and the 
    # name of the single range feature produced for it by this function is rng_nm p3_8.0_8.0_Bin_0.
    # Caution !!!: lo and hi values are rounded to four decimal degits (hard coded) for readabolity
    # of the names, thus they do not represnt the precise bounds of an envoleved range.
    def range_tuple_to_range_name(self, feat_name, range_lo, range_hi, bin_lo, bin_hi):
        decimal_zeros_to_round = 4
        range_lo = round(range_lo, 4)
        range_hi = round(range_hi, 4)
        range_lo_str = str(range_lo) if range_lo >= 0 else 'minus'+str(range_lo)
        range_hi_str = str(range_hi) if range_hi >= 0 else 'minus'+str(range_hi)
        bin_str = 'Bin_'+str(bin_lo) if bin_lo == bin_hi else '_'.join(['Bin', str(bin_lo), str(bin_hi)])
        return '_'.join([feat_name, range_lo_str, range_hi_str, bin_str])
        
    # Converts pysubgroup (psg) representation of a single range feature (a numeric feature and a range
    # or a categorical feature and a level) into smlp representation of the same single range feature.
    # argument col_type in this function is the type of the correponding feature in the input data, and
    # it does not necessarily correponds to the type of the original feature as computed by pysubgroup --
    # in particular, pysubgroup treats integer features as "nominal" (same or similar to "unordered 
    # categorical" feature) while in smlp representation the feature type is specified as "numeric".
    # The function returns the smlp representation of the range tuple and name associated to it (name of
    # the single range feature correponding to this range tuple).
    def _psg_range_tuple_to_range_tuple(self, psg_range_tuple, col_type):
        feat_name = psg_range_tuple[0] #print('feat_name', feat_name); 

        #col = labeled_features[[feat_name]]; #print('col', col);
        if col_type == self._ORIGIN_NUM:
            range_lo = float(psg_range_tuple[1]); #print('range_lo', range_lo); 
            range_hi = float(psg_range_tuple[2]) if psg_range_tuple[2] != 'dummy_psg' else \
                float(psg_range_tuple[1]); #print('range_hi', range_hi); 
            bin_hi = bin_lo = int(0) # TODO we do not have bins info from pysubgroup
            range_name = self.range_tuple_to_range_name(feat_name, range_lo, range_hi, bin_lo, bin_hi)
        elif col_type == self._ORIGIN_CTG:
            range_lo = psg_range_tuple[1]
            range_hi = bin_lo = bin_hi = np.nan
            level_name = psg_range_tuple[1];
            range_name = self.level_to_unique_level(feat_name, level_name)
        else:
            raise Exception('Usupported column type ' + str(col_type) + ' in function ranges_map_csv2ranges_map')

        range_tuple = {'name':feat_name, 'lo':range_lo, 'hi':range_hi, 'binLo': bin_lo, 'binHi':bin_hi, 'origin':col_type}
        return range_name, range_tuple
    
    # Converts a set of pysubgroup style single range tuples into a map (dictionary) of smlp
    # representation of single range feature names as the keys and smlp representation of the
    # corresponding range tuples as the values -- uses self._psg_range_tuple_to_range_tuple().
    def _psg_ranges_to_single_ranges_map(self, ranges_map_csv, labeled_features):
        ranges_map = {}
        for c in ranges_map_csv.columns.tolist():
            psg_range_tuple = tuple(ranges_map_csv[c])
            feat_name = psg_range_tuple[0]
            if pd_df_col_is_numeric(labeled_features, feat_name):
                col_type = self._ORIGIN_NUM
            elif pd_df_col_is_categorical(labeled_features, feat_name):
                col_type = self._ORIGIN_CTG
            else:
                raise Exception('Usupported column type for column ' + str(c) + 
                                ' in function _psg_ranges_to_single_ranges_map')
            range_name, ramge_typle = self._psg_range_tuple_to_range_tuple(psg_range_tuple, col_type)
            
            ranges_map[range_name] = ramge_typle
        return ranges_map
    
    
    # Computes a single range feature from a numeric feature and its range (interval) as well as
    # from a categorical feature and its level. The range information (bounds / level) is
    # given by argument rng_tuple which is a tule of the form (described through examples):
    # numeric feature: single range feature -- p3_7.0_7.0_Bin_0, the correponding rage tuple -- 
    # {'name': 'p3', 'lo': 7.0, 'hi': 7.0, 'binLo': 0, 'binHi': 0, 'origin': 'numeric'}
    # categorical feature: single range feature -- WAFER__ABCD, the correponding range tuple --
    # {'name': 'WAFER', 'lo': 'ABCD', 'hi': nan, 'binLo': nan, 'binHi': nan, 'origin': 'factor'}
    def range_tuple_and_feature_to_binary_single_range(self, rng_tuple, ft):
        if rng_tuple['origin'] != self._ORIGIN_NUM and rng_tuple['origin'] != self._ORIGIN_BIN:
            assert rng_tuple['origin'] == self._ORIGIN_CTG
            rng_nm = self.unique_level_to_level(rng_tuple); #print(rng_nm)
            rng_ft = (ft == rng_nm).astype(int); #print('\nrng_ft\n', rng_ft)
        else:
            #ft = feat_df[rng_tuple['name']]; #print('ft\n', ft); print(rng_tuple['hi'], rng_tuple['lo'])
            #print(rng_tuple); print(rng_tuple$hi); print(rng_tuple$lo)
            if rng_tuple['hi'] < rng_tuple['lo']:
                #print("inverse range")
                raise Exception('Inverse ranges are currently not supported')  
                rng_ft = (ft <= rng_tuple['hi']) | (ft >= rng_tuple['lo']) 
            else:
                #print("regular range")
                if rng_tuple['origin'] == self._ORIGIN_BIN:
                    # rng_tuple$hi and rng_tuple$lo coincide
                    rng_ft = ft == rng_tuple['hi']
                else:
                    assert rng_tuple['origin'] == self._ORIGIN_NUM
                    rng_ft = ((ft <= rng_tuple['hi']) & (ft >= rng_tuple['lo'])).astype(int); #print('\nrng_ft\n', rng_ft)
        return rng_ft

    # compute range feature from a numeric feature and its range (interval)
    def feature_and_range_to_in_range(self, feat_df, ranges_vec:list, ranges_map:dict):
        if len(ranges_vec) == 0:
            return None
        lavels_dict = {}
        for rng in ranges_vec:
            rng_tuple = ranges_map[rng]; #print('\nrng\n', rng); print('\nrng_tuple\n', rng_tuple);
            ft = feat_df[rng_tuple['name']]; #print('ft\n', ft)
            lavels_dict[rng] = self.range_tuple_and_feature_to_binary_single_range(rng_tuple, ft)
        #print('lavels_dict', lavels_dict, '\n', pd.DataFrame.from_dict(lavels_dict));
        return pd.DataFrame.from_dict(lavels_dict)
    
    
    # Assume we are interested in finding ranges that explain (a) values of the response
    # in its high value range, or explain (b) values of the response in its low value range.
    # Then without loss of generality we can assume that the response is scaled to [0, 1] 
    # and in addition, in case (a) we can assume positive samples have values in the high
    # range and in case (b) we assume that positive samples have values in the low range.
    # Now, if the response is the scaled vector resp, also in case (b) positive samples
    # have values in the high range of response resp = 1-resp. So without loss of generality
    # we assume that the response variable has values between 0 and 1 and positive samples
    # are the ones that have values close to 1, and we are looking for ranges with positive 
    # samples, thus the mean value of samples in such ranges must be close to one, or at 
    # least the mean value of the response for samples in this range must be higher than 
    # the mean value of the response resp. Now, the mean value of response values in our 
    # range can be seen as the probablity of being positive for samples in this range.
    # If we could view each sample as positive or negative, to acheive same probability for
    # a sample within the range to be positive, we would need PosIn = mean_rng*AllIn samples.
    # Similarly, we can define the count of positive samples as PosAll = mean_all*All, and hence
    # we can define PosOut = PosAll-PosIn. Simmilarly, we can compute NegIn and NegOut as
    # PosIn and PosOut for response variable resp = 1 - resp. Finlly, with frequences 
    # PosIn, PosOut, NegIn and NegOut, all accuracy scores for binary responses generalize
    # to the case of numeric response resp.
    def feat_resp2num_freqs(self, feat_name, feat, resp, non_range_feat, pos_value_name): #neg_value_name
        if feat_name in non_range_feat:
            return {FALSE_NEGATIVE:0,TRUE_NEGATIVE:0, TRUE_POSITIVE:0, FALSE_POSITIVE:0}
        
        #print('feat\n', feat); print('resp\n', resp);
        
        # n denotes samples count within the range, N denotes the entire samples count
        in_samples = (feat == 1); #print('in_samples\n', in_samples)
        n = sum(in_samples); #print('in samples -- n', n)
        N = len(in_samples);  #print('all samples -- N', N)

        # The argument norm_resp denotes the response resp scaled to 0 to 1.
        # So values close to 0 in resp_norm are vealues close to min of resp
        # and values close 1 in resp_norm are values close to max in resp.
        # Below such a response will be denoted as resp01. Thus for resp01
        # this function will give high scores (such as WRAcc, PosInRatio, etc.)
        # to ranges where the resp_norm mean is the highest. 
        # Now, often we want to select ranges where the resp_norm mean is the lowest.
        # in paricular, we define that when pos_value_name is 0 and we are looking 
        # for ranges where the mean mean_rng is smaller than the avarage mean mean_all.
        # Thus in such cases we pass to function resp10 = 1 - resp01 as resp_norm.
        def range_norm_resp2means(norm_resp):
            mean_all = np.mean(norm_resp)
            mean_rng = np.mean(norm_resp[in_samples])
            PosIn = mean_rng*n
            PosAll = mean_all*N
            PosOut = PosAll - PosIn; #print(c(mean_all,mean_rng,PosIn,PosAll,PosOut,n,N))
            scores = {TRUE_POSITIVE:PosIn, FALSE_NEGATIVE:PosOut}
            return scores

        # Scale response to 0 to 1 segment -- the result is resp01
        # resp10 is its reflection, reversing the max and min
        resp_min = min(resp); #print('resp_min', resp_min, max(resp))
        resp01 = (resp - resp_min)/(max(resp)-resp_min); #pnv(resp); pnv(resp01)
        resp10 = 1 - resp01 
        if pos_value_name == self.STAT_NEGATIVE_VALUE:
            # we are looking for low values as problematic values
            freqs = range_norm_resp2means(resp10)
            freqs_inv = range_norm_resp2means(resp01)
        elif pos_value_name == self.STAT_POSITIVE_VALUE:
            # we are looking for high values as problematic values
            freqs = range_norm_resp2means(resp01)
            freqs_inv = range_norm_resp2means(resp10)
        else:
            raise Exception('Error: pos_value_name must be 0 or 1')
        #print('freqs_inv', freqs_inv)
        return {FALSE_NEGATIVE:freqs[FALSE_NEGATIVE],TRUE_NEGATIVE:freqs_inv[FALSE_NEGATIVE], 
                TRUE_POSITIVE:freqs[TRUE_POSITIVE], FALSE_POSITIVE:freqs_inv[TRUE_POSITIVE]}


    # For a range feature feat and response resp, this function computes multiple
    # scores associated with the range that are based on counts of positive and negative samples
    # withing the range and outside. These scores might include multiple PosInBalanced
    # scores computed based on respective rpa_target_condition conditions/ratios. The
    # function is called within ensemble_features_single_response on range features 
    # and then on range pairs in order to select target important ranges and pairs for 
    # further building range pairs and range triplets, respectively, as well as for
    # selecting the ranges and pairs that will be reported in the features ranking file.
    # The function is also called within ranked_fs_summary_to_fs_ranking just to compute 
    # positive vs negative sample statistics for ranges, pairs and triplets reported in features 
    # ranking file.
    def feat_resp2opt_scores(self, feat_name, feat, resp, non_range_feat, cls_reg_mode, 
            pos_min_freq_thresh, pos_value_name, neg_value_name):
        freqs_dict = self.feat_resp2num_freqs(feat_name, feat, resp, non_range_feat, pos_value_name) #neg_value_name
        PosOut = freqs_dict[FALSE_NEGATIVE]    # fale negative FN
        NegOut = freqs_dict[TRUE_NEGATIVE]  # true negative TN
        PosIn = freqs_dict[TRUE_POSITIVE]      # true positive  TP
        NegIn = freqs_dict[FALSE_POSITIVE]    # false positive FP
        AllIn = PosIn+NegIn        # predicted_negative PN
        AllOut = PosOut+NegOut     # predicted_positive PP
        PosAll = PosIn+PosOut       # response_negative RN
        NegAll = NegIn+NegOut    # response_positive RP
        All = AllIn + AllOut; #pnv(c(AllIn, AllOut, PosAll, NegAll, All)) 
        #vec = freqs_dict$vec
            
        # True Positive Rate, TPr
        PosInVsPosAll_full = self.instPrecisions.compute_true_positive_rate(PosIn, PosOut, True)  
        PosInVsPosAll = round(PosInVsPosAll_full, 4)
        # Predictive Positive Rate, PPr
        PosInVsAllIn_full = self.instPrecisions.compute_predictive_positive_rate(PosIn, AllIn, PosAll, True) 
        PosInVsAllIn = round(PosInVsAllIn_full, 4)

        precisions_dict = {}
        if WR_ACC in RANGE_ANALYSIS_PRECISIONS:
            WRAcc_full = self.instPrecisions.compute_weighted_relative_accuracy(PosOut, NegOut, PosIn, NegIn)
            WRAcc = round(WRAcc_full, 4)
            precisions_dict[WR_ACC] = WRAcc
            
        if ROC_ACC in RANGE_ANALYSIS_PRECISIONS:
            ROCAcc_full = self.instPrecisions.compute_roc_accuracy(NegOut, PosOut, PosIn, NegIn)
            ROCAcc = round(ROCAcc_full, 4)
            precisions_dict[ROC_ACC] = ROCAcc
            
        # sanity check that normalized versions of WRAcc and ROCAcc must both be greater than 0.5 or smaller than 0.5
        if (not feat_name in non_range_feat) and \
           (ROC_ACC in RANGE_ANALYSIS_PRECISIONS) and (WR_ACC in RANGE_ANALYSIS_PRECISIONS) and \
            (not np.isnan(WRAcc_full)) and (not np.isnan(ROCAcc_full)): # (not multi_layer_numeric_encoding) and
            normalize_roc_wracc = True
            if normalize_roc_wracc:
                if (np.sign(WRAcc_full-0.5) != np.sign(ROCAcc_full-0.5) and 
                     np.sign(WRAcc_full-0.5) != 0 and np.sign(ROCAcc_full-0.5) != 0):
                    #print(np.sign(WRAcc_full-0.5)); print(np.sign(ROCAcc_full-0.5));
                    #print(WRAcc_full); print(ROCAcc_full)
                    raise Exception('WRAcc and ROCAcc must both be greater than 0.5 or smaller than 0.5')
            elif np.sign(WRAcc_full) != np.sign(ROCAcc_full):
                #print(WRAcc_full); print(ROCAcc_full)
                raise Exception('WRAcc and ROCAcc must have the same sign')

        if COHEN_KAPPA in RANGE_ANALYSIS_PRECISIONS:
            accuracy = self.instPrecisions.compute_accuracy(PosIn, NegOut, All)
            # accuracy, predicted_negative, predicted_positive, true_negative, TP, samples_count)
            cohen_kappa_full = self.instPrecisions.compute_cohen_kappa(accuracy, AllIn, AllOut, PosIn, NegOut, All); 
            CohenKappa = round(cohen_kappa_full, 4)
            precisions_dict[COHEN_KAPPA] = CohenKappa
            
        if ACCURACY in RANGE_ANALYSIS_PRECISIONS:
            #pnv(c(PosIn, NegOut, All))
            accuracy_full = self.instPrecisions.compute_accuracy(PosIn, NegOut, All)
            Accuracy = round(accuracy_full, 4)
            precisions_dict[ACCURACY] = Accuracy
            
        if F1_SCORE in RANGE_ANALYSIS_PRECISIONS:
            F1Score_full = self.instPrecisions.compute_F1_from_precision_recall(PosInVsPosAll, PosInVsAllIn)
            F1Score = round(F1Score_full, 4)
            precisions_dict[F1_SCORE] = F1Score
            
        if LIFT in RANGE_ANALYSIS_PRECISIONS:
            PosInRatio_full = self.instPrecisions.compute_pos_in_ratio_from_ppr(PosInVsAllIn_full, PosAll, All)
            PosInRatio = round(PosInRatio_full, 4);
            precisions_dict[LIFT] = PosInRatio
            
        if NORM_POSITIVE_LR in RANGE_ANALYSIS_PRECISIONS: # or BALANCED_PRECISION in RANGE_ANALYSIS_PRECISIONS:
            NormPosLR_full = self.instPrecisions.compute_normalized_plr_from_tpr(PosInVsPosAll_full, NegIn, NegOut)
            NormPosLR = round(NormPosLR_full, 4)
            precisions_dict[NORM_POSITIVE_LR] = NormPosLR
            
        if BALANCED_PRECISION in RANGE_ANALYSIS_PRECISIONS:
            prec_weight = 0.5 # TODO expose as a parameter for user control
            PosInBalanced_full = self.instPrecisions.compute_balanced_precision(prec_weight, PosInVsPosAll, PosInVsAllIn) # NormPosLR
            PosInBalanced = round(PosInBalanced_full, 4)
            precisions_dict[BALANCED_PRECISION] = PosInBalanced
            if not NORM_POSITIVE_LR in RANGE_ANALYSIS_PRECISIONS:
                # NORM_POS_LR was added to precisions_dict only because it was needed for PosInBalanced_full, 
                # therefore, we now delete precisions_dict[NORM_POSITIVE_LR]
                del precisions_dict[NORM_POSITIVE_LR]
        
        if ENSEMBLE_ACC in RANGE_ANALYSIS_PRECISIONS:
            # we do not use Lift in ensemble since it is not a normalized score (not between 0 and 1)
            ensembele_precisions_dict = precisions_dict.copy()
            del ensembele_precisions_dict[LIFT]
            EnsAcc = np.mean(list(ensembele_precisions_dict.values()))
            EnsAcc = round(EnsAcc, 4);
            precisions_dict[ENSEMBLE_ACC] = EnsAcc
            # TODO dummy = precisions_sanity_check(precs_all)
        return freqs_dict | precisions_dict

    # The argument feat is a range feature, rownms are names of the rows in the input
    # data (normally these are indices of the rows but sometimes can be strings).
    # When expand is False, this function returns a string built by concatinating
    # the indices/names of positive samples captured within this range feature (feat). 
    # Otherise it returns a boolean vector (list) indicating the locations of positive samples
    # captured within the range in a vector of indices of the positive samples.
    # The above is actually true when the argument isPos is T, otherwise the function
    # computes the same values for the negative samples instead.
    # Similarly, the argument inVsOut decides we are looking for the positive or negative samples 
    # within the range (value T) or outside (value F), thus the function can compute
    # the positive and negative samples indices outside the range as well
    def feat_resp2pos_sample_indices(self, feat, resp, rownms, pos_value_name, neg_value_name, 
            isPos=True, expand=False, inVsOut=True):
        #print('feat\n', feat.tolist(), '\nrownms\n', rownms, '\nresp\n', resp.tolist())
        if len(feat) != len(rownms) or len(resp) != len(rownms):
            raise Exception('Implementation error in function feat_resp2pos_sample_indices')
        #print('pos_value_name', pos_value_name, 'neg_value_name', neg_value_name)
        pos_val = pos_value_name if isPos else neg_value_name; #print('pos_val', pos_val)
        feat_val = 1 if inVsOut else 0; #print('feat_val', feat_val)
        # condition to select rows/indices of positive in, positive out, negative in and negative out
        # samples, depending on values of pos_val and feat_val, for the range featre feat
        #print('feat == feat_val', (feat == feat_val).tolist()); print('resp == pos_val ', (resp == pos_val).tolist())
        subset_cond_list = (feat == feat_val) & (resp == pos_val); #print('subset_cond_list', subset_cond_list.tolist())
        class_samples_in = [str(rownms[i]) for i in range(len(rownms)) if subset_cond_list[i]]; #print('class_samples_in', class_samples_in);
        
        if expand:
            subset_cond_list = resp == pos_val
            class_samples = [rownms[i] for i in range(len(rownms)) if subset_cond_list[i]];
            #rownms [ resp == pos_val ]; #pnv(class_samples)
            # class_samples %in% class_samples_in
            feat_class_bin_col = [e in class_samples_in for e in class_samples] ; print(feat_class_bin_col); 
            return feat_class_bin_col
        else:
            # paste(class_samples_in, collapse=rpa_class_indices_separator)
            rpa_class_indices_separator = '~~' # TODO create a command line option for this
            class_samples_in_str = 'none' if len(class_samples_in) == 0 else rpa_class_indices_separator.join(class_samples_in)
            #print('class_samples_in_str', class_samples_in_str)
            return class_samples_in_str
             
    def fs_ranking_with_frequencies(self, resp:pd.Series, resp_name:str, non_range_feat:list, labeled_features:pd.DataFrame, 
            important_range_comb_levels_df:pd.DataFrame,  curr_fs_ranking_df:pd.DataFrame, ranked_fs_curr_1, cls_reg_mode:str, 
            all_min_freq_thresh, pos_min_freq_thresh, pos_value_name, neg_value_name, pos_sam_ind):
        #print('curr_fs_ranking_df\n', curr_fs_ranking_df)
        #print('curr_fs_ranking_df important features', list(curr_fs_ranking_df['important_features']))
        #print('important_range_comb_levels_df cols', important_range_comb_levels_df.columns.tolist())
        #print('unselected features', set(important_range_comb_levels_df.columns.tolist())
        #      .difference(set(list(curr_fs_ranking_df['important_features']))))
        if important_range_comb_levels_df is None:
            if not ranked_fs_curr_1 is None:
                curr_fs_ranking = pd.concat([ranked_fs_curr_1, curr_fs_ranking_df], axis=1)
        if not important_range_comb_levels_df is None:
            fs_rankinng_precisions_dict = {}
            for feat_name in important_range_comb_levels_df.columns.tolist():
                #print('feat_name', feat_name, '\n', important_range_comb_levels_df[feat_name])
                row = self.feat_resp2opt_scores(feat_name, important_range_comb_levels_df[feat_name], resp,
                    non_range_feat, cls_reg_mode, pos_min_freq_thresh, pos_value_name, neg_value_name)
                fs_rankinng_precisions_dict[feat_name] = row
            rpa_frequencies_df = rows_dict_to_df(fs_rankinng_precisions_dict, RANGE_ANALYSIS_INDICATORS)
            assert list(curr_fs_ranking_df['important_features']) == important_range_comb_levels_df.columns.tolist()
            assert curr_fs_ranking_df.shape[0] == rpa_frequencies_df.shape[0]
            
            #print('rpa_frequencies_df\n', rpa_frequencies_df); 
            #print(important_range_comb_levels_df.shape, rpa_frequencies_df.shape, curr_fs_ranking_df.shape)
            #print(list(curr_fs_ranking_df['important_features'])); print(important_range_comb_levels_df.columns)
            
            if cls_reg_mode == CLASSIFICATION:
                TruePosSampleIndices_dict = {}
                FalsePosSampleIndices_dict = {}
                TrueNegSampleIndices_dict = {}
                FalseNegSampleIndices_dict = {}
                row_indices = important_range_comb_levels_df.index.tolist()
                for col in important_range_comb_levels_df.columns.tolist():
                    feat = important_range_comb_levels_df[col]
                    TruePosSampleIndices_dict[col] = self.feat_resp2pos_sample_indices(feat, resp, 
                        row_indices, pos_value_name, neg_value_name, True, False)
                    FalsePosSampleIndices_dict[col] = self.feat_resp2pos_sample_indices(feat, resp, 
                        row_indices, pos_value_name, neg_value_name, False, False)
                    FalseNegSampleIndices_dict[col] = self.feat_resp2pos_sample_indices(feat, resp, 
                        row_indices, pos_value_name, neg_value_name, True, False, False)
                    TrueNegSampleIndices_dict[col] = self.feat_resp2pos_sample_indices(feat, resp, 
                        row_indices, pos_value_name, neg_value_name, False, False, False)
                tp_df = rows_dict_to_df(TruePosSampleIndices_dict, [TP_SAMPLE_INDICES])
                fp_df = rows_dict_to_df(FalsePosSampleIndices_dict, [FP_SAMPLE_INDICES])
                fn_df = rows_dict_to_df(FalseNegSampleIndices_dict, [FN_SAMPLE_INDICES])
                tn_df = rows_dict_to_df(TrueNegSampleIndices_dict, [TN_SAMPLE_INDICES])
                #print('tp_df\n',tp_df); print('fp_df\n',fp_df); print('fn_df\n',fn_df);print('tn_df\n',tn_df)
                rpa_frequencies_df = pd.concat([rpa_frequencies_df, tp_df, fp_df, fn_df, tn_df], axis=1)
                #print('rpa_frequencies_df\n', rpa_frequencies_df)   
                #print('curr_fs_ranking_df\n', curr_fs_ranking_df)
                #assert False    
            curr_fs_ranking_df.index=curr_fs_ranking_df['important_features']    
            curr_fs_ranking_df = pd.concat([curr_fs_ranking_df, rpa_frequencies_df], axis=1)
            #fs_ranking_df['important_features'] = fs_ranking_df.index
        #print('curr_fs_ranking_df', curr_fs_ranking_df.columns, '\n', curr_fs_ranking_df)
        return curr_fs_ranking_df
        

    # The arguments psg_ranges_map_df, psg_pairs_map_df, psg_triplets_map_df store important single  
    # ranges, pairs and triplets, respectively, selected by PSG (= pysubgroup) algorithm. The argument 
    # ranges_map is built from psg_ranges_map_df before invoking this function, and so is levels_df,
    # which contains 0/1 features built from PSG-selected single ranges that occur in ranges_map.
    # The argument psg_df is also a returned value of the PSG selection and summarizes all the selected
    # ranges (single, pair and triplet ranges, and maybe ranges of higher dimensions that are ignored
    # by this function because the ranking_df computed by this fnction is mainly used for visualizing
    # the selected range features). The argument feat_df contains original features passed to PSG algo.
    def _psg_rules_to_ranking_df(self, resp_name:str, resp_df:pd.DataFrame,
            psg_results_dict:dict, feat_df:pd.DataFrame, levels_df:pd.DataFrame, cls_reg_mode:str):
        # components of PSG result
        # summary info on all selcted ranges (single, pair, ...)
        psg_df = psg_results_dict['psg_df'] 
        # info on features/levels/range for selected single ranges 
        psg_ranges_map_df = psg_results_dict['psg_ranges_map_df'] 
        # info on how selcted range pairs are made up from single ranges
        psg_pairs_map_df = psg_results_dict['psg_pairs_map_df']
        # info on how selcted range triplets are made up from single ranges
        psg_triplets_map_df = psg_results_dict['psg_triplets_map_df']

        # computed in this function and returned
        # df containing info on selected ranges, used for ploting results
        fs_ranking_df = None 
        # names of single ranges selcted by PSG algorithm
        sel_range_names = [] 
        # names of range tuples selected by PSG algo
        sel_range_comb_names = [] 
        # dataframe of range pair features generated from PSG-selected range pairs
        range_pair_levels_df = None 
        # df of range triplet features generated from PSG-selected range triplets
        range_triplet_levels_df = None 
        
        if cls_reg_mode == CLASSIFICATION:
            lift_vec = psg_df['lift']
        elif cls_reg_mode == REGRESSION:
            lift_vec = psg_df['mean_lift']
        else:
            raise Exception('Usupported analytics mode in function psg_rules_to_ranking_df')
        
        compute_full_fs_ranking_df = True
        if compute_full_fs_ranking_df:
            emty_range = ('', 'NA:NA', 'NA:NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA')
        else:
            emty_range = ('', 'NA:NA', 'NA:NA', 'NA')
        rules = psg_df['subgroup']; #print(rules)
        single_ranges = psg_ranges_map_df.columns.tolist()
        range_pairs = psg_pairs_map_df.columns.tolist()
        range_triplets = psg_triplets_map_df.columns.tolist()
        
        def single_range_rule_to_row(r):
            psg_range_tuple = tuple(psg_ranges_map_df[r])  #ranges_map_csv
            feat_name = psg_range_tuple[0]
            if pd_df_col_is_numeric(feat_df, feat_name):
                col_type = self._ORIGIN_NUM
            elif pd_df_col_is_categorical(feat_df, feat_name):
                col_type = self._ORIGIN_CTG
            else:
                raise Exception('Usupported column type for column ' + str(c) + 
                                ' in function _psg_ranges_to_single_ranges_map')
            
            rng_nm, rng_tup = self._psg_range_tuple_to_range_tuple(psg_range_tuple, col_type); 
            #print('rng_tup', rng_tup, 'rng_nm', rng_nm);
            if compute_full_fs_ranking_df:
                # (feature_i, range_i_lo:range_i_hi, start_bin:end_bin, max_bins, min, max, mean, std)
                if col_type == self._ORIGIN_NUM:
                    row = (rng_tup['name'], str(rng_tup['lo']) + ':' + str(rng_tup['hi']), '1:1', 1, #'NA:NA', 'NA', 
                           np.min(feat_df[rng_tup['name']]), np.max(feat_df[rng_tup['name']]),
                           np.mean(feat_df[rng_tup['name']]), np.std(feat_df[rng_tup['name']]))
                else:
                    row = (rng_tup['name'], psg_range_tuple[1], 'NA:NA', 'NA', 
                           np.nan, np.nan, np.nan, np.nan); #print(row)
            else:
                # (feature_i, range_i_lo:range_i_hi, start_bin:end_bin, max_bins)
                if col_type == self._ORIGIN_NUM:
                    row = (rng_tup['name'], str(rng_tup['lo']) + ':' + str(rng_tup['hi']), '1:1', 1) #'NA:NA', 'NA'
                else:
                    row = (rng_tup['name'], psg_range_tuple[1], 'NA:NA', 'NA') #print(row)

            return {'row':row, 'rng_nm':rng_nm, 'rng_tup':rng_tup}
        
        self._psg_logger.info('Ranges (subgroups) selected by PSG Subgroup Discovery algorithm for response ' + 
            str(resp_name) +':\n' + str(rules))

        for i, r in enumerate(rules):
            skip_this_range = False
            # py subgroup results dataframe often contains special range called "Dataset", which
            # means that all samples are selected. We omit such a range from ranking reports
            if r == "Dataset":
                # we omit the corresponding element from lift_vec
                del lift_vec[i]
                continue
            
            # determine the dimension of the rule
            #print('r', r, range_pairs[0], type(r), type(range_pairs[0])); print('range_pairs', range_pairs)
            if str(r) in single_ranges:
                r_dim = 1
            elif str(r) in range_pairs:
                r_dim = 2
            elif str(r) in range_triplets:
                r_dim = 3
            else:
                raise Exception('Range with dimension greater than 3 in function psg_rules_to_ranking_df')
    
            row_i = None
            name_i = [] # name of the range (single, pair or triplet)
            nm_vec = [] # list of names of the single ranges within the range pair or triplet
            rng_feat_i = [True] * levels_df.shape[0] # range feature (ninary 1/0 feature)
            
            # we only extract up to triplets, hence the indices [0,1,2] in the for loop below to access
            # range feature info in PSG result tables psg_ranges_map_df, psg_pairs_map_df, psg_triplets_map_df
            for j in [0,1,2]: # indices in 
                #print('j', j)
                if j >= r_dim:
                    row_i = (row_i) + emty_range; #print('row_i', row_i);
                    continue
        
                if r_dim == 1:
                    #print("single range")
                    row_dict = single_range_rule_to_row(str(r)); #print(row_dict)
                    row = row_dict['row']; #print('row', row)
                    name_i = rng_nm = row_dict['rng_nm']; #print(rng_nm)
                    row_i = (row_i) + row if not row_i is None else row
                    sel_range_names.append(rng_nm); #print(sel_range_names)
                    
                if r_dim == 2: 
                    #print("range pair")
                    pr = psg_pairs_map_df[str(r)] ; #print('pr\n', pr);
                    sel = pr[j]; #print('selector', sel)
                    row_dict = single_range_rule_to_row(sel); #print('row_dict', row_dict)
                    rng_nm = row_dict['rng_nm']; #print('rmg_nm',rng_nm)
                    row = row_dict['row']; #print('row', row)
                    if not row_i is None:
                        row_i = (row_i) + row; #print('row_i', row_i)
                        assert not name_i is None
                        name_i = name_i + self._RANGES_SEP + rng_nm; #print('name_i', name_i)
                        rng_feat_i = rng_feat_i & levels_df[rng_nm]; #print('\nrng_feat_i\n', rng_feat_i)
                    else:
                        row_i = row
                        name_i = rng_nm
                        rng_feat_i = levels_df[rng_nm]
                    
                    if j == 1:
                        #print('rng_feat_i\n', rng_feat_i); print('name_i', name_i); 
                        if len(list_unique_unordered(list(rng_feat_i))) == 1:
                            self._psg_logger.warning('[-v-] Range ' + str(name_i) + ' has only one value ' + 
                                str(list_unique_unordered(rng_feat_i)[0]) + '; skipping it...\n')
                            skip_this_range = True
                            del lift_vec[i];
                            continue
                        else:
                            if range_pair_levels_df is None:
                                rng_feat_i = pd.Series(rng_feat_i, name=name_i)
                                range_pair_levels_df = pd.concat([rng_feat_i], axis=1)
                            else:
                                range_pair_levels_df[name_i] = rng_feat_i; 
                            #print('range_pair_levels_df\n', range_pair_levels_df);
                    
                if r_dim == 3:
                    #print("range triplet")
                    tr = psg_triplets_map_df[str(r)]; #print(tr); 
                    sel = tr[j]; #print(sel)
                    row_dict = single_range_rule_to_row(sel); #print('row_dict', row_dict)
                    rng_nm = row_dict['rng_nm']; #print('rmg_nm',rng_nm)
                    row = row_dict['row']; #print('row', row)
                    if not row_i is None:
                        row_i = (row_i) + row; #print('row_i', row_i)
                        assert not name_i is None
                        name_i = name_i + self._RANGES_SEP + rng_nm; #print('name_i', name_i)
                        rng_feat_i = rng_feat_i & levels_df[rng_nm]; #print('\nrng_feat_i\n', rng_feat_i)
                    else:
                        row_i = row
                        name_i = rng_nm
                        #print('rng_nm', rng_nm); print('levels_df cols', levels_df.columns.tolist())
                        rng_feat_i = levels_df[rng_nm]

                    if j == 2:
                        feat_df_cond = feat_df
                        colnms = []
                        rngnms = []
                        for m in [0,1]:
                            #print(m); print(tr); print(length(tr))
                            sel_m = tr[m]; #print(sel_m)
                            row_dict_m = single_range_rule_to_row(sel_m); #print('row_dict', row_dict)
                            rng_nm_m = row_dict_m['rng_nm']; #print('rmg_nm_m',rng_nm_m)
                            rng_tup_m = row_dict_m['rng_tup']; #print('rmg_tup_m',rng_tup_m)
                            row = row_dict_m['row']; #print('row', row)
                            origin = rng_tup_m['origin']
                            nm_m = rng_tup_m['name']
                            colnms.append(nm_m)
                            rngnms.append(rng_nm_m)
                                
                            col = feat_df_cond[nm_m]; #print('col\n', col)
                            if origin == self._ORIGIN_NUM:
                                hi = rng_tup_m['hi']; #print(hi)
                                lo = rng_tup_m['lo']; #print(lo)
                                cond = (col <= hi) & (col >= lo); #print(cond)
                            elif origin == self._ORIGIN_CTG:
                                lvl = self.unique_level_to_level(rng_tup_m)
                                #lvl = unique_level_to_level(rng_nm_m, ranges_map); #print(lvl)
                                cond = str(col) == lvl
                            else:
                                raise Exception('Unknown tuple origin in function psg_rules2ranking_df')
                            feat_df_cond = feat_df_cond[cond]; #print('feat_df_cond\n', feat_df_cond)
                            
                        if len(list_unique_unordered(list(rng_feat_i))) == 1:
                            self._psg_logger.warning('[-v-] Range ' + str(name_i) + ' has only one value ' + 
                                str(list_unique_unordered(rng_feat_i)[0]) + '; skipping it...\n')
                            skip_this_range = True
                            del lift_vec[i];
                            continue
                        else:
                            if range_triplet_levels_df is None:
                                rng_feat_i = pd.Series(rng_feat_i, name=name_i)
                                range_triplet_levels_df = pd.concat([rng_feat_i], axis=1)
                            else:
                                range_triplet_levels_df[name_i] = rng_feat_i; 
                            #print('range_pair_levels_df\n', range_triplet_levels_df)
                                            
            if not skip_this_range:  
                if compute_full_fs_ranking_df:
                    colnames=(
                        self._FEATURE_1, self._RANGE_1, self._BINS_1, self._MAX_BINS_1, 'min_1', 'max_1', 'mean_1', 'std_1',
                        self._FEATURE_2, self._RANGE_2, self._BINS_2, self._MAX_BINS_2, 'min_2', 'max_2', 'mean_2', 'std_2',
                        self._FEATURE_3, self._RANGE_3, self._BINS_3, self._MAX_BINS_3, 'min_3', 'max_3', 'mean_3', 'std_3')
                else:
                    colnames=(
                        "feature", "range_1", "bins_1", self._MAX_BINS_1, 
                        "feature_2", "range_2", "bins_2", self._MAX_BINS_2,
                        "feature_3", "range_3", "bins_3", self._MAX_BINS_3)
                #print('colnames', colnames); print('row_i', row_i)
                row_dict = dict(zip(colnames, row_i)); #print('row_dict', row_dict)
                row_df = pd.DataFrame([row_dict]);  #print('row_df\n', row_df)
                if fs_ranking_df is None:
                    fs_ranking_df = row_df
                else:
                    fs_ranking_df = pd.concat([fs_ranking_df, row_df], axis=0)
                sel_range_comb_names.append(name_i)
        
        #print('fs_ranking_df\n', fs_ranking_df); print('resp_name', resp_name)
        
        
        if compute_full_fs_ranking_df:
            fs_ranking_df.insert(0, 'response', resp_name);
            # TODO here for pysubgroup we do not have max_bins option value -- 
            # using value 0. Need a proper support for other selection methods
            fs_ranking_df.insert(1, 'max_bins', 0);
            fs_ranking_df.insert(2, 'min', np.min(resp_df[resp_name]))
            fs_ranking_df.insert(3, 'max', np.max(resp_df[resp_name]))
            fs_ranking_df.insert(4, 'mean', np.mean(resp_df[resp_name]))
            fs_ranking_df.insert(5, 'std', np.std(resp_df[resp_name]))
            fs_ranking_df.insert(6, 'range', sel_range_comb_names)
        else:
            # add column "Label" with the response names as values
            fs_ranking_df['Label'] = resp_name
            
        # add column "score" with lift_vec as the values 
        fs_ranking_df['score'] = list(map(float, lift_vec))

        # add column "selection" with all values equal to "PSG" to indicate that the ranges
        # were generated using PSG algorithm (which can be seen as rule inference algorithm)
        fs_ranking_df['selection'] = 'PSG' 

        # TODO This is temporary, will use 'range' column instead
        # add column "important_features" with sel_range_comb_names as values, which contains
        # the names of all the ranges (single, pair triplet) selected by PSG algo.
        fs_ranking_df.insert(0, 'important_features', sel_range_comb_names)

        #print('fs_ranking_df\n', fs_ranking_df)
        #print('range_pair_levels_df\n', range_pair_levels_df); 
        #print('range_triplet_levels_df\n', range_triplet_levels_df)
        return {'fs_ranking_df':fs_ranking_df, 
                'sel_range_names':sel_range_names, 
                'sel_range_comb_names':sel_range_comb_names,
                'single_range_levels_df':levels_df,
                'range_pair_levels_df':range_pair_levels_df,
                'range_triplet_levels_df':range_triplet_levels_df}
    
    
    # Computes and returns "important" single ranges, range pairs and triplets using the 
    # pysubgroup (PSG) implementation of the subgroup Discovery algorothm.Atgument feat_df 
    # is the input features dataframe passed to PSG, resp_name is the response name, qf is
    # a quality function that defines the "importance" or the score for selcting ranges,
    # dim is the maximal dimension of selected ranges (useually = 3 because ranges up to
    # triplets can conveniently be plotted for user inspection), and top_n is the required
    # count of selected ranges. After selecting ranges (single ranges, pairs, triplets...),
    # the function also computes and returns single range, range pair and range triplet 
    # features (which are binary 1/0 features) and genrates a ranking table (as dataframe)
    # containing all the range information required to plot selected ranges, up to triplets.
    # function to visualize the selected ranges).
    def _smlp_subgroups_single_response(self, feat_df:pd.DataFrame, resp_df:pd.DataFrame, 
            resp_name:str, pos_value:int, qf:str, dim:int, top_n:int):
        assert pos_value == 0 or pos_value == 1
        #print('smlp_subgroups_single_response: feat_df cols', feat_df.columns.tolist())
        cls_reg_mode = get_response_type(resp_df, resp_name); #print('cls_reg_mode', cls_reg_mode)
        assert cls_reg_mode == CLASSIFICATION or cls_reg_mode == REGRESSION
        if cls_reg_mode == REGRESSION:
            #print('Numeric target')
            target = ps.NumericTarget(resp_name)
            feat_resp_df = pd.concat([feat_df, resp_df], axis=1);
        else:
            # keep resp_df as is, to use in fs_ranking_with_frequencies (thus the use of inplace=False)
            pf_resp_df = resp_df[resp_name].replace({pos_value: self._psg_positive_value, 
                1-pos_value: self._psg_negative_value}, inplace=False)
            #print('resp_df\n', resp_df); print('pf_resp_df\n', pf_resp_df)
            target = ps.BinaryTarget(resp_name, self._psg_positive_value)
            feat_resp_df = pd.concat([feat_df, pf_resp_df], axis=1); #print('feat_resp_df\n', feat_resp_df);
        searchspace = ps.create_selectors(feat_resp_df, ignore=[resp_name])
        qf = self.evaluateSupportedQF(qf, cls_reg_mode == CLASSIFICATION)
        task = ps.SubgroupDiscoveryTask(feat_resp_df, target, searchspace,
            result_set_size=int(top_n), depth=int(dim), qf=qf)
        self._psg_logger.info('PSG Subgroup Discovery started')
        result = ps.BeamSearch(beam_width_adaptive=True).execute(task)
        self._psg_logger.info('PSG Subgroup Discovery completed')
        psg_df = result.to_dataframe()
        results_to_iterate = result.results

        psg_ranges_map_df =  pd.DataFrame()
        psg_pairs_map_df = pd.DataFrame()
        psg_triplets_map_df = pd.DataFrame()

        # iterate to create / fill-in psg_ranges_map_df, psg_pairs_map_df, psg_pairs_map_df
        # we ignore ranges with dimentionality greater than 3
        for i in results_to_iterate:
            #print('i', i, type(i)); print(i[1], type(i[1]))
            (q, sg, stats) = i
            sr_nm = str(sg)

            # we ignore ranges with dimentionality greater than 3
            if len(sg) > 3:
                continue

            k = 0
            tup = list()
            for s in sg.selectors:
                s_nm = str(s)
                lft = s.attribute_name; #print('lft', lft)
                isEqualitySelector = isinstance(s, ps.subgroup_description.EqualitySelector)
                isIntervalSelector = isinstance(s, ps.subgroup_description.IntervalSelector)
                isNegatedSelector = isinstance(s, ps.subgroup_description.NegatedSelector)
                if isIntervalSelector:
                    #print('numeric selector', s)
                    lo = s.lower_bound
                    hi = s.upper_bound
                    psg_ranges_map_df[s_nm] = [lft, str(lo), str(hi), "numeric"]
                    tup.append(s_nm)
                elif isEqualitySelector:
                    #print('nominal selector', s)
                    rgt = s.attribute_value
                    pred = s.attribute_name
                    #print("pred", pred)
                    psg_ranges_map_df[s_nm] = [lft, str(rgt),"dummy_psg", "nominal"]
                    tup.append(s_nm)
                elif isNegatedSelector:
                    #print('Negated Selector; abort')
                    psg_df = pd.DataFrame()
                    return(psg_df)
                else:
                    self._psg_logger.warning('Object of unexpected type in PSG rules; abort')
                    #print('Object of unexpected type in PSG rules; abort')
                    psg_df = pd.DataFrame()
                    return(psg_df)

                # last element of the loop
                if k == len(sg.selectors) - 1:
                    if k == 1:
                        # we have a pair
                        #print('add to pairs'); print(sr_nm); print(tup)
                        psg_pairs_map_df[sr_nm] = tup
                    elif k == 2:
                        # we have triplet
                        #print('add to triplets')
                        psg_triplets_map_df[sr_nm] = tup
                k = k + 1

        #print('return psg_df\n', psg_df, '\psg_ranges_map_df\n', psg_ranges_map_df, '\npsg_pairs_map_df\n', 
        #    psg_pairs_map_df, '\npsg_triplets_map_df\n', psg_triplets_map_df);
        
        ranges_map = self._psg_ranges_to_single_ranges_map(psg_ranges_map_df, feat_df)
        levels_df = self.feature_and_range_to_in_range(feat_df, ranges_map.keys(), ranges_map); 
        #print('levels_df\n', levels_df, '\n', levels_df.columns.tolist())
        psg_results_dict = {'psg_df':psg_df, 
                            'psg_ranges_map_df':psg_ranges_map_df, 
                            'psg_pairs_map_df':psg_pairs_map_df, 
                            'psg_triplets_map_df':psg_triplets_map_df}
        smlp_psg_results_dict = self._psg_rules_to_ranking_df(resp_name, resp_df, 
            psg_results_dict, feat_df, levels_df, cls_reg_mode) #ranges_map,
        
        
        # prepare argumnets to run fs_ranking_with_frequencies()
        fs_ranking_df = smlp_psg_results_dict['fs_ranking_df']
        sel_range_names = smlp_psg_results_dict['sel_range_names']
        sel_range_comb_names = smlp_psg_results_dict['sel_range_comb_names']
        #print('sel_range_names', sel_range_names, 'sel_range_comb_names', sel_range_comb_names)
        important_range_comb_levels_df = pd.concat([smlp_psg_results_dict['single_range_levels_df'], 
            smlp_psg_results_dict['range_pair_levels_df'], smlp_psg_results_dict['range_triplet_levels_df']], axis=1)
        assert sel_range_comb_names == list(fs_ranking_df['important_features'])
        important_range_comb_levels_df = important_range_comb_levels_df[sel_range_comb_names]
        important_range_comb_levels_df.index = feat_df.index
        assert any(important_range_comb_levels_df.index == feat_df.index)
        #print(feat_df.index, important_range_comb_levels_df.index)
        labeled_features = feat_df
        resp = resp_df[resp_name] # pd.Series object instead of pd.DataFrame object resp_df
        non_range_feat = []
        ranked_fs_curr_1 = None;
        all_min_freq_thresh = None  # TODO check if pysubgroup supports this 
        pos_min_freq_thresh = None # TODO check if pysubgroup supports this 
        pos_value_name = 1 # TODO currently there is no such command line option, using default
        neg_value_name = 0 # TODO currently there is no such command line option, using default
        pos_sam_ind = None # will need to provide a value in classification mode, it is NONE in regression mode in early versions (?)
        
        fs_ranking_df = self.fs_ranking_with_frequencies(resp, resp_name, non_range_feat, labeled_features, 
            important_range_comb_levels_df, fs_ranking_df, ranked_fs_curr_1, cls_reg_mode, all_min_freq_thresh, 
            pos_min_freq_thresh, pos_value_name, neg_value_name, pos_sam_ind)
        
        subgroup_results_dict_single_response = {
            'rpa_labeled_features':None, # needed in prediction mode
            'important_range_comb_levels_df':important_range_comb_levels_df,
            'resp':resp, 
            'resp_name':resp_name,
            'sel_feat_names':None, # needed in prediction mode
            'sel_range_names':None, # needed in prediction mode
            'sel_range_pair_names':None,  # needed in prediction mode
            'sel_range_triplet_names':None, # needed in prediction mode
            'bins_map':None, 
            'ranges_map':None, 
            'range_pairs_map':None, 
            'range_triplet_map':None, # maps needed in prediction mode
            'fs_ranking_df':fs_ranking_df, 
            'ranked_fs_summary_df':None}

        return subgroup_results_dict_single_response    
    
    # This function combines feature selection results for multiple responses into a single result;
    # the latter result contains feature ranking summary reported in file ...features_summary.csv
    # and feature ranking compact results summary reported in file ...features_ranking.csv
    # full_fs_list contains results of _features_single_response run on each response
    # extracted_resp_feat_list_full contains info about each response and corresponding features
    # TODO since we do not have combined responses, can easily drop arguments 
    # number_of_responses, i_start. Argument output_file is also not used currently,
    def _fs_multi_response_ensemble_summary(self, full_fs_list, extracted_resp_feat_list_full,
        number_of_responses, i_start, pos_value, neg_value, plots, mode, output_file):
        #print("Function fs_multi_response_ensemble_summary starts")

        multi_fs_summary_df = None
        multi_fs_ranking_df = None
        multi_fs_full_data = None

        resp_names = list(full_fs_list.keys()); #print('resp_names', resp_names)
        for resp_name, single_fs in full_fs_list.items():
            #print('resp_name', resp_name)
            assert resp_name == single_fs['resp_name'];
            resp = single_fs['resp']; #print('resp\n', resp, '\nresp_df\n', pd.DataFrame(resp))
            ranked_fs_summary_df = single_fs['ranked_fs_summary_df']; #print(ranked_fs_summary_df)
            fs_ranking_df = single_fs['fs_ranking_df']
            ranges_map = single_fs['ranges_map'] 
            #range_pairs_map = single_fs$range_pairs_map
            #range_triplet_map = single_fs$range_triplet_map
            #pnv(ranges_map); pnv(range_pairs_map); pnv(range_triplet_map)
            
            # merge the thred's result into the main output file and remove the former
            
            # export important features file per response -- TODO !!!
            
            # sanity check of class samples table for the current response
            cls_reg_mode = get_response_type(pd.DataFrame(resp), resp_name); #print('cls_reg_mode', cls_reg_mode)
            # TOD0 !!! dummy = sanity_check_class_sample_tables(fs_ranking_df, cls_reg_mode)

            # join ranked features and selected ranges files of multiple responses
            # TODO !!! need support for ranked_fs_summary_df, currently it is None as
            # feature selection heuristics (Pearson, etc.) are not implemented
            if resp_name == resp_names[0]: #is.null(multi_fs_summary_df)
                multi_fs_summary_df = ranked_fs_summary_df
                multi_fs_ranking_df = fs_ranking_df
            else:
                if not multi_fs_summary_df is None:
                    #multi_fs_summary_df = merge_by_important_features(multi_fs_summary_df, ranked_fs_summary_df)
                    # TODO !!! : usage of pd.conacat() instead of merge_by_important_features() has not been tested
                    multi_fs_summary_df = pd.concat([multi_fs_summary_df, ranked_fs_summary_df], axis=1)
                # TODO !!! usage of merge instead of safe_rbind_all_columns has not been tested
                #multi_fs_ranking_df = safe_rbind_all_columns(multi_fs_ranking_df, fs_ranking_df)
                #print('before merge: multi_fs_ranking_df\n', multi_fs_ranking_df, '\nfs_ranking_df\n', fs_ranking_df)
                multi_fs_ranking_df = multi_fs_ranking_df.merge(fs_ranking_df, how = 'outer')
        
        #print(multi_fs_ranking_df)
        multi_fs_ranking_df.drop(['important_features'], axis=1, inplace=True)
        # export important features file, combined, for all responses
        fs_ranking_csv_filename = self.get_features_ranking_file()
        multi_fs_ranking_df.to_csv(fs_ranking_csv_filename, index=False)
        
        # generate and write out ranking_resp_features.csv file which contain processed input data, and only
        # the features that occur in multi_fs_ranking_df; such feature names are within columns 'feature_1', 
        # 'feature_2', 'feature_3'. These column might also contain his input data file is used for generating plots along
        # with multi_fs_ranking_df. 
        #full_feature_rows = multi_fs_ranking_df['range_1'] != 'NA:NA'
        #multi_fs_range_ranking_df = full_feature_rows[multi_fs_range_ranking_df['range_1'] != 'NA:NA']
        
        ranked_features = []
        for i in [1,2,3]:
            col = 'feature_{i}'.format(i=i)
            if col in multi_fs_ranking_df:
                ranked_features = ranked_features + multi_fs_ranking_df[col].dropna().unique().tolist()
        ranked_features = list_unique_ordered(ranked_features)
        #ranked_features might contain empty string '' and / or Na string 'NA' -- need to drop them, they are likely 
        # not ranked feature names 
        feat_df = extracted_resp_feat_list_full['features']
        assert not '' in feat_df
        exists_feature_called_NA = 'NA' in feat_df
        names_to_drop = ['','NA'] if not exists_feature_called_NA else ['']
        for e in names_to_drop:
            if e in ranked_features:
                ranked_features.remove(e)
        #print('ranked_features', ranked_features)
        ranking_resp_feat_df = extracted_resp_feat_list_full['features'][ranked_features]
        ranking_resp_feat_df = pd.concat([ranking_resp_feat_df, extracted_resp_feat_list_full['responses']], axis=1)
        ranking_resp_feat_csv_filename = self.get_ranking_resp_feat_file()
        ranking_resp_feat_df.to_csv(ranking_resp_feat_csv_filename, index=False)
        
        # generate plots html if RangePlots module was available / has been imported
        if plots:
            if RangePlots_are_missing:
                self._psg_logger.warning('Range plots are not supported in this version of SMLP')
            else:
                plots_dir = self.get_range_plots_dir()
                plots_config = '' # TODO !!!: this is the default value, need to understand why config is required and 
                                  # decide whether a command line option name is useful
                pos_is_low = pos_value == STAT_NEGATIVE_VALUE #False # TODO !!!: need to define the command line option for this
                summary_table = False # TODO !!!: using default value, need to decide whether a command line option
                                      # is useful and in fact understand whether to support this table in the future
                html_file = self.instRangePlots.mainRun(fs_ranking_csv_filename, plots_dir, plots_config, pos_is_low, summary=summary_table)
                ''' TODO !!!" we might want plots interactive by opening plots html and pausing excution
                    till user notifies so interactively. The code below is to open the html link but does
                    not work properly if a firefox window is already open in user's environment
                # import module
                import webbrowser
                # open html file
                #webbrowser.open(html_file) 
                webbrowser.open_new_tab(html_file)
                '''        
        return multi_fs_ranking_df, multi_fs_summary_df        

    # counterpart of rules, except data preparation is done outside (before this call) and also
    # it also covers pysubgroup and not other subgroup iplementations like prim or cn2 
    def smlp_subgroups(self, feat_df:pd.DataFrame, resps_df:pd.DataFrame, resp_names:list, 
            pos_value:int, neg_value:int, qf:str, dim:int, top_n:int, plots:bool):
        #print('smlp_subgroups: resp_names', resp_names)
        #print('smlp_subgroups: feat_df cols', feat_df.columns.tolist())
        results_dict = {}
        fs_ranking_df = None
        for rn in resp_names:
            resp_df = resps_df[[rn]]; #print('resp_df\n', resp_df)
            results_dict[rn] = self._smlp_subgroups_single_response(feat_df, resp_df, rn, 
                pos_value, qf, dim, top_n)
            curr_fs_ranking_df = results_dict[rn]['fs_ranking_df']
            fs_ranking_df = pd.concat([fs_ranking_df, curr_fs_ranking_df], axis=0)
        #print('feat_df\n', feat_df)
        full_fs_list = results_dict
        extracted_resp_feat_list_full = {'features':feat_df, 'responses':resps_df}
        # TODO !!!: we might want to keep features_df as a seprate item in extracted_resp_feat_list_full 
        # and do not include in extracted_resp_feat_list_full[rn] for every response rn. In addition,
        # include resp_feat_dict which will specify the relevant features (selcted through MRMR for eaxmple) 
        # per response.
        #for rn in resp_names:
        #    extracted_resp_feat_list_full[rn] = {'features':feat_df, 'responses':resps_df}
        
        number_of_responses = len(resp_names)
        i_start = 1
        mode = "features"
        output_file = None # TODO !!! complete implementation
        fs_ranking_df, fs_summary_df = self._fs_multi_response_ensemble_summary( 
            full_fs_list, extracted_resp_feat_list_full, number_of_responses, i_start, 
            pos_value, neg_value, plots, mode, output_file)
        
        return fs_ranking_df, fs_summary_df, results_dict
                
                
        
        
    

