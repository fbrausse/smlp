import numpy as np



class PrecisionMeasures():
    def __init__(self):
        # default values of postive and negative in statistics
        self.STAT_NEGATIVE_VALUE = int(0)
        self.STAT_POSITIVE_VALUE = int(1)
        # default values of postive and negative in SMLP
        self.SMLP_NEGATIVE_VALUE = int(0)
        self.SMLP_POSITIVE_VALUE = int(1)
        
    def contingency_table(self, predictions: np.ndarray, ground_truth: np.ndarray,
            negative: int, positive: int) -> dict:
            # TODO !!! seting defaults to negative and positive as in next line does not work...
            # negative: int=self.STAT_NEGATIVE_VALUE, positive: int=self.STAT_POSITIVE_VALUE) -> dict:
            # this error is reported: 
            # negative: int=self.STAT_NEGATIVE_VALUE, positive: int=self.STAT_POSITIVE_VALUE) -> dict:
            # NameError: name 'self' is not defined
        tp = np.sum(np.logical_and(predictions == positive, ground_truth == positive))
        tn = np.sum(np.logical_and(predictions == negative, ground_truth == negative))
        fp = np.sum(np.logical_and(predictions == positive, ground_truth == negative))
        fn = np.sum(np.logical_and(predictions == negative, ground_truth == positive))

        return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}

    
    #true_pred_df = data.frame(true= bin_true_vec, pred=bin_pred_vec); #pnv(true_pred_df); 
    #kappa2_res = kappa2(true_pred_df, "unweighted"); 
    #kappa2_res = kappa2(true_pred_df, "squared"); 
    #cohen_kappa = kappa2_res$value; #pnv(cohen_kappa); 
    #true_pred_categ_df = data.frame(true=factor(bin_true_vec, levels=c(0,1)), 
    #          pred=factor(bin_pred_vec, levels=c(0,1))); pnv(true_pred_categ_df); 
    def compute_cohen_kappa(self, accuracy, predicted_negative, predicted_positive,
            true_negative, TP, samples_count):
        #pnv(accuracy); pnv(is.nan(accuracy))
        if np.isnan(accuracy): 
            return np.nan

        if accuracy == 1:
            cohen_kappa = 1 
        else:
            #pos_expectation = (predicted_negative/samples_count)*true_negative; pnv(pos_expectation)
            #neg_expectation = (predicted_positive/samples_count)*TP; pnv(neg_expectation)
            pos_expectation = predicted_positive*TP/samples_count; 
            neg_expectation = predicted_negative*true_negative/samples_count; 
            expected_accuracy = (pos_expectation+neg_expectation)/samples_count; 
            if expected_accuracy == 1:
                raise Exception('Implementation error in computing Kappa statistic')

            cohen_kappa = (accuracy - expected_accuracy)/(1 - expected_accuracy); #pnv(cohen_kappa);
        
        # TODO !!! dummy = precision_sanity_check(cohen_kappa, COHEN_KAPPA)
        return cohen_kappa
    
    
    # WRAcc(Class <- Cond) = p(Cond)·(p(Class|Cond)-p(Class)). 
    # Argument TN = true_negative, FN=false_negative, TP=true_positive, FP=false_posive
    # this function assumes that range analysis isolates positive units (failures)
    # Usage: compute_weighted_relative_accuracy(PosOut, NegOut, PosIn, NegIn)
    def compute_weighted_relative_accuracy(self, FN,TN,TP,FP):
        #print(c(TN,FN,TP,FP))
        if TN == 0 and FN == 0 and TP == 0 and FP == 0:
            wr_acc = NaN
        else:
            All = TN+FN+TP+FP
            AllIn = FP+TP 
            wr_acc = (AllIn/All)*(TP/AllIn-(TP+FN)/All); #print(wr_acc)
            # normalize
            wr_acc = (wr_acc+1)/2
            # TODO !!! dummy = precision_sanity_check(wr_acc, WR_ACC)
        return wr_acc
    
    def compute_false_positive_rate(self, FP,TN):
        #print(c(TN,FN,TP,FP))
        FPr = 0 if FP == 0 else FP/(FP+TN)
        # TODO !!! dummy = precision_sanity_check(FPr, FALSE_POSITIVE_RATE)
        return FPr

    def compute_true_positive_rate(self, TP, FN, useNaN=False):
        if useNaN:
            TPr = TP/(TP+FN)
        elif FN == 0:
            TPr = 1
        elif TP == 0:
            TPr = 0
        else:
            TPr = TP/(TP+FN)            
        # TODO !!! dummy = precision_sanity_check(TPr, TRUE_POSITIVE_RATE)
        return TPr
        
    def compute_false_negative_rate(self, FN,TP):
        FNr = 0 if FN == 0 else FN/(FN+TP)
        # TODO !!! dummy = precision_sanity_check(FNr, FALSE_NEGATIVE_RATE)
        return FNr
        
    def compute_true_negative_rate(self, TN,FP):
        TNr = 1 if FP == 0 else 0 if TN == 0 else TN/(TN+FP)
        # TODO !!! dummy = precision_sanity_check(TNr, TRUE_NEGATIVE_RATE)
        return TNr

    def compute_accuracy(self, TN,TP,All):
        acc = (TN+TP)/All
        # TODO !!! dummy = precision_sanity_check(acc, ACCURACY)
        return acc
        
    def compute_roc_accuracy_from_rates(self, TPr, FPr):
        roc = TPr-FPr; #print(roc)
        # normalize
        roc = (roc+1)/2
        # TODO !!! dummy = precision_sanity_check(roc, ROC_ACC)
        return roc
        
    # Argument pos==T means that we are evaluating the prediction precision of positive class 
    # (i.e., the response has the pos value); when pos == F, we are evaluating the prediction 
    # of the negative class (i.e., the response has the neg value); 
    # Argument TN = true_negative, FN=false_negative, TP=true_positive, FP=false_positive
    def compute_roc_accuracy(self, TN,FN,TP,FP):
        #print(c(TN,FN,TP,FP))
        if TN == 0 and FN == 0 and TP == 0 and FP == 0:
            roc_acc = np.nan
            # TODO !!! dummy = precision_sanity_check(roc_acc, ROC_ACC)
            return roc_acc

        FPr = self.compute_false_positive_rate(FP,TN)
        TPr = self.compute_true_positive_rate(TP,FN)
        roc_acc_pos = self.compute_roc_accuracy_from_rates(TPr, FPr)

        # ROC accuracy is actually symetric wrt pos vs neg, thus the following can be used as 
        # sanity check; desabled as default to avaoid otherwise redundant computation
        ROCAcc_sanity_check = True # TODO !!! this check is disabled in early versions
        if ROCAcc_sanity_check:
            FNr = self.compute_false_negative_rate(FN,TP); 
            TNr = self.compute_true_negative_rate(TN,FP); 
            #roc_acc_neg = TNr-FNr; 
            roc_acc_neg = self.compute_roc_accuracy_from_rates(TNr, FNr);
            multi_layer_numeric_encoding = False # TODO !!! drop usage of this variable
            if not multi_layer_numeric_encoding and abs(roc_acc_pos-roc_acc_neg) > 1e-8:
                raise Exception('Implementation error in function compute_roc_accuracy')

        # TODO !!! dummy = precision_sanity_check(roc_acc_pos, ROC_ACC)
        return roc_acc_pos

    # This is an adapted version to insure that the result is always defined (not NaN).
    # TO DO: try Laplace smoothing instead, adapt Laplace smoothing for conditional probabilities
    def compute_predictive_positive_rate(self, TP, predicted_positive, response_positive, useNaN=False):
        if useNaN:
            PPr = TP / predicted_positive
        elif predicted_positive == 0 and response_positive == 0:
            PPr = 1
        elif predicted_positive == 0:
            PPr = 0
        else:
            PPr = TP / predicted_positive
        # TODO !!! dummy = precision_sanity_check(PPr, PRED_POSITIVE_RATE)
        return PPr

    # This is an adapted version to insure that the result is always defines (not NaN).
    # TO DO: try Laplace smoothing instead, adapt Laplace smoothing for conditional probabilities
    def compute_predictive_negative_rate(self, true_negative, predicted_negative, response_negative):
        if predicted_negative == 0 and response_negative == 0:
            PNr = 1
        elif predicted_negative==0:
            PNr = 0
        else:
            PNr = true_negative / predicted_negative
        # TODO !!! dummy = precision_sanity_check(PNr, PRED_NEGATIVE_RATE)
        return PNr

    # In the two AUC (rather, AUROC) functions below, a score is meant as
    # the probability of a sample being positive. Then AUC is the probability
    # that a positive sample will receive a score higher than a negative sample
    # When the label of pos samples is 0 (not the default value), the
    # probabilities returned by a clasifier are probabilities of a sample being 1
    # thus in such a case we pass to these function 1-prob for the argument "scores"
    # and pos.scores and neg.scores are derived from scores by subseting at indicies
    # of positive samples and negative samples, respectively.
    # The AUC is a measure of the ability to rank examples according to the probability 
    # of class membership. Thus if all of the probabilities are above 0.5 you can still 
    # have an AUC of one if all of the positive patterns have higher probabilities than 
    # all of the negative patterns. In this case there will be a decision threshold that 
    # is higher than 0.5, which would give an error rate of zero. Note that because the 
    # AUC only measures the ranking of the probabilities, it doesn't tell you if the 
    # probabilities are well calibrated (e.g. there is no systematic bias), if calibration 
    # of the probabilities is important then look at the cross-entropy metric.
    #
    # The auroc function by Miron Kursa, from https://mbq.me/blog/augh-roc/ .
    # Argment "cls" has to be a logical vector, with TRUEs for positive objects
    # Argument "scores" is the predicted probability of a sample being positive
    # The original version is modified here by supporting the cases when there
    # are no positive samples or there are no negative samples.
    def compute_auroc(self, scores, cls, nan_default=np.nan):
        #pnv(cls); pnv(scores)
        n1 = sum(~cls); 
        n2 = sum(cls); #pnv(c(n1,n2))
        assert n2 == len(cls) - n1
        if n1 == 0 or n2 == 0:
            AUCAcc = nan_default
            # TODO !!! dummy = precision_sanity_check(AUCAcc, AUC_ACC)
            return AUCAcc
        U = sum(rank(scores)[~cls])-n1*(n1+1)/2;
        AUCAcc = 1-U/n1/n2

        # sanity check comparing two methods of AUC computation -- compute_auroc and computeAUC.
        # computeAUC is not desired to use since it uses random number genertor and affects results 
        # of subsequent computations -- makes them non-deterministic; desabled as deafult
        if AUC_sanity_check:
            #pos.scores = pos_prob_vec[bin_true_vec_pos]; pnv(pos.scores); pnv(min(pos.scores))
            #neg.scores = pos_prob_vec[bin_true_vec_neg]; pnv(neg.scores); pnv(max(neg.scores))
            pos.scores=scores[cls]; neg.scores=scores[~cls]
            AUCAcc2 = computeAUC(pos.scores, neg.scores, nan_default); #pnv(AUCAcc2); 
        if not effectively_equal_numeric(AUCAcc,AUCAcc2, 3): # TODO !!!! undefined function effectively_equal_numeric
            raise Exception ('Implementation error 2 in auc computation in function compute_auroc')
        # TODO !!! dummy = precision_sanity_check(AUCAcc, AUC_ACC)
        return AUCAcc

    # another function for computing AUC, from 
    # https://stackoverflow.com/questions/4903092/calculate-auc-in-r
    # The original version is modified here by supporting the cases when there
    # are no positive samples or there are no negative samples.
    # Args:
    #   pos_scores: scores of positive observations
    #   neg_scores: scores of negative observations
    #   n_samples : number of samples to approximate AUC
    def computeAUC(self, pos_scores, neg_scores, nan_default=np.nan, n_sample=100000):
        if is_empty_vec(pos_scores) or is_empty_vec(neg_scores): # TODO !!! undefined function is_empty_vec
            AUCAcc = nan_default
            # TODO !!! dummy = precision_sanity_check(AUCAcc, AUC_ACC)
            return AUCAcc

        pos_sample = pos_scores.sample(n_sample, replace=True)
        neg_sample = neg_scores.sample(n_sample, replace=True)
        AUCAcc = mean(1.0*(pos_sample > neg_sample) + 0.5*(pos_sample==neg_sample))
        # TODO !!! dummy = precision_sanity_check(AUCAcc, AUC_ACC)
        return AUCAcc

    # Next two functions compute F1 score, from different arguments.
    def compute_F1(self, TP, FP, FN):
        if TP == 0:
            F1Score = 0
        else:
            # TP+FP is predicted_positive, TP+FN is all_pos (or all actual/try positive)
            precision = compute_predictive_positive_rate(TP, TP+FP, TP+FN)
            recall = compute_true_positive_rate(TP, FN)
            F1Score = 2*precision*recall/(precision + recall)
        # TODO !!! dummy = precision_sanity_check(F1Score, F1_SCORE)
        return F1Score   

    # Terminology reminder: precision = PPr, recall = sensitivity = TPr
    def compute_F1_from_precision_recall(self, precision, recall, nan_default=np.nan):
        if np.isnan(precision) or np.isnan(recall):
            F1Score = nan_default
        elif precision == 0 and recall == 0:
            F1Score = 0
        else:
            F1Score = 2*precision*recall/(precision + recall)
        # TODO !!! dummy = precision_sanity_check(F1Score, F1_SCORE)
        return F1Score    
    
    # score as defined in level selection (PosIn/AllIn)/(PosAll/All)
    def compute_pos_in_ratio_from_ppr(self, PPr, RP, All):
        # adapt PosInRatio (Lift) so that it will be between 0 and 1 (the higher the better accuracy)
        # TODO !!! what should be thedefault? does it require command line control?
        normalize_mlbt_score = False
        if normalize_mlbt_score: 
            RelPPr = PPr/(PPr+RP/All)
        else:
            RelPPr = PPr/(RP/All)
        # TODO !!! dummy = precision_sanity_check(RelPPr, Pos_IN_RATIO)
        return RelPPr

    # score as defined in level selection (PosIn/AllIn)/(PosAll/All)
    def compute_pos_in_ratio(self, TP, PP, RP, All, useNaN=False):
        PPr = self.compute_predictive_positive_rate(TP, PP, RP, useNaN)   
        RelPPr = self.compute_pos_in_ratio_from_ppr(PPr, RP, All)
        return RelPPr

    # NORM_POS_LR = (TP/(TP+FN) / [ TP/(TP+FN)+ FP/(FP+TN) ]
    def compute_normalized_plr_from_tpr(self, TPr, FP, TN):
        NormPLR = TPr/(TPr+(FP/(FP+TN)))
        # TODO !!! dummy = precision_sanity_check(NormPLR, NORM_POS_LR)
        return NormPLR

    # "color" score -- used to colour cells in the pair plots
    # NORM_POS_LR = (TP/(TP+FN) / [ TP/(TP+FN)+ FP/(FP+TN) ]
    # NORM_POS_LR = (TPr / [ TPr + FPr ]
    # (PosIn/PosAll) / [ (PosIn/PosAll)+(NegIn/NegdAll) ]
    def compute_normalized_positive_likelihood_ratio(self, TP,FN,FP,TN,useNaN=False):
        TPr = self.compute_true_positive_rate(TP,FN,useNaN)
        NormPLR = self.compute_normalized_plr_from_tpr(TPr,FP,TN)
        return NormPLR
    
    # original version of this function, currently in use, is compute_balanced_precision
    #def compute_pos_in_balanced(self, weight, TPr, NormPLR):
    #    assert weight >= 0 and weight <= 1
    #    PosInBalanced = weight * TPr + (1 - weight) * NormPLR
    #    return PosInBalanced

    # weighted average of precision and sensitivity
    # PosInBalanced = weight*PosInVsPosAll+(1-weight)*PosInVsAllIn = weight*TPr+(1-weight)*PPV
    def compute_balanced_precision(self, weight, TPr, ppv):
        assert weight >= 0 and weight <= 1
        PosInBalanced = weight * TPr + (1 - weight) * ppv
        return PosInBalanced