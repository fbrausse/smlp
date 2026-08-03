from .smlp_correlations import SmlpCorrelations, PEARSON, SPEARMAN

from .ra.ranking import RankingAlgorithm

import logging
import pandas as pd

class SmlpRankingAlgorithm(RankingAlgorithm):
    def __init__(self, logger: logging.Logger, correlations: SmlpCorrelations):
        self.logger = logger
        self.correlations = correlations

    def rank(self, feat_df: pd.DataFrame, feat_names: list[str], resp_df: pd.DataFrame, resp_name: str) -> pd.DataFrame:
        self.logger.info("Starting the default ranking")

        # do not consider features that have only one unique value.
        # this might happen when range features are passed, because they contain the binary values.
        # And it might be the case that values are either all 0 or all 1 in the range feature.
        relevant_features = []
        irrelevant_features = []
        for feat in feat_names:
            if len(feat_df[feat].unique()) >= 2:
                relevant_features.append(feat)
                continue

            irrelevant_features.append(feat)
            
        # this flag shows whether or not the range features were passed to the function. 
        # it is important to compute the value of the discretize_numeric_features flag for the 
        # ensemble_features_single_response function
        range_features = all(len(feat_df[feat].unique()) == 2 for feat in feat_names)

        fs_summary_df, _ = self.correlations.ensemble_features_single_response(
            feat_df[relevant_features],
            resp_df[resp_name],
            None,
            None,
            'features',
            'uniform',
            10,
            True, 
            'category',
            not range_features, 
            'normalized',
            True, 
            len(relevant_features), 
            [PEARSON, SPEARMAN])
        
        result_df = pd.DataFrame()
        result_df['Feature'] = fs_summary_df['important_features']
        result_df['Score'] = fs_summary_df['ranking']

        # append the irrelevant features with score 0 and return the result
        irrelevant_df = pd.DataFrame({
            'Feature': irrelevant_features,
            'Score': [0 for _ in irrelevant_features]
        }, columns=['Feature', 'Score'])

        result_df = pd.concat([result_df, irrelevant_df], ignore_index=True)
        
        return result_df