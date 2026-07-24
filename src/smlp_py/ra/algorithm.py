from logging import Logger
import pandas as pd

from .representatives_selection import RepresentativesSelectionAlgorithm
from .range_features import RangeFeaturesFormer
from .ranking import RankingAlgorithm

class RaAlgorithm:
    def __init__(self, 
    range_features_former: RangeFeaturesFormer,
    representatives_selector: RepresentativesSelectionAlgorithm,
    ranking: RankingAlgorithm,
    logger: Logger):
        self.range_features_former = range_features_former
        self.representatives_selector = representatives_selector
        self.ranking = ranking
        self.logger = logger

    def run(
        self,
        feat_df: pd.DataFrame,
        feat_names: list[str],
        resp_df: pd.DataFrame,
        resp_name: str,
        top_ranking_features_count: int,
        top_final_features_count: int
    ) -> pd.DataFrame:
        self.logger.info(f"Starting the execution of the feature range analysis algorithm...")

        features_representatives = self.representatives_selector.select(feat_df, feat_names.copy())
        self.logger.info(f"Selected representatives are {features_representatives}")

        features_scores = self.ranking.rank(feat_df, features_representatives, resp_df, resp_name)
        
        features_selected = self._top(features_scores, top_ranking_features_count)
        self.logger.info(f"Selected features are {features_selected}")

        single_range_features_df, single_range_feature_to_feature_map = self.range_features_former.form_single_range_features(feat_df, features_selected.copy(), resp_df, resp_name)
        self.logger.info(f"Formed {len(single_range_features_df.columns)} single range features")

        single_range_features_representatives = self.representatives_selector.select(single_range_features_df, single_range_features_df.columns)
        self.logger.info(f"Selected representatives from single range features are {single_range_features_representatives}")

        single_range_features_scores = self.ranking.rank(single_range_features_df, single_range_features_representatives, resp_df, resp_name)

        single_range_features_selected = self._top(single_range_features_scores, top_ranking_features_count)
        self.logger.info(f"Selected single range features are {single_range_features_selected}")

        range_pairs_df, range_pairs_to_features_map = self.range_features_former.form_range_pairs(single_range_features_df, single_range_features_selected.copy(), single_range_feature_to_feature_map)
        self.logger.info(f"Formed {len(range_pairs_df.columns)} range pairs")

        range_pairs_representatives = self.representatives_selector.select(range_pairs_df, range_pairs_df.columns)
        self.logger.info(f"Selected representatives from range pairs are {range_pairs_representatives}")

        range_pairs_scores = self.ranking.rank(range_pairs_df, range_pairs_representatives, resp_df, resp_name)

        range_pairs_selected = self._top(range_pairs_scores, top_ranking_features_count)
        self.logger.info(f"Selected range pairs are {range_pairs_selected}")

        range_triplets_df = self.range_features_former.form_range_triplets(range_pairs_df, range_pairs_selected.copy(), range_pairs_to_features_map, single_range_features_df, single_range_features_selected.copy(), single_range_feature_to_feature_map)
        self.logger.info(f"Formed {len(range_triplets_df.columns)} range triplets")

        range_triplets_representatives = self.representatives_selector.select(range_triplets_df, range_triplets_df.columns)
        self.logger.info(f"Selected representatives from range triplets are {range_triplets_representatives}")

        range_triplets_scores = self.ranking.rank(range_triplets_df, range_triplets_representatives, resp_df, resp_name)

        range_triplets_selected = self._top(range_triplets_scores, top_ranking_features_count)
        self.logger.info(f"Selected range triplets are {range_triplets_selected}")

        return features_selected[:top_final_features_count] + single_range_features_selected[:top_final_features_count] + range_pairs_selected[:top_final_features_count] + range_triplets_selected[:top_final_features_count]

    def _top(self, features_scores: dict[str, float], top_features_count: int) -> list[str]:
        return list(map(lambda x: x[0], sorted(features_scores.items(), key=lambda x: x[1], reverse=True)[:top_features_count]))