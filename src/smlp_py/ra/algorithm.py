from logging import Logger
import pandas as pd

from .representatives_selection import RepresentativesSelectionAlgorithm
from .range_features import RangeFeaturesFormer
from .ranking import RankingAlgorithm
from .basis import BasisAlgorithm
from .quality import Quality

class RaAlgorithm:
    def __init__(self, 
    range_features_former: RangeFeaturesFormer,
    representatives_selector: RepresentativesSelectionAlgorithm,
    ranking: RankingAlgorithm,
    basis: BasisAlgorithm,
    quality: Quality,
    logger: Logger):
        self.range_features_former = range_features_former
        self.representatives_selector = representatives_selector
        self.ranking = ranking
        self.basis = basis
        self.quality = quality
        self.logger = logger

    def run(
        self,
        feat_df: pd.DataFrame,
        feat_names: list[str],
        resp_df: pd.DataFrame,
        resp_name: str,
        top_ranking_features_count: int,
        top_final_features_count: int,
        quality_function: str
    ) -> pd.DataFrame:
        self.logger.info(f"Starting the execution of the feature range analysis algorithm...")

        # Select features
        features_representatives = self.representatives_selector.select(feat_df, feat_names.copy())
        self.logger.info(f"Selected representatives are {features_representatives}")

        features_corr = self.ranking.rank(feat_df, features_representatives, resp_df, resp_name)
        features_basis = self.basis.rank(feat_df, features_representatives, resp_df, resp_name)

        self._normalize_score(features_corr)
        self._normalize_score(features_basis)

        features_selected_df =  self._concat(self._top(features_corr, top_ranking_features_count), self._top(features_basis, top_ranking_features_count))
        features_selected = list(features_selected_df['Feature'])
        self.logger.info(f"Selected features are {features_selected}")

        # Form single range features
        single_range_features_df, single_range_feature_to_feature_map = self.range_features_former.form_single_range_features(feat_df, features_selected.copy(), resp_df, resp_name)
        self.logger.info(f"Formed {len(single_range_features_df.columns)} single range features")
        
        # Select single range features
        single_range_features_representatives = self.representatives_selector.select(single_range_features_df, list(single_range_features_df.columns))
        self.logger.info(f"Selected representatives from single range features are {single_range_features_representatives}")

        single_range_features_corr = self.ranking.rank(single_range_features_df, single_range_features_representatives, resp_df, resp_name)
        single_range_features_basis = self.basis.rank(single_range_features_df, single_range_features_representatives, resp_df, resp_name)
        single_range_features_quality = self.quality.rank(single_range_features_representatives, quality_function, single_range_features_df, resp_df, resp_name)

        self._normalize_score(single_range_features_corr)
        self._normalize_score(single_range_features_basis)
        self._normalize_score(single_range_features_quality)

        single_range_features_selected_df = self._concat(
            self._top(single_range_features_quality, top_ranking_features_count),
            self._concat(
                self._top(single_range_features_corr, top_ranking_features_count), 
                self._top(single_range_features_basis, top_ranking_features_count)
            )
        )
        single_range_features_selected = list(single_range_features_selected_df['Feature'])
        self.logger.info(f"Selected single range features are {single_range_features_selected}")

        # Form range pairs
        range_pairs_df, range_pairs_to_features_map = self.range_features_former.form_range_pairs(single_range_features_df, single_range_features_selected.copy(), single_range_feature_to_feature_map)
        self.logger.info(f"Formed {len(range_pairs_df.columns)} range pairs")

        # Select range pairs
        range_pairs_representatives = self.representatives_selector.select(range_pairs_df, list(range_pairs_df.columns))
        self.logger.info(f"Selected representatives from range pairs are {range_pairs_representatives}")

        range_pairs_corr = self.ranking.rank(range_pairs_df, range_pairs_representatives, resp_df, resp_name)
        range_pairs_basis = self.basis.rank(range_pairs_df, range_pairs_representatives, resp_df, resp_name)
        range_pairs_quality = self.quality.rank(range_pairs_representatives, quality_function, range_pairs_df, resp_df, resp_name)

        self._normalize_score(range_pairs_corr)
        self._normalize_score(range_pairs_basis)
        self._normalize_score(range_pairs_quality)

        range_pairs_selected_df = self._concat(
            self._top(range_pairs_quality, top_ranking_features_count),
            self._concat(
                self._top(range_pairs_corr, top_ranking_features_count),
                self._top(range_pairs_basis, top_ranking_features_count)
            )
        )
        range_pairs_selected = list(range_pairs_selected_df['Feature'])
        self.logger.info(f"Selected range pairs are {range_pairs_selected}")

        # Form range triplets
        range_triplets_df = self.range_features_former.form_range_triplets(range_pairs_df, range_pairs_selected.copy(), range_pairs_to_features_map, single_range_features_df, single_range_features_selected.copy(), single_range_feature_to_feature_map)
        self.logger.info(f"Formed {len(range_triplets_df.columns)} range triplets")

        # Select range triplets
        range_triplets_representatives = self.representatives_selector.select(range_triplets_df, list(range_triplets_df.columns))
        self.logger.info(f"Selected representatives from range triplets are {range_triplets_representatives}")

        range_triplets_corr = self.ranking.rank(range_triplets_df, range_triplets_representatives, resp_df, resp_name)
        range_triplets_basis = self.basis.rank(range_triplets_df, range_triplets_representatives, resp_df, resp_name)
        range_triplets_quality = self.quality.rank(range_triplets_representatives, quality_function, range_triplets_df, resp_df, resp_name)

        self._normalize_score(range_triplets_corr)
        self._normalize_score(range_triplets_basis)
        self._normalize_score(range_triplets_quality)

        range_triplets_selected_df = self._concat(
            self._top(range_triplets_quality, top_ranking_features_count),
            self._concat(
                self._top(range_triplets_corr, top_ranking_features_count), 
                self._top(range_triplets_basis, top_ranking_features_count)
            )
        )
        range_triplets_selected = list(range_triplets_selected_df['Feature'])
        self.logger.info(f"Selected range triplets are {range_triplets_selected}")

        return self._concat(
            self._top(range_triplets_selected_df, top_final_features_count), 
            self._concat(
                self._top(range_pairs_selected_df, top_final_features_count), 
                self._concat(
                    self._top(single_range_features_selected_df, top_final_features_count), 
                    self._top(features_selected_df, top_final_features_count)
                )
            )
        ).sort_values(by='Score', ascending=False)
    
    def _top(self, df: pd.DataFrame, top_n: int) -> pd.DataFrame:
        return df \
            .sort_values(by='Score', ascending=False) \
            .head(top_n)

    def _concat(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        df = pd.concat([df1, df2], ignore_index=True)

        # if there are duplicates, choose the ones which have higher values
        return df.groupby('Feature', as_index=False).max()

    def _normalize_score(self, df: pd.DataFrame):
        df['Score'] = (df['Score'] - df['Score'].mean()) / df['Score'].std()