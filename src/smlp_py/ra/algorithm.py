from logging import Logger
import logging
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
        top_final_features_count: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
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

        single_range_quality_metrics = self.quality.compute_metrics(single_range_features_representatives, single_range_features_df, resp_df, resp_name)

        single_range_features_corr = self.ranking.rank(single_range_features_df, single_range_features_representatives, resp_df, resp_name)
        single_range_features_basis = self.basis.rank(single_range_features_df, single_range_features_representatives, resp_df, resp_name)
        single_range_features_quality = self.quality.rank(single_range_quality_metrics)

        self._normalize_score(single_range_features_corr)
        self._normalize_score(single_range_features_basis)
        self._normalize_score(single_range_features_quality)

        # the call of the _add_quality_metrics is added solely to provide quality metrics for all selected features
        single_range_features_selected_df = self._add_quality_metrics(
            self._concat(
                self._top(single_range_features_quality, top_ranking_features_count),
                self._concat(
                    self._top(single_range_features_corr, top_ranking_features_count), 
                    self._top(single_range_features_basis, top_ranking_features_count)
                )
            ),
            single_range_quality_metrics
        )
        single_range_features_selected = list(single_range_features_selected_df['Feature'])
        self.logger.info(f"Selected single range features are {single_range_features_selected}")

        # Form range pairs
        range_pairs_df, range_pairs_to_features_map = self.range_features_former.form_range_pairs(single_range_features_df, single_range_features_selected.copy(), single_range_feature_to_feature_map)
        self.logger.info(f"Formed {len(range_pairs_df.columns)} range pairs")

        # Select range pairs
        range_pairs_representatives = self.representatives_selector.select(range_pairs_df, list(range_pairs_df.columns))
        self.logger.info(f"Selected representatives from range pairs are {range_pairs_representatives}")

        range_pairs_quality_metrics = self.quality.compute_metrics(range_pairs_representatives, range_pairs_df, resp_df, resp_name)

        range_pairs_corr = self.ranking.rank(range_pairs_df, range_pairs_representatives, resp_df, resp_name)
        range_pairs_basis = self.basis.rank(range_pairs_df, range_pairs_representatives, resp_df, resp_name)
        range_pairs_quality = self.quality.rank(range_pairs_quality_metrics)

        self._normalize_score(range_pairs_corr)
        self._normalize_score(range_pairs_basis)
        self._normalize_score(range_pairs_quality)

        # the call of the _add_quality_metrics is added solely to provide quality metrics for all selected features
        range_pairs_selected_df = self._add_quality_metrics(
            self._concat(
                self._top(range_pairs_quality, top_ranking_features_count),
                self._concat(
                    self._top(range_pairs_corr, top_ranking_features_count),
                    self._top(range_pairs_basis, top_ranking_features_count)
                )
            ),
            range_pairs_quality_metrics
        )
        range_pairs_selected = list(range_pairs_selected_df['Feature'])
        self.logger.info(f"Selected range pairs are {range_pairs_selected}")

        # Form range triplets
        range_triplets_df, range_triplets_to_features_map = self.range_features_former.form_range_triplets(range_pairs_df, range_pairs_selected.copy(), range_pairs_to_features_map, single_range_features_df, single_range_features_selected.copy(), single_range_feature_to_feature_map)
        self.logger.info(f"Formed {len(range_triplets_df.columns)} range triplets")

        # Select range triplets
        range_triplets_representatives = self.representatives_selector.select(range_triplets_df, list(range_triplets_df.columns))
        self.logger.info(f"Selected representatives from range triplets are {range_triplets_representatives}")

        range_triplets_quality_metrics = self.quality.compute_metrics(range_triplets_representatives, range_triplets_df, resp_df, resp_name)

        range_triplets_corr = self.ranking.rank(range_triplets_df, range_triplets_representatives, resp_df, resp_name)
        range_triplets_basis = self.basis.rank(range_triplets_df, range_triplets_representatives, resp_df, resp_name)
        range_triplets_quality = self.quality.rank(range_triplets_quality_metrics)

        self._normalize_score(range_triplets_corr)
        self._normalize_score(range_triplets_basis)
        self._normalize_score(range_triplets_quality)

        # the call of the _add_quality_metrics is added solely to provide quality metrics for all selected features
        range_triplets_selected_df = self._add_quality_metrics(
            self._concat(
                self._top(range_triplets_quality, top_ranking_features_count),
                self._concat(
                    self._top(range_triplets_corr, top_ranking_features_count), 
                    self._top(range_triplets_basis, top_ranking_features_count)
                )
            ),
            range_triplets_quality_metrics
        )
        range_triplets_selected = list(range_triplets_selected_df['Feature'])
        self.logger.info(f"Selected range triplets are {range_triplets_selected}")

        result_summary_df = self._concat(
            self._top(range_triplets_selected_df, top_final_features_count), 
            self._concat(
                self._top(range_pairs_selected_df, top_final_features_count), 
                self._concat(
                    self._top(single_range_features_selected_df, top_final_features_count), 
                    self._top(features_selected_df, top_final_features_count)
                )
            )
        ).sort_values(by='Score', ascending=False)

        # this function call is added to have the response type similar to the SubgroupDiscovery's result in smlp_subgroups.py file
        self._add_extensive_features_metrics(
            result_summary_df,
            feat_df,
            single_range_feature_to_feature_map,
            range_pairs_to_features_map,
            range_triplets_to_features_map
        )

        result_df = self._form_result_dataframe(
            result_summary_df,
            feat_df,
            single_range_features_df,
            range_pairs_df,
            range_triplets_df,
            resp_df,
            resp_name
        )

        return result_df, result_summary_df
    
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

    def _form_result_dataframe(
        self, 
        result_summary_df: pd.DataFrame,
        feat_df: pd.DataFrame,
        single_ranges_df: pd.DataFrame,
        range_pairs_df: pd.DataFrame,
        range_triplets_df: pd.DataFrame,
        resp_df: pd.DataFrame,
        resp_name: str):
        single_ranges = []
        range_pairs = []
        range_triplets = []
        features = []

        result = {}
        for row in result_summary_df.itertuples():
            feature = row.Feature
            
            if row.Feature_1 and row.Feature_2 and row.Feature_3:
                range_triplets.append(feature)
                continue

            if row.Feature_1 and row.Feature_2:
                range_pairs.append(feature)
                continue

            if row.Feature_1:
                single_ranges.append(feature)
                continue
            
            features.append(feature)

        for feature in features:
            result[feature] = feat_df[feature]
        
        for feature in single_ranges:
            result[feature] = single_ranges_df[feature]

        for feature in range_pairs:
            result[feature] = range_pairs_df[feature]
        
        for feature in range_triplets:
            result[feature] = range_triplets_df[feature]

        result[resp_name] = resp_df[resp_name]

        return pd.DataFrame(result, columns=list(result.keys()))

    def _add_quality_metrics(self, features_df: pd.DataFrame, quality_metrics_df: pd.DataFrame):
        selected_features = features_df['Feature']
        selected_quality_metrics_df = quality_metrics_df[quality_metrics_df['Feature'].isin(selected_features)]

        return features_df.merge(selected_quality_metrics_df, on='Feature', how='left')

    def _add_extensive_features_metrics(
        self, 
        result_df: pd.DataFrame, 
        feat_df: pd.DataFrame, 
        single_ranges_map: dict, 
        range_pairs_map: dict, 
        range_triplets_map: dict):
        metrics = {
            'Feature_1': [],
            'Range_start_1': [],
            'Range_end_1': [],
            'Min_1': [],
            'Max_1': [],
            'Mean_1': [],
            'Std_1': [],
            'Feature_2': [],
            'Range_start_2': [],
            'Range_end_2': [],
            'Min_2': [],
            'Max_2': [],
            'Mean_2': [],
            'Std_2': [],
            'Feature_3': [],
            'Range_start_3': [],
            'Range_end_3': [],
            'Min_3': [],
            'Max_3': [],
            'Mean_3': [],
            'Std_3': [],
        }
        for row in result_df.itertuples():
            feature = row.Feature

            is_range_triplet = feature in range_triplets_map.keys()
            if is_range_triplet:
                single_ranges = range_triplets_map[feature]

                for i, single_range in enumerate(single_ranges):
                    self._add_feature_extensive_metrics(feat_df, single_ranges_map[single_range], i + 1, metrics)
                
                continue

            is_range_pair = feature in range_pairs_map.keys()
            if is_range_pair: 
                single_ranges = range_pairs_map[feature]

                for i, single_range in enumerate(single_ranges):
                    self._add_feature_extensive_metrics(feat_df, single_ranges_map[single_range], i + 1, metrics)
                
                self._add_empty_feature_extensive_metrics(3, metrics)
                continue

            is_single_range_feature = feature in single_ranges_map.keys()
            if is_single_range_feature:
                self._add_feature_extensive_metrics(feat_df, single_ranges_map[feature], 1, metrics)
                self._add_empty_feature_extensive_metrics(2, metrics)
                self._add_empty_feature_extensive_metrics(3, metrics)
                continue

            self._add_empty_feature_extensive_metrics(1, metrics)
            self._add_empty_feature_extensive_metrics(2, metrics)
            self._add_empty_feature_extensive_metrics(3, metrics)

        # append metrics to the result_df
        for key in metrics.keys():
            result_df[key] = metrics[key]

    def _add_feature_extensive_metrics(
        self, 
        feat_df: pd.DataFrame,
        feat: dict,
        feat_count: int,
        metrics: dict
    ):          
        feat_series = feat_df[feat['name']]

        metrics[f'Feature_{feat_count}'].append(feat['name'])
        metrics[f'Range_start_{feat_count}'].append(feat['range_start'])
        metrics[f'Range_end_{feat_count}'].append(feat['range_end'])
        metrics[f'Min_{feat_count}'].append(feat_series.min())
        metrics[f'Max_{feat_count}'].append(feat_series.max())
        metrics[f'Mean_{feat_count}'].append(feat_series.mean())
        metrics[f'Std_{feat_count}'].append(feat_series.std())
    
    def _add_empty_feature_extensive_metrics(self, feat_count: int, metrics: {}):
        metrics[f'Feature_{feat_count}'].append(None)
        metrics[f'Range_start_{feat_count}'].append(None)
        metrics[f'Range_end_{feat_count}'].append(None)
        metrics[f'Min_{feat_count}'].append(None)
        metrics[f'Max_{feat_count}'].append(None)
        metrics[f'Mean_{feat_count}'].append(None)
        metrics[f'Std_{feat_count}'].append(None)