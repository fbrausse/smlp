from logging import Logger
import pandas as pd

from .representatives_selection import RepresentativesSelectionAlgorithm
from .discretization import DiscretizationAlgorithm
from .ranking import RankingAlgorithm

class RaAlgorithm:
    def __init__(self, 
    discretization: DiscretizationAlgorithm,
    representatives_selector: RepresentativesSelectionAlgorithm,
    ranking: RankingAlgorithm,
    logger: Logger):
        self.discretization = discretization
        self.representatives_selector = representatives_selector
        self.ranking = ranking
        self.logger = logger

    def run(
        self,
        feat_df: pd.DataFrame,
        feat_names: list[str],
        resp_df: pd.DataFrame,
        resp_name: str,
        top_features_count: int
    ) -> pd.DataFrame:
        self.logger.info(f"Starting the execution of the feature range analysis algorithm...")

        features_representatives = self.representatives_selector.select(feat_df, feat_names)
        self.logger.info(f"Selected representatives are {features_representatives}")

        features_scores = self.ranking.rank(feat_df, features_representatives, resp_df, resp_name)
        
        features_selected = self._top(features_scores, top_features_count)
        self.logger.info(f"Selected features are {features_selected}")

    def _top(self, features_scores: dict[str, float], top_features_count: int) -> list[str]:
        return sorted(features_scores.items(), key=lambda x: x[1], reverse=True)[:top_features_count]