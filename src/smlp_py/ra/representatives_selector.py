from abc import ABC, abstractmethod

import logging
import pandas as pd

from scipy import stats
import random as r

class RepresentativesSelectionAlgorithm(ABC):
    def __init__(self, logger: logging.Logger, representatives_threshold: float):
        self.logger = logger
        self.representatives_threshold = representatives_threshold

    @abstractmethod
    def select(self, feat_df: pd.DataFrame, feat_names: list[str]) -> list[str]:
        pass

class DefaultRepresentativesSelectionAlgorithm(RepresentativesSelectionAlgorithm):
    def select(self, feat_df: pd.DataFrame, feat_names: list[str]) -> list[str]:
        self.logger.info(f"Selecting representatives with threshold: {self.representatives_threshold}")

        representatives = []
        for feat_name in feat_names:
            other_features = [f for f in feat_names if f != feat_name]
            
            if self._is_representative(feat_name, other_features, feat_df):
                representatives.append(feat_name)
                
        return representatives

    def _is_representative(self, feat_name: str, other_features: list[str], feat_df: pd.DataFrame) -> bool:
        for other_feature in other_features:
            res = stats.pearsonr(feat_df[feat_name].values, feat_df[other_feature].values)
            corr = res.statistic

            if corr < self.representatives_threshold:
                return False
        
        return True

class RandomRepresentativesSelectionAlgorirthm(RepresentativesSelectionAlgorithm):
    def select(self, feat_df: pd.DataFrame, feat_names: list[str]) -> list[str]:
        n = len(feat_names)
        return r.choices(feat_names, k = 4 if n > 4 else n)