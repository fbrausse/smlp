from abc import ABC, abstractmethod

import logging
import pandas as pd
import random as r
import numpy as np

class RepresentativesSelectionAlgorithm(ABC):
    def __init__(self, logger: logging.Logger, representatives_threshold: float):
        self.logger = logger
        self.representatives_threshold = representatives_threshold

    @abstractmethod
    def select(self, feat_df: pd.DataFrame, feat_names: list[str]) -> list[str]:
        pass

class DefaultRepresentativesSelectionAlgorithm(RepresentativesSelectionAlgorithm):
    def __init__(self, logger: logging.Logger, representatives_threshold: float):
        super().__init__(logger, representatives_threshold)

    def select(self, feat_df: pd.DataFrame, feat_names: list[str]) -> list[str]:
        self.logger.info(f"Starting default representatives selection with threshold {self.representatives_threshold}")

        corr_matrix = self._compute_corr_matrix(feat_df, feat_names)

        representatives = []
        remaining = set(feat_names)
        while remaining:
            next_feat_name = remaining.pop()
            representatives.append(next_feat_name)

            # select the stronly correlating features with the next_feat_name
            corr = corr_matrix.loc[next_feat_name, list(remaining)]
            subset = set(corr[corr >= self.representatives_threshold].index)
            
            # subset is a set of features that next_feat_name represents
            # therefore they can be removed from the features set and be rapleced 
            # by their representative
            remaining -= subset
                
        return representatives

    def _compute_corr_matrix(self, feat_df: pd.DataFrame, feat_names: list[str]):
        corr_matrix = np.corrcoef(feat_df[feat_names].to_numpy(dtype=np.float64), rowvar=False)
        corr_matrix = np.abs(corr_matrix)

        corr_matrix = pd.DataFrame(
            corr_matrix,
            index=feat_names,
            columns=feat_names
        )

        return corr_matrix

class RandomRepresentativesSelectionAlgorirthm(RepresentativesSelectionAlgorithm):
    def select(self, feat_df: pd.DataFrame, feat_names: list[str]) -> list[str]:
        k = 20
        n = len(feat_names)
        
        return r.sample(feat_names, k = k if n > k else n)

# TODO: Add unit tests for the DefaultRepresentativesSelectionAlgorithm