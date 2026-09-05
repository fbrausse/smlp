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

# Below are the tests for the DefaultRepresentativesSelectionAlgorithm
def run_tests():
    default_representatives_selection_select_features_that_correlate_strongly_with_other_features()

def default_representatives_selection_select_features_that_correlate_strongly_with_other_features():
    # arrange
    feat_df = pd.DataFrame({
        # F1, F2, F3 strongly correlate with each other (|corr| > 0.95)
        'F1': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'F2': [0.0, 1.05, 1.95, 3.02, 3.98, 5.01, 5.97, 7.03, 8.01, 8.99],
        'F3': [0.02, 0.98, 2.03, 2.97, 4.01, 4.99, 6.02, 6.98, 8.04, 9.01],
        # F4 does not correlate strongly with the others (|corr| < 0.95)
        'F4': [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
    })
    feat_names = ['F1', 'F2', 'F3', 'F4']

    sut = DefaultRepresentativesSelectionAlgorithm(logging.Logger(__name__), 0.95)

    # act
    actual = sut.select(feat_df, feat_names)

    # assert
    assert len(actual) == 2

    assert 'F4' in actual

    assert \
        ('F1' in actual and 'F2' not in actual and 'F3' not in actual) or \
        ('F2' in actual and 'F1' not in actual and 'F3' not in actual) or \
        ('F3' in actual and 'F1' not in actual and 'F2' not in actual)

    print("✅ Passed")