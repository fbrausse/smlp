from abc import ABC, abstractmethod

import logging
import pandas as pd
from scipy import stats
import random as r

from .correlation_method import CorrelationMethod

class RepresentativesSelectionAlgorithm(ABC):
    def __init__(self, logger: logging.Logger, representatives_threshold: float):
        self.logger = logger
        self.representatives_threshold = representatives_threshold

    @abstractmethod
    def select(self, feat_df: pd.DataFrame, feat_names: list[str]) -> list[str]:
        pass

class DefaultRepresentativesSelectionAlgorithm(RepresentativesSelectionAlgorithm):
    def __init__(self, logger: logging.Logger, representatives_threshold: float, correlation_method: CorrelationMethod):
        super().__init__(logger, representatives_threshold)
        self.correlation_method = correlation_method

    def select(self, feat_df: pd.DataFrame, feat_names: list[str]) -> list[str]:
        self.logger.info(f"Starting default representatives selection with threshold {self.representatives_threshold}")

        representatives = []
        while len(feat_names) > 0:
            next_feat_name = feat_names.pop(0)
            subset = self._select_subset(next_feat_name, feat_names, feat_df)
            representatives.append(next_feat_name)    
            feat_names = [f for f in feat_names if f not in subset]
                
        return representatives

    def _select_subset(self, feat_name: str, feat_names: list[str], feat_df: pd.DataFrame) -> bool:
        subset = [feat_name]
        for other_feature in feat_names:
            if other_feature == feat_name:
                continue
            
            corr = self.correlation_method.compute_correlation(feat_df[feat_name].values, feat_df[other_feature].values)
            
            if corr >= self.representatives_threshold:
                subset.append(other_feature)
        
        return subset

class RandomRepresentativesSelectionAlgorirthm(RepresentativesSelectionAlgorithm):
    def select(self, feat_df: pd.DataFrame, feat_names: list[str]) -> list[str]:
        k = 20
        n = len(feat_names)
        
        return r.sample(feat_names, k = k if n > k else n)

# BELOW ARE THE TESTS FOR THE DEFAULT REPRESENTATIVES SELECTION ALGORITHM
def run_tests():
    default_representatives_selection_algorithm_should_select_representatives_based_on_correlation_threshold()

class TestCorrelationMethod(CorrelationMethod):
    def compute_correlation(self, var1, var2) -> float:
        v1 = var1[0]
        v2 = var2[0]

        if (v1 == 1 and v2 in [2, 5, 6]) or \
            (v1 == 3 and v2 in [4, 7]):
            return 0.95
        
        return 0.94


def default_representatives_selection_algorithm_should_select_representatives_based_on_correlation_threshold():
    # arrange
    df = pd.DataFrame({
        'F1': [1], # representative
        'F2': [2], # subset of 1
        'F3': [3], # representative
        'F4': [4], # subset of 3
        'F5': [5], # subset of 1
        'F6': [6], # subset of 1
        'F7': [7], # subset of 3
        'F8': [8] # representative
    })
    feat_names = list(df.columns)

    representatives_threshold = 0.95
    correlation_method = TestCorrelationMethod()

    logger = logging.getLogger(__name__)
    sut = DefaultRepresentativesSelectionAlgorithm(logger, representatives_threshold, correlation_method)

    # act
    actual = sut.select(df, feat_names)

    # assert
    expected = ['F1', 'F3', 'F8']

    assert len(actual) == len(expected)

    for actual_feat_name, expected_feat_name in zip(actual, expected):
        assert actual_feat_name == expected_feat_name

    print("✅ Passed")