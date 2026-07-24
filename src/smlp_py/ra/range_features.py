import logging
import pandas as pd

from .discretization import DiscretizationMethod
from .range import Range, InverseRange

class RangeFeaturesFormer:
    def __init__(self, logger: logging.Logger, discretization_method: DiscretizationMethod):
        self.logger = logger
        self.discretization_method = discretization_method

    def form_single_range_features(
        self, 
        feat_df: pd.DataFrame, 
        feat_names: list[str], 
        resp_df: pd.DataFrame, 
        resp_name: str) -> pd.DataFrame:
        self.logger.info(f"Forming single range features with {self.discretization_method.name()} discretization method")
        ranges_map = self.discretization_method.discretize(feat_df, feat_names, resp_df, resp_name)

        single_ranges_df = pd.DataFrame()
        for feat in feat_names:
            ranges = ranges_map[feat]
            for r in ranges:
                new_feat_name = f"{feat}_{r}"
                single_ranges_df[new_feat_name] = r.contains(feat_df[feat])

        # convert True to 1 and False to 0
        single_ranges_df = single_ranges_df.astype(int)

        return single_ranges_df

# BELOW ARE THE TESTS FOR THE RANGE FEATURES FORMER CLASS
def run_tests():
    single_range_feature_should_contain_true_if_value_is_within_the_range_and_false_otherwise()

class TestDiscretizationMethod(DiscretizationMethod):
    def discretize(self, feat_df: pd.DataFrame, feat_names: list[str], resp_df: pd.DataFrame, resp_name: str) -> dict[str, list[Range]]:
        return {
            'F1': [Range(1, 5)],
            'F2': [InverseRange(6, 8, 0, 10)],
        }

    def name(self) -> str:
        return "test"

def single_range_feature_should_contain_true_if_value_is_within_the_range_and_false_otherwise():
    # arrange
    feat_df = pd.DataFrame({
        'F1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'F2': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    })
    feat_names = ['F1', 'F2']
    resp_df = pd.DataFrame({
        'R': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    })
    resp_name = 'R'

    logger = logging.getLogger(__name__)
    sut = RangeFeaturesFormer(logger, TestDiscretizationMethod(logger, 1, 1))

    # act
    result = sut.form_single_range_features(feat_df, feat_names, resp_df, resp_name)

    # assert
    assert result.equals(pd.DataFrame({
        'F1_1_5': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        'F2_8_6': [1, 1, 1, 1, 1, 0, 0, 0, 1, 1],
    }))

    print("✅ Passed")