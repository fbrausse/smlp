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
        resp_name: str) -> tuple[pd.DataFrame, dict[str, str]]:
        self.logger.info(f"Forming single range features with {self.discretization_method.name()} discretization method")
        ranges_map = self.discretization_method.discretize(feat_df, feat_names, resp_df, resp_name)

        single_ranges_df = pd.DataFrame()
        single_range_feature_to_feature_map = {}
        for feat in feat_names:
            ranges = ranges_map[feat]
            for r in ranges:
                new_feat_name = f"{feat}_{r}"
                
                single_range_feature_to_feature_map[new_feat_name] = feat
                single_ranges_df[new_feat_name] = r.contains(feat_df[feat])

        # convert True to 1 and False to 0
        single_ranges_df = single_ranges_df.astype(int)

        return single_ranges_df, single_range_feature_to_feature_map
    
    def form_range_pairs(self, single_range_features_df: pd.DataFrame, single_range_feature_names: list[str], single_range_feature_to_feature_map: dict[str, str]) -> pd.DataFrame:
        consistent_singe_range_features = self._define_consistent_single_range_feature(single_range_features_df, single_range_feature_names)

        range_pairs_df = pd.DataFrame()
        range_pairs_to_features_map = {}
        while len(consistent_singe_range_features) > 0:
            single_range_feat_name = consistent_singe_range_features.pop(0)
            feat_name = single_range_feature_to_feature_map[single_range_feat_name]

            other_features_single_range_features = [
                other_feat_single_feat_range_name  \
                for other_feat_single_feat_range_name in consistent_singe_range_features \
                if single_range_feature_to_feature_map[other_feat_single_feat_range_name] != feat_name
            ]

            for other_feat_single_feat_range_name in other_features_single_range_features:
                new_feat_name = f"{single_range_feat_name}_{other_feat_single_feat_range_name}"
                range_pairs_df[new_feat_name] = single_range_features_df[single_range_feat_name] & single_range_features_df[other_feat_single_feat_range_name]
                range_pairs_to_features_map[new_feat_name] = [single_range_feat_name, other_feat_single_feat_range_name]

        # convert True to 1 and False to 0
        range_pairs_df = range_pairs_df.astype(int)

        return range_pairs_df, range_pairs_to_features_map
            
    # consistent single range feature is a single range feature that 
    # has at least one sample that is in the range and at least one sample that is not in the range
    # thus, the single range feature is consistent if it has at least one 1 and at least one 0        
    def _define_consistent_single_range_feature(self, single_range_features_df: pd.DataFrame, single_range_feature_names: list[str]) -> list[str]:
        consistent_singe_range_features = []

        for feat_name in single_range_feature_names:
            # if the single range feature has at least one 1 then the sum will be at least 1
            if single_range_features_df[feat_name].sum() > 0 and \
                single_range_features_df[feat_name].sum() < len(single_range_features_df): # if the sinle range feature has at least one 0 then the sum will be at most len(single_range_features_df) - 1
                consistent_singe_range_features.append(feat_name)
            
        return consistent_singe_range_features

# BELOW ARE THE TESTS FOR THE RANGE FEATURES FORMER CLASS
def run_tests():
    single_range_feature_should_contain_1_if_value_is_within_the_range_and_0_otherwise()
    range_pairs_should_contain_1_both_single_range_features_contain_1_and_0_otherwise()

class TestDiscretizationMethod(DiscretizationMethod):
    def discretize(self, feat_df: pd.DataFrame, feat_names: list[str], resp_df: pd.DataFrame, resp_name: str) -> dict[str, list[Range]]:
        return {
            'F1': [Range(1, 5)],
            'F2': [InverseRange(6, 8, 0, 10)],
        }

    def name(self) -> str:
        return "test"

def single_range_feature_should_contain_1_if_value_is_within_the_range_and_0_otherwise():
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
    result, map = sut.form_single_range_features(feat_df, feat_names, resp_df, resp_name)

    # assert
    assert result.equals(pd.DataFrame({
        'F1_[1, 5]': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        'F2_[8, 6]': [1, 1, 1, 1, 1, 0, 0, 0, 1, 1],
    }))

    assert map == {
        'F1_[1, 5]': 'F1',
        'F2_[8, 6]': 'F2',
    }

    print("✅ Passed")

def range_pairs_should_contain_1_both_single_range_features_contain_1_and_0_otherwise():
    # arrange
    single_range_features_df = pd.DataFrame({
        'F1_[1, 5]': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        'F2_[8, 6]': [1, 1, 1, 1, 1, 0, 0, 0, 1, 1],
        'F3_[2, 4]': [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        # non consistent single range feature, should be ignored
        'F4_[7, 9]': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    })
    single_range_feature_names = ['F1_[1, 5]', 'F2_[8, 6]', 'F3_[2, 4]']
    single_range_feature_to_feature_map = {
        'F1_[1, 5]': 'F1',
        'F2_[8, 6]': 'F2',
        'F3_[2, 4]': 'F3',
    }

    logger = logging.getLogger(__name__)
    sut = RangeFeaturesFormer(logger, TestDiscretizationMethod(logger, 1, 1))

    # act
    result, map = sut.form_range_pairs(single_range_features_df, single_range_feature_names, single_range_feature_to_feature_map)

    # assert
    assert result.equals(pd.DataFrame({
        'F1_[1, 5]_F2_[8, 6]': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        'F1_[1, 5]_F3_[2, 4]': [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        'F2_[8, 6]_F3_[2, 4]': [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    }))

    assert map == {
        'F1_[1, 5]_F2_[8, 6]': ['F1_[1, 5]', 'F2_[8, 6]'],
        'F1_[1, 5]_F3_[2, 4]': ['F1_[1, 5]', 'F3_[2, 4]'],
        'F2_[8, 6]_F3_[2, 4]': ['F2_[8, 6]', 'F3_[2, 4]'],
    }

    print("✅ Passed")