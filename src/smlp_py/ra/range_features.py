from .discretization import DiscretizationMethod
from .range import Range, InverseRange

import logging
import pandas as pd

class RangeFeaturesFormer:
    def __init__(self, logger: logging.Logger, discretization_method: DiscretizationMethod):
        self.logger = logger
        self.discretization_method = discretization_method

    def form_single_range_features(
        self, 
        feat_df: pd.DataFrame, 
        feat_names: list[str], 
        resp_df: pd.DataFrame, 
        resp_name: str) -> tuple[pd.DataFrame, dict]:
        self.logger.info(f"Forming single range features with {self.discretization_method.name()} discretization method")
        ranges_map = self.discretization_method.discretize(feat_df, feat_names, resp_df, resp_name)

        single_ranges_df = pd.DataFrame()
        single_range_feature_to_feature_map = {}
        for feat in feat_names:
            ranges = ranges_map[feat]
            for r in ranges:
                new_feat_name = f"{feat}_{r}"
                
                single_range_feature_to_feature_map[new_feat_name] = {
                    'name': feat,
                    'range_start': r.start,
                    'range_end': r.end
                }
                single_ranges_df[new_feat_name] = r.contains(feat_df[feat])

        # convert True to 1 and False to 0
        single_ranges_df = single_ranges_df.astype(int)

        return single_ranges_df, single_range_feature_to_feature_map
    
    def form_range_pairs(self, single_range_features_df: pd.DataFrame, single_range_feature_names: list[str], single_range_feature_to_feature_map: dict[str, str]) -> tuple[pd.DataFrame, dict]:
        range_pairs_df = pd.DataFrame()
        range_pairs_to_features_map = {}
        while len(single_range_feature_names) > 0:
            single_range_feat_name = single_range_feature_names.pop(0)
            feat = single_range_feature_to_feature_map[single_range_feat_name]

            other_features_single_range_features = [
                other_feat_single_feat_range_name \
                for other_feat_single_feat_range_name in single_range_feature_names \
                if single_range_feature_to_feature_map[other_feat_single_feat_range_name]['name'] != feat['name']
            ]

            for other_feat_single_feat_range_name in other_features_single_range_features:
                new_feat_name = f"{single_range_feat_name}_{other_feat_single_feat_range_name}"
                range_pairs_df[new_feat_name] = single_range_features_df[single_range_feat_name] & single_range_features_df[other_feat_single_feat_range_name]
                range_pairs_to_features_map[new_feat_name] = [single_range_feat_name, other_feat_single_feat_range_name]

        # convert True to 1 and False to 0
        range_pairs_df = range_pairs_df.astype(int)

        # remove inconsistent range pairs
        for range_pair_name in range_pairs_df.columns:
            if not self._is_consistent_range_feature(range_pairs_df, range_pair_name):
                range_pairs_df.drop(range_pair_name, axis=1, inplace=True)
                range_pairs_to_features_map.pop(range_pair_name)

        return range_pairs_df, range_pairs_to_features_map

    def form_range_triplets(
        self, 
        range_pairs_df: pd.DataFrame, 
        range_pairs_names: list[str], 
        range_pairs_to_features_map: dict[str, list[str]],
        single_range_features_df: pd.DataFrame,
        single_range_feature_names: list[str],
        single_range_feature_to_feature_map: dict[str, str],
        ) -> tuple[pd.DataFrame, dict]:
        range_triplets_df = pd.DataFrame()
        range_triplets_to_features_map  = {}
        while len(range_pairs_names) > 0:
            range_pair_name = range_pairs_names.pop(0)
            range_pair_features = range_pairs_to_features_map[range_pair_name]
            range_pair_features_names = [single_range_feature_to_feature_map[feat_name] for feat_name in range_pair_features]

            other_features_single_range_features = [
                other_feat_single_feat_range_name  \
                for other_feat_single_feat_range_name in single_range_feature_names \
                if single_range_feature_to_feature_map[other_feat_single_feat_range_name] not in range_pair_features_names
            ]

            for other_feat_single_feat_range_name in other_features_single_range_features:
                new_feat_name = f"{range_pair_name}_{other_feat_single_feat_range_name}"
                range_triplets_df[new_feat_name] = range_pairs_df[range_pair_name] & single_range_features_df[other_feat_single_feat_range_name]
                range_triplets_to_features_map[new_feat_name] = [
                    range_pairs_to_features_map[range_pair_name][0],
                    range_pairs_to_features_map[range_pair_name][1],
                    other_feat_single_feat_range_name
                ]

        # convert True to 1 and False to 0
        range_triplets_df = range_triplets_df.astype(int)

        # remove inconsistent range triplets
        for range_triplet_name in range_triplets_df.columns:
            if not self._is_consistent_range_feature(range_triplets_df, range_triplet_name):
                range_triplets_df.drop(range_triplet_name, axis=1, inplace=True)
                range_triplets_to_features_map.pop(range_triplet_name)

        return range_triplets_df, range_triplets_to_features_map
            
    # consistent range feature is a range feature that 
    # has at least one sample that is in the range and at least one sample that is not in the range
    # thus, the range feature is consistent if it has at least one 1 and at least one 0        
    def _is_consistent_range_feature(self, range_features_df: pd.DataFrame, range_feature_name: str) -> bool:
        # if the range feature has at least one 1 then the sum will be at least 1
        # if the range feature has at least one 0 then the sum will be at most length - 1
        return \
            range_features_df[range_feature_name].sum() > 0 and \
            range_features_df[range_feature_name].sum() < len(range_features_df)


# BELOW ARE THE TESTS FOR THE RANGE FEATURES FORMER CLASS
def run_tests():
    single_range_feature_should_contain_1_if_value_is_within_the_range_and_0_otherwise()
    range_pairs_should_contain_1_both_single_range_features_contain_1_and_0_otherwise()
    range_triplets_should_contain_1_all_range_pairs_and_single_range_features_contain_1_and_0_otherwise()

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
        'F1_[1, 5]': {
            'name': 'F1',
            'range_start': 1,
            'range_end': 5
        },
        'F2_[8, 6]': {
            'name': 'F2',
            'range_start': 8,
            'range_end': 6
        }
    }

    print("✅ Passed")

def range_pairs_should_contain_1_both_single_range_features_contain_1_and_0_otherwise():
    # arrange
    single_range_features_df = pd.DataFrame({
        'F1_[1, 5]': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        'F2_[5, 6]': [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        'F3_[2, 4]': [0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    })
    single_range_feature_names = ['F1_[1, 5]', 'F2_[5, 6]', 'F3_[2, 4]']
    single_range_feature_to_feature_map = {
        'F1_[1, 5]': {
            'name': 'F1',
            'range_start': 1,
            'range_end': 5
        },
        'F2_[5, 6]': {
            'name': 'F2',
            'range_start': 5,
            'range_end': 6
        },
        'F3_[2, 4]': {
            'name': 'F3',
            'range_start': 2,
            'range_end': 4
        },
    }

    logger = logging.getLogger(__name__)
    sut = RangeFeaturesFormer(logger, TestDiscretizationMethod(logger, 1, 1))

    # act
    result, map = sut.form_range_pairs(single_range_features_df, single_range_feature_names, single_range_feature_to_feature_map)

    # assert
    assert result.equals(pd.DataFrame({
        'F1_[1, 5]_F2_[5, 6]': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        'F1_[1, 5]_F3_[2, 4]': [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        # 'F2_[5, 6]_F3_[2, 4]': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] is non consistent, therefore it should be ignored
    }))

    assert map == {
        'F1_[1, 5]_F2_[5, 6]': ['F1_[1, 5]', 'F2_[5, 6]'],
        'F1_[1, 5]_F3_[2, 4]': ['F1_[1, 5]', 'F3_[2, 4]'],
    }

    print("✅ Passed")

def range_triplets_should_contain_1_all_range_pairs_and_single_range_features_contain_1_and_0_otherwise():
    # arrange
    range_pairs_df = pd.DataFrame({
        'F1_[1, 5]_F2_[5, 6]': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        'F1_[1, 5]_F3_[2, 4]': [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    })
    range_pairs_names = ['F1_[1, 5]_F2_[5, 6]', 'F1_[1, 5]_F3_[2, 4]']
    range_pairs_to_features_map = {
        'F1_[1, 5]_F2_[5, 6]': ['F1_[1, 5]', 'F2_[5, 6]'],
        'F1_[1, 5]_F3_[2, 4]': ['F1_[1, 5]', 'F3_[2, 4]'],
    }
    single_range_features_df = pd.DataFrame({
        'F4_[2, 4]': [0, 1, 1, 1, 0, 0, 0, 0, 0, 0], # it should produce a inconsistent range triplet with F1_[1, 5]_F2_[5, 6]
        'F1_[6, 8]': [0, 0, 0, 0, 0, 1, 1, 1, 0, 0], # should be ignored because it is of the feature F1 which is already in the range pairs
        'F3_[2, 4]': [0, 1, 1, 1, 0, 0, 0, 0, 0, 0], # it should produce a inconsistent range triplet with F1_[1, 5]_F3_[5, 6] and it contains the same feature F3 as the range pair F1_[1, 5]_F3_[2, 4]
        'F2_[5, 6]': [0, 0, 0, 0, 0, 1, 1, 0, 0, 0] # it should produce a inconsistent range triplet with F1_[1, 5]_F2_[2, 4] and it contains the same feature F2 as the range pair F1_[1, 5]_F2_[5, 6]
    })
    single_range_feature_names = ['F4_[2, 4]', 'F1_[6, 8]', 'F1_[1, 5]', 'F2_[5, 6]', 'F3_[2, 4]']
    single_range_feature_to_feature_map = {
        'F4_[2, 4]': {
            'name': 'F4',
            'range_start': 2,
            'range_end': 4
        },
        'F1_[6, 8]': {
            'name': 'F1',
            'range_start': 6,
            'range_end': 8
        },
        'F1_[1, 5]': {
            'name': 'F1',
            'range_start': 1,
            'range_end': 5
        },
        'F2_[5, 6]': {
            'name': 'F2',
            'range_start': 5,
            'range_end': 6
        },
        'F3_[2, 4]': {
            'name': 'F3',
            'range_start': 2,
            'range_end': 4
        },
    }

    logger = logging.getLogger(__name__)
    sut = RangeFeaturesFormer(logger, TestDiscretizationMethod(logger, 1, 1))

    # act
    result, map = sut.form_range_triplets(range_pairs_df, range_pairs_names, range_pairs_to_features_map, single_range_features_df, single_range_feature_names, single_range_feature_to_feature_map)

    # assert
    assert result.equals(pd.DataFrame({
        'F1_[1, 5]_F3_[2, 4]_F4_[2, 4]': [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    }))

    assert map == {
        'F1_[1, 5]_F3_[2, 4]_F4_[2, 4]': ['F1_[1, 5]', 'F3_[2, 4]', 'F4_[2, 4]']
    }

    print("✅ Passed")