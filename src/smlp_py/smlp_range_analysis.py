# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

from math import isclose
import re
from typing import Callable
from range import Range, InverseRange

import pandas as pd

class RangeAnalysis:
    def __init__(self):
        self._range_logger = None
        self._report_file_prefix = None

        self._DEF_BINS_COUNT = 10
        self._DEF_ADJACENT_BINS_COUNT = 5
        self._DEF_POSITIVE_SAMPLE_CRITERION = ''

        self.range_analysis_params_dict = {
            'bins_count': {
                'abbr': 'bins_count',
                'default': self._DEF_BINS_COUNT, 
                'type': int,
                'help':' Number of bins to devide a range into [default: {}]'.format(self._DEF_BINS_COUNT) 
            },
            'adjacent_bins_count': {
                'abbr': 'adjacent_bins_count',
                'default': self._DEF_ADJACENT_BINS_COUNT,
                'type': int,
                'help': f'Maximal number of adjacent bins allowed to merge to form a range [default: {self._DEF_ADJACENT_BINS_COUNT}]' 
            },
            'positive_sample_criterion': {
                'abbr': 'positive_sample_criterion',
                'default': self._DEF_POSITIVE_SAMPLE_CRITERION,
                'type': str,
                'help': f'Positive sample criterion, leave empty if not applicable [default: {self._DEF_POSITIVE_SAMPLE_CRITERION}]'
            }
        }

    def set_logger(self, logger):
        self._range_logger = logger

    def set_report_file_prefix(self, report_file_prefix):
        self._report_file_prefix = report_file_prefix

    def smlp_range_analysis(self, feat_df: pd.DataFrame, resp_df: pd.DataFrame, feat_names: list[str], resp_names: list[str], bins_count: int, adjacent_bins_count: int, positive_sample_criterion: str):
        self._range_logger.info('Starting SMLP range analysis...')

        self._range_logger.info('Paramters: bins_count={}, adjacent_bins_count={}, positive_sample_criterion={}'.format(bins_count, adjacent_bins_count, positive_sample_criterion))

        self._form_ranges(feat_df, resp_df, feat_names, bins_count, adjacent_bins_count, positive_sample_criterion)

        self._range_logger.info('SMLP range analysis completed.')

    def _form_ranges(self, feat_df: pd.DataFrame, resp_df: pd.DataFrame, feat_names: list[str], bins_count: int, adjacent_bins_count: int, positive_sample_criterion: str) -> dict[str, list[list[float]]]:
        ranges_map = {}

        for feat_name in feat_names:
            minf = feat_df[feat_name].min()
            maxf = feat_df[feat_name].max()
            step = (maxf - minf) / bins_count

            bin_bounds = [minf + i * step for i in range(bins_count + 1)]
            bin_bounds[-1] = maxf

            bins = [
                [bin_bounds[i], bin_bounds[i + 1]]
                for i in range(bins_count)
            ]

            ranges = []
            for start in range(bins_count):
                for length in range(1, min(adjacent_bins_count, bins_count - start) + 1):
                    ranges.append(Range(
                        bins[start][0],
                        bins[start + length - 1][1],
                    ))

            # add inverse ranges
            for range1 in list(ranges):
                has_lower_complement = False
                has_upper_complement = False
                for range2 in ranges:
                    if range1 == range2:
                        continue

                    if range2.start == minf and range2.end == range1.start:
                        has_lower_complement = True
                    if range2.end == maxf and range2.start == range1.end:
                        has_upper_complement = True

                if has_lower_complement and has_upper_complement:
                    ranges.append(InverseRange(range1.start, range1.end, minf, maxf))

            # range pruning
            for bin in bins:
                if self._should_prune_bin(bin, feat_df, feat_name, resp_df, positive_sample_criterion):
                    ranges = list(filter(
                        lambda r: 
                            isinstance(r, InverseRange) or
                            not (isclose(r.start, bin[0]) or isclose(r.end, bin[1])),
                        ranges))
                    
            ranges_map[feat_name] = ranges

        return ranges_map

    def _should_prune_bin(self, bin: list[float], feat_df: pd.DataFrame, feat_name: str, resp_df: pd.DataFrame, positive_sample_criterion: str) -> bool:
        if positive_sample_criterion == self._DEF_POSITIVE_SAMPLE_CRITERION:
            return False

        index = feat_df[(feat_df[feat_name] >= bin[0]) & (feat_df[feat_name] <= bin[1])].index

        try: 
            filtered = resp_df.iloc[index].query(positive_sample_criterion)

            return filtered.empty
        except Exception as e:
            self._range_logger.error(f"Error pruning bin {bin} for feature {feat_name}: {e}")
            return False

def run_tests():
    form_ranges_should_form_ranges_with_inverse_ranges_and_should_not_prune_bins_if_criterion_is_not_specified()
    form_ranges_should_form_ranges_with_inverse_ranges_and_prune_bins_if_criterion_is_specified()

def form_ranges_should_form_ranges_with_inverse_ranges_and_should_not_prune_bins_if_criterion_is_not_specified():
    # arrange
    feat_df = pd.DataFrame({'F': range(11)})
    resp_df = pd.DataFrame({'R': range(11)})
    feat_names = ['F']

    bins_count = 3
    adjacent_bins_count = 2
    positive_sample_criterion = ''

    sut = RangeAnalysis()

    # act
    result = sut._form_ranges(feat_df, resp_df, feat_names, bins_count, adjacent_bins_count, positive_sample_criterion)

    # assert
    actual = result['F']
    
    expected = [
        Range(0.0, 3.33),
        Range(0.0, 6.67),
        Range(3.33, 6.67),
        Range(3.33, 10.0),
        Range(6.67, 10.0),
        # inverse ranges
        InverseRange(3.33, 6.67, 0.0, 10.0),
    ]

    assert len(actual) == len(expected)

    for r in actual:
        r.start = round(r.start, 2)
        r.end = round(r.end, 2)

    for i, v in enumerate(actual):
        assert v == expected[i]

    print("✅ Passed")

def form_ranges_should_form_ranges_with_inverse_ranges_and_prune_bins_if_criterion_is_specified():
    # arrange
    df = pd.DataFrame({'F': range(11), 'R': range(11)})
    feat_df = df[['F']]
    resp_df = df[['R']]
    feat_names = ['F']

    bins_count = 3
    adjacent_bins_count = 2
    positive_sample_criterion = 'R < 3 or R > 6'

    sut = RangeAnalysis()

    # act
    result = sut._form_ranges(feat_df, resp_df, feat_names, bins_count, adjacent_bins_count, positive_sample_criterion)

    # assert
    actual = result['F']

    # [3.33, 6.67] should be pruned
    expected = [
        Range(0.0, 3.33),
        Range(6.67, 10.0),
        # inverse ranges
        InverseRange(3.33, 6.67, 0.0, 10.0),
    ]

    assert len(actual) == len(expected)

    for r in actual:
        r.start = round(r.start, 2)
        r.end = round(r.end, 2)

    for i, v in enumerate(actual):
        assert v == expected[i]

    print("✅ Passed")

if __name__ == "__main__":
    run_tests()
