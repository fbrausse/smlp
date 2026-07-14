# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

from math import isclose

import pandas as pd

class RangeAnalysis:
    def __init__(self):
        self._range_logger = None
        self._report_file_prefix = None

        self._DEF_M = 10
        self._DEF_K = 5

        self.range_analysis_params_dict = {
            'm': {
                'abbr': 'm',
                'default': self._DEF_M, 
                'type': int,
                'help':' Number of bins to devide a range into [default: {}]'.format(self._DEF_M) 
            },
            'k': {
                'abbr': 'k',
                'default': self._DEF_K,
                'type': int,
                'help': f'Maximal number of adjacent bins allowed to merge to form a range [default: {self._DEF_K}]' 
            }
        }

    def set_logger(self, logger):
        self._range_logger = logger

    def set_report_file_prefix(self, report_file_prefix):
        self._report_file_prefix = report_file_prefix

    def smlp_range_analysis(self, feat_df: pd.DataFrame, resp_df: pd.DataFrame, feat_names: list[str], resp_names: list[str], m: int, k: int):
        self._range_logger.info('Starting SMLP range analysis...')
        print(feat_df)
        print(resp_df)
        print(feat_names)
        print(resp_names)
        print(f"m: {m}, k: {k}")
        self._range_logger.info('SMLP range analysis completed.')

    def _form_ranges(self, feat_df: pd.DataFrame, feat_names: list[str], m: int, k: int) -> dict[str, list[list[float]]]:
        ranges_map = {}

        for feat_name in feat_names:
            minf = feat_df[feat_name].min()
            maxf = feat_df[feat_name].max()
            step = (maxf - minf) / m

            bin_bounds = [minf + i * step for i in range(m + 1)]
            bin_bounds[-1] = maxf

            bins = [
                [bin_bounds[i], bin_bounds[i + 1]]
                for i in range(m)
            ]

            ranges = []
            for start in range(m):
                for length in range(1, min(k, m - start) + 1):
                    ranges.append([
                        bins[start][0],
                        bins[start + length - 1][1],
                    ])

            # add inverse ranges
            for range1 in list(ranges):
                has_lower_complement = False
                has_upper_complement = False
                for range2 in ranges:
                    if isclose(range1[0], range2[0]) and isclose(range1[1], range2[1]):
                        continue

                    if isclose(range2[0], minf) and isclose(range2[1], range1[0]):
                        has_lower_complement = True
                    if isclose(range2[1], maxf) and isclose(range2[0], range1[1]):
                        has_upper_complement = True

                if has_lower_complement and has_upper_complement:
                    ranges.append([range1[1], range1[0]])

            ranges_map[feat_name] = ranges

        return ranges_map

def run_tests():
    form_ranges_should_form_ranges_with_inverse_ranges()

def form_ranges_should_form_ranges_with_inverse_ranges():
    # arrange
    feat_df = pd.DataFrame({'F': range(11)})
    feat_names = ['F']

    m = 3
    k = 2

    sut = RangeAnalysis()

    # act
    result = sut._form_ranges(feat_df, feat_names, m, k)

    # assert
    actual = result['F']
    
    expected = [
        [0.0, 3.33],
        [0.0, 6.67],
        [3.33, 6.67],
        [3.33, 10.0],
        [6.67, 10.0],
        # inverse ranges
        [6.67, 3.33],

    ]

    assert len(actual) == len(expected)

    for i, v in enumerate(actual):
        assert round(v[0], 2) == round(expected[i][0], 2) and round(v[1], 2) == round(expected[i][1], 2)

    print("✅ Passed")

if __name__ == "__main__":
    test = True 
    if test:
        run_tests()
    else:
        ra = RangeAnalysis()
        result = ra._form_ranges(pd.DataFrame({'F': range(11)}), ['F'], 10, 5)
        print(result['F'])
