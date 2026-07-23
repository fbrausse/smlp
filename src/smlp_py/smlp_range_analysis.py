# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

import pandas as pd

from .ra.discretization import DefaultDiscretizationAlgorithm

class RangeAnalysis:
    def __init__(self):
        self._range_logger = None
        self._report_file_prefix = None

        self._DEF_BINS_COUNT = 10
        self._DEF_ADJACENT_BINS_COUNT = 5
        self._DEF_DISCRETIZATION = 'default'

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
            'discretization': {
                'abbr': 'discretization',
                'default': self._DEF_DISCRETIZATION,
                'type': str,
                'help': f'Discretization algorithm to use [default: {self._DEF_DISCRETIZATION}]'
            }
        }

    def set_logger(self, logger):
        self._range_logger = logger

    def set_report_file_prefix(self, report_file_prefix):
        self._report_file_prefix = report_file_prefix

    def smlp_range_analysis(
        self,
        feat_df: pd.DataFrame,
        resp_df: pd.DataFrame,
        feat_names: list[str],
        resp_name: str,
        bins_count: int,
        adjacent_bins_count: int,
        discretization: str
    ): 
        self._range_logger.info(f"Starting SMLP range analysis with discretization: {discretization}")
        
        if discretization == 'default':
            discretization_algorithm = DefaultDiscretizationAlgorithm(self._range_logger, bins_count, adjacent_bins_count)
        else:
            raise ValueError(f"The specified discretization algorithm is not supported: {discretization}")

        ranges_map = discretization_algorithm.discretize(feat_df, feat_names, resp_df, resp_name)

        print(f"ranges_map: {ranges_map}")

        self._range_logger.info('SMLP range analysis completed.')
