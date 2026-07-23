# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

import pandas as pd

from .ra.algorithm import RaAlgorithm
from .ra.discretization import DefaultDiscretizationAlgorithm
from .ra.representatives_selector import DefaultRepresentativesSelectionAlgorithm, RandomRepresentativesSelectionAlgorirthm

class RangeAnalysis:
    def __init__(self):
        self._range_logger = None
        self._report_file_prefix = None

        self._DEF_BINS_COUNT = 10
        self._DEF_ADJACENT_BINS_COUNT = 5
        self._DEF_DISCRETIZATION = 'default'
        
        self._DEF_REPRESENTATIVES_THRESHOLD = 0.95
        self._DEF_REPRESENTATIVES_SELECTION = 'default'

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
            },
            'representatives_selection': {
                'abbr': 'representatives_selection',
                'default': self._DEF_REPRESENTATIVES_SELECTION,
                'type': str,
                'help': f'Representatives selection algorithm to use [default: {self._DEF_REPRESENTATIVES_SELECTION}]'
            },
            'representatives_threshold': {
                'abbr': 'representatives_threshold',
                'default': self._DEF_REPRESENTATIVES_THRESHOLD,
                'type': float,
                'help': f'Threshold for selecting representatives [default: {self._DEF_REPRESENTATIVES_THRESHOLD}]'
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
        discretization: str,
        representatives_threshold: float,
        representatives_selection: str
    ): 
        self._range_logger.info(f"Starting SMLP range analysis with discretization: {discretization}")

        ra = self._setup(
            discretization, 
            representatives_selection, 
            representatives_threshold, 
            bins_count, 
            adjacent_bins_count)

        result = ra.run(feat_df, feat_names, resp_df, resp_name)

        self._range_logger.info('SMLP range analysis completed.')

    def _setup(self, 
        discretization: str, 
        representatives_selection: str, 
        representatives_threshold: float, 
        bins_count: int, 
        adjacent_bins_count: int) -> RaAlgorithm:
        if discretization == self._DEF_DISCRETIZATION:
            discretization_algorithm = DefaultDiscretizationAlgorithm(self._range_logger, bins_count, adjacent_bins_count)
        else:
            raise ValueError(f"The specified discretization algorithm is not supported: {discretization}")

        if representatives_selection == self._DEF_REPRESENTATIVES_SELECTION:
            representatives_selection_algorithm = DefaultRepresentativesSelectionAlgorithm(self._range_logger, representatives_threshold)
        elif representatives_selection == "random":
            representatives_selection_algorithm = RandomRepresentativesSelectionAlgorirthm(self._range_logger, representatives_threshold)
        else:
            raise ValueError(f"The specified representatives selection algorithm is not supported: {representatives_threshold}")

        return RaAlgorithm(discretization_algorithm, representatives_selection_algorithm, self._range_logger)

