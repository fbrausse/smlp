# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

import pandas as pd

from .ra.algorithm import RaAlgorithm
from .ra.discretization import DefaultDiscretizationMethod
from .ra.representatives_selection import DefaultRepresentativesSelectionAlgorithm, RandomRepresentativesSelectionAlgorirthm
from .ra.correlation_method import PearsonCorrelationMethod
from .ra.ranking import DefaultRankingAlgorithm
from .ra.range_features import RangeFeaturesFormer
from .smlp_basis import SmlpBasisAlgorithm
from .smlp_mrmr import SmlpMrmr

class RangeAnalysis:
    def __init__(self):
        self._range_logger = None
        self._report_file_prefix = None

        self._DEF_BINS_COUNT = 10
        self._DEF_ADJACENT_BINS_COUNT = 5
        self._DEF_DISCRETIZATION = 'default'
        
        self._DEF_REPRESENTATIVES_THRESHOLD = 0.95
        self._DEF_REPRESENTATIVES_SELECTION = 'default'
        self._DEF_CORRELATION_METHOD = 'pearson'
        self._DEF_RANKING = 'efs' # stands for ensmble feature selection
        self._DEF_BASIS = 'mrmr'
        self._DEF_TOP_RANKING_FEATURES_COUNT = 15
        self._DEF_TOP_FINAL_FEATURES_COUNT = 5

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
            },
            'correlation_method': {
                'abbr': 'correlation_method',
                'default': self._DEF_CORRELATION_METHOD,
                'type': str,
                'help': f'Correlation method to use for representatives selection [default: {self._DEF_CORRELATION_METHOD}]'
            },
            'ranking': {
                'abbr': 'ranking',
                'default': self._DEF_RANKING,
                'type': str,
                'help': f'Ranking algorithm to use [default: {self._DEF_RANKING}]'
            },
            'top_ranking_features_count': {
                'abbr': 'top_ranking_features_count',
                'default': self._DEF_TOP_RANKING_FEATURES_COUNT,
                'type': int,
                'help': f'Number of top features to select [default: {self._DEF_TOP_RANKING_FEATURES_COUNT}]'
            },
            'top_final_features_count': {
                'abbr': 'top_final_features_count',
                'default': self._DEF_TOP_FINAL_FEATURES_COUNT,
                'type': int,
                'help': f'Number of top features to select [default: {self._DEF_TOP_FINAL_FEATURES_COUNT}]'
            },
            'basis': {
                'abbr': 'basis',
                'default': self._DEF_BASIS,
                'type': str,
                'help': f'Basis algorithm to use [default: {self._DEF_BASIS}]'
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
        representatives_selection: str,
        correlation_method: str,
        ranking: str,
        top_ranking_features_count: int,
        top_final_features_count: int,
        basis: str
    ): 
        self._range_logger.info(f"Starting SMLP range analysis...")

        ra: RaAlgorithm = self._setup(
            discretization, 
            representatives_selection, 
            representatives_threshold, 
            correlation_method,
            bins_count, 
            adjacent_bins_count,
            ranking,
            basis)

        result = ra.run(feat_df, feat_names, resp_df, resp_name, top_ranking_features_count, top_final_features_count)
        self._range_logger.info(f"Feature range analysis completed. The result: \n {result}")

        self._range_logger.info('SMLP range analysis completed.')

    def _setup(self, 
        discretization: str, 
        representatives_selection: str, 
        representatives_threshold: float, 
        correlation_method: str,
        bins_count: int, 
        adjacent_bins_count: int,
        ranking: str,
        basis: str) -> RaAlgorithm:
        if ranking == self._DEF_RANKING:
            ranking_algorithm = DefaultRankingAlgorithm(self._range_logger)
        else:
            raise ValueError(f"The specified ranking algorithm is not supported: {ranking}")

        if discretization == self._DEF_DISCRETIZATION:
            discretization_method = DefaultDiscretizationMethod(self._range_logger, bins_count, adjacent_bins_count)
            range_features_former = RangeFeaturesFormer(self._range_logger, discretization_method)
        else:
            raise ValueError(f"The specified discretization method is not supported: {discretization}")

        if representatives_selection == self._DEF_REPRESENTATIVES_SELECTION:
            if correlation_method == self._DEF_CORRELATION_METHOD:
                correlation_method = PearsonCorrelationMethod()
            else:
                raise ValueError(f"The specified correlation method is not supported: {correlation_method}")

            representatives_selection_algorithm = DefaultRepresentativesSelectionAlgorithm(self._range_logger, representatives_threshold, correlation_method)
        elif representatives_selection == "random":
            representatives_selection_algorithm = RandomRepresentativesSelectionAlgorirthm(self._range_logger, representatives_threshold)
        else:
            raise ValueError(f"The specified representatives selection algorithm is not supported: {representatives_threshold}")

        if basis == self._DEF_BASIS:
            mrmr = SmlpMrmr()
            mrmr.set_logger(self._range_logger)

            basis_algorithm = SmlpBasisAlgorithm(self._range_logger, mrmr)
        else:
            raise ValueError(f"The specified basis algorithm is not supported: {basis}")

        return RaAlgorithm(
            range_features_former,
            representatives_selection_algorithm,
            ranking_algorithm,
            basis_algorithm,
            self._range_logger)

