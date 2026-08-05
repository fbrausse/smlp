# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

import pandas as pd

from .ra.algorithm import RaAlgorithm
from .ra.discretization import DefaultDiscretizationMethod
from .ra.representatives_selection import DefaultRepresentativesSelectionAlgorithm, RandomRepresentativesSelectionAlgorirthm
from .ra.range_features import RangeFeaturesFormer

from .smlp_basis import SmlpBasisAlgorithm
from .smlp_mrmr import SmlpMrmr
from .smlp_quality import SmlpQuality
from .smlp_subgroups import SubgroupDiscovery
from .smlp_ranking import SmlpRankingAlgorithm
from .smlp_correlations import SmlpCorrelations

class RangeAnalysis:
    def __init__(self):
        self._range_logger = None
        self._report_file_prefix = None

        self._DEF_BINS_COUNT = 10
        self._DEF_ADJACENT_BINS_COUNT = 5
        self._DEF_DISCRETIZATION = 'default'
        
        self._DEF_REPRESENTATIVES_THRESHOLD = 0.95
        self._DEF_REPRESENTATIVES_SELECTION = 'default'
        self._DEF_RANKING = 'efs' # stands for ensmble feature selection
        self._DEF_BASIS = 'mrmr'
        self._DEF_TOP_RANKING_FEATURES_COUNT = 15
        self._DEF_TOP_FINAL_FEATURES_COUNT = 5
        self._DEF_QUALITY_FUNCTION = 'TPR'
        self._POSSIBLE_QUALITY_FUNCTIONS = ['TPR', 'PPV', 'Lift', 'ROCAcc', 'NPLR', 'WRAcc', 'F1Score', 'Acc', 'Kappa']

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
            },
            'quality_function': {
                'abbr': 'quality_function',
                'default': self._DEF_QUALITY_FUNCTION,
                'type': str,
                'help': f'Quality function to be used in the quality ranking [default: {self._DEF_QUALITY_FUNCTION}]'
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
        ranking: str,
        top_ranking_features_count: int,
        top_final_features_count: int,
        basis: str,
        quality_function: str
    ): 
        self._range_logger.info(f"Starting SMLP range analysis...")

        self._validate_args(bins_count, adjacent_bins_count, top_ranking_features_count, top_final_features_count, quality_function)

        ra: RaAlgorithm = self._setup(
            discretization, 
            representatives_selection, 
            representatives_threshold, 
            bins_count, 
            adjacent_bins_count,
            ranking,
            basis)

        result = ra.run(feat_df, feat_names, resp_df, resp_name, top_ranking_features_count, top_final_features_count, quality_function)
        path = self._save_result(result)

        self._range_logger.info(f'SMLP range analysis completed. Result is saved in {path}')

    def _setup(self, 
        discretization: str, 
        representatives_selection: str, 
        representatives_threshold: float, 
        bins_count: int, 
        adjacent_bins_count: int,
        ranking: str,
        basis: str) -> RaAlgorithm:
        ranking_algorithm = self._setup_ranking(ranking)
        range_features_former = self._setup_discretization(discretization, bins_count, adjacent_bins_count)
        representatives_selection_algorithm = self._setup_representatives_selection(
            representatives_selection, representatives_threshold)
        basis_algorithm = self._setup_basis(basis)
        quality = self._setup_quality()

        return RaAlgorithm(
            range_features_former,
            representatives_selection_algorithm,
            ranking_algorithm,
            basis_algorithm,
            quality,
            self._range_logger)

    def _setup_ranking(self, ranking: str) -> SmlpRankingAlgorithm:
        if ranking == self._DEF_RANKING:
            correlations = SmlpCorrelations()
            correlations.set_logger(self._range_logger)

            return SmlpRankingAlgorithm(self._range_logger, correlations)
        else:
            raise ValueError(f"The specified ranking algorithm is not supported: {ranking}")

    def _setup_discretization(self, discretization: str, bins_count: int, adjacent_bins_count: int) -> RangeFeaturesFormer:
        if discretization == self._DEF_DISCRETIZATION:
            discretization_method = DefaultDiscretizationMethod(self._range_logger, bins_count, adjacent_bins_count)
            return RangeFeaturesFormer(self._range_logger, discretization_method)
        else:
            raise ValueError(f"The specified discretization method is not supported: {discretization}")

    def _setup_representatives_selection(
        self,
        representatives_selection: str,
        representatives_threshold: float,
    ):
        if representatives_selection == self._DEF_REPRESENTATIVES_SELECTION:
            return DefaultRepresentativesSelectionAlgorithm(self._range_logger, representatives_threshold)
        elif representatives_selection == "random":
            return RandomRepresentativesSelectionAlgorirthm(self._range_logger, representatives_threshold)
        else:
            raise ValueError(f"The specified representatives selection algorithm is not supported: {representatives_threshold}")

    def _setup_basis(self, basis: str) -> SmlpBasisAlgorithm:
        if basis == self._DEF_BASIS:
            mrmr = SmlpMrmr()
            mrmr.set_logger(self._range_logger)

            return SmlpBasisAlgorithm(self._range_logger, mrmr)
        else:
            raise ValueError(f"The specified basis algorithm is not supported: {basis}")

    def _setup_quality(self) -> SmlpQuality:
        sd = SubgroupDiscovery()
        sd.set_logger(self._range_logger)

        return SmlpQuality(self._range_logger, sd)

    def _validate_args(self, bins_count: int, adjacent_bins_count: int, top_ranking_features_count, top_final_features_count, quality_function: str):
        if bins_count <= 0:
            raise ValueError("The bins_count argument must be greater than 0")

        if adjacent_bins_count <= 0:
            raise ValueError("The adjacent_bins_count argument must be greater than 0")
        
        if top_ranking_features_count <= 0:
            raise ValueError("The top_ranking_features_count argument must be greater than 0")

        if top_final_features_count <= 0:
            raise ValueError("The top_final_features_count argument must be greater than 0")
        
        if quality_function not in self._POSSIBLE_QUALITY_FUNCTIONS:
            raise ValueError(f"The quality function must be one of {self._POSSIBLE_QUALITY_FUNCTIONS}")

    def _save_result(self, result: pd.DataFrame):
        path = self._report_file_prefix + '_range_analysis.csv'
        result.to_csv(path)

        return path