# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.


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

    def smlp_range_analysis(self, X, y, feat_names, resp_names):
        self._range_logger.info('Starting SMLP range analysis...')
        print(X)
        print(y)
        print(feat_names)
        print(resp_names)
        self._range_logger.info('SMLP range analysis completed.')

