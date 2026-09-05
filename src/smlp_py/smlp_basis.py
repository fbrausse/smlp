from .ra.basis import BasisAlgorithm
from .smlp_mrmr import SmlpMrmr

import logging
import pandas as pd

class SmlpBasisAlgorithm(BasisAlgorithm):
    def __init__(self, logger: logging.Logger, mrmr: SmlpMrmr):
        super().__init__(logger)
        self.mrmr = mrmr

    def rank(self, feat_df: pd.DataFrame, feat_names: list[str], resp_df: pd.DataFrame, resp_name: str) -> pd.DataFrame:
        if (len(feat_names) == 0):
            return pd.DataFrame(columns=['Feature', 'Score'])
        
        self.logger.info(f"Starting basis selection with mrmr for features {feat_names} and response {resp_name}")

        _, mrmr_result = self.mrmr.smlp_mrmr(feat_df[feat_names], resp_df[resp_name], len(feat_names))

        return mrmr_result