import logging
import pandas as pd

from .discretization import DiscretizationMethod

class RangeFeaturesFormer:
    def __init__(self, logger: logging.Logger, discretization_method: DiscretizationMethod):
        self.logger = logger
        self.discretization_method = discretization_method

    def form_single_range_features(
        self, 
        feat_df: pd.DataFrame, 
        feat_names: list[str], 
        resp_df: pd.DataFrame, 
        resp_name: str) -> pd.DataFrame:
        self.logger.info(f"Forming single range features with {self.discretization_method.name()} discretization method")
        ranges_map = self.discretization_method.discretize(feat_df, feat_names, resp_df, resp_name)
        ...