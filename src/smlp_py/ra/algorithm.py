from .discretization import DiscretizationAlgorithm
from logging import Logger

import pandas as pd

from .representatives_selector import RepresentativesSelectionAlgorithm

class RaAlgorithm:
    def __init__(self, 
    discretization: DiscretizationAlgorithm,
    representatives_selector: RepresentativesSelectionAlgorithm,
    logger: Logger):
        self.discretization = discretization
        self.representatives_selector = representatives_selector
        self.logger = logger

    def run(
        self,
        feat_df: pd.DataFrame,
        feat_names: list[str],
        resp_df: pd.DataFrame,
        resp_name: str
    ) -> pd.DataFrame:
        self.logger.info(f"Starting the execution of the feature range analysis algorithm...")

        representatives = self.representatives_selector.select(feat_df, feat_names)

        return representatives