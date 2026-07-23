from .discretization import IDiscretization
from logging import Logger
import pandas as pd

class RaAlgorithm:
    def __init__(self, discretization: IDiscretization, logger: Logger):
        self.discretization = discretization
        self.logger = logger

    def run(
        self,
        feat_df: pd.DataFrame,
        resp_df: pd.DataFrame
    ) -> pd.DataFrame:
        self.logger.info(f"Starting the execution of the feature range analysis algorithm...")
        
        return self.discretization.discretize(feat_df, resp_df)