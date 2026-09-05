from abc import ABC, abstractmethod

import logging
import pandas as pd

class Quality(ABC):
    def __init__(self, logger: logging.Logger, quality_function: str) -> None:
        self.logger = logger
        self.quality_function = quality_function

    @abstractmethod
    def rank(self, quality_metrics_df: pd.DataFrame):
        pass

    @abstractmethod
    def compute_metrics(self, feat_names: list[str], feat_df: pd.DataFrame, resp_df: pd.DataFrame, resp_name: str):
        pass
