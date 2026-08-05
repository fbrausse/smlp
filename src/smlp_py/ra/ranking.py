from abc import ABC, abstractmethod

import logging
import pandas as pd

class RankingAlgorithm(ABC):
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    @abstractmethod
    def rank(self, feat_df: pd.DataFrame, feat_names: list[str], resp_df: pd.DataFrame, resp_name: str) -> pd.DataFrame:
        pass