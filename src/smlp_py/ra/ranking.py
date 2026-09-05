from abc import ABC, abstractmethod

import logging
import pandas as pd
import random as r

class RankingAlgorithm(ABC):
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    @abstractmethod
    def rank(self, feat_df: pd.DataFrame, feat_names: list[str], resp_df: pd.DataFrame, resp_name: str) -> pd.DataFrame:
        pass

class RandomRankingAlgorithm(RankingAlgorithm):
    def rank(self, feat_df: pd.DataFrame, feat_names: list[str], resp_df: pd.DataFrame, resp_name: str) -> pd.DataFrame:
        shuffled = r.sample(feat_names, k=len(feat_names))

        return pd.DataFrame({
            'Feature': shuffled,
            'Score': list(range(len(shuffled), 0, -1))
        }, columns=['Feature', 'Score'])
