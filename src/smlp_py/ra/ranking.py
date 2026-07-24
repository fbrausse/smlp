from abc import ABC, abstractmethod
import logging
import pandas as pd
import random

class RankingAlgorithm(ABC):
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    @abstractmethod
    def rank(self, feat_df: pd.DataFrame, feat_names: list[str], resp_df: pd.DataFrame, resp_name: str) -> dict[str, float]:
        pass

class DefaultRankingAlgorithm(RankingAlgorithm):
    def rank(self, feat_df: pd.DataFrame, feat_names: list[str], resp_df: pd.DataFrame, resp_name: str) -> dict[str, float]:
        self.logger.info(f"Starting default ranking of features with respect to response {resp_name}")

        # assign randomly scores to the features
        scores = {feat_name: random.random() for feat_name in feat_names}

        return scores