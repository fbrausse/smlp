from abc import ABC, abstractmethod

import logging
import pandas as pd

class Quality(ABC):
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    @abstractmethod
    def rank(self, feat_names: list[str], quality_function: str, feat_df: pd.DataFrame, resp_df: pd.DataFrame, resp_name: str):
        pass