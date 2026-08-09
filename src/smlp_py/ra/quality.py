from abc import ABC, abstractmethod

import logging
import pandas as pd

class Quality(ABC):
    def __init__(self, logger: logging.Logger, quality_function: str) -> None:
        self.logger = logger
        self.quality_function = quality_function

    @abstractmethod
    def rank(self, feat_names: list[str], quality_function: str, feat_df: pd.DataFrame, resp_df: pd.DataFrame, resp_name: str):
        pass