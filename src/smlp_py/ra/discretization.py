from abc import ABC, abstractmethod
from math import isclose

from .range import Range, InverseRange

import pandas as pd
import logging


class DiscretizationMethod(ABC):
    def __init__(
        self, 
        logger: logging.Logger,
        bins_count: int,
        adjacent_bins_count: int
    ):
        self.logger = logger
        self.bins_count = bins_count
        self.adjacent_bins_count = adjacent_bins_count

    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def discretize(
        self,
        feat_df: pd.DataFrame,
        feat_names: list[str],
        resp_df: pd.DataFrame,
        resp_name: str
    ) -> dict[str, list[Range]]:
        pass


class DefaultDiscretizationMethod(DiscretizationMethod):
    def discretize(
        self,
        feat_df: pd.DataFrame,
        feat_names: list[str],
        resp_df: pd.DataFrame,
        resp_name: str
    ) -> dict[str, list[Range]]:
        self.logger.info(f"Starting default discretization with bins count {self.bins_count} and adjacent bins count {self.adjacent_bins_count}")

        return self._form_ranges(feat_df, feat_names, resp_df, resp_name)
    
    def name(self) -> str:
        return "default"
    
    def _form_ranges(self, feat_df: pd.DataFrame, feat_names: list[str], resp_df: pd.DataFrame, resp_name: str) -> dict[str, list[Range]]:
        ranges_map = {}

        for feat_name in feat_names:
            minf = feat_df[feat_name].min()
            maxf = feat_df[feat_name].max()
            step = (maxf - minf) / self.bins_count

            bin_bounds = [minf + i * step for i in range(self.bins_count + 1)]
            bin_bounds[-1] = maxf

            bins = [
                [bin_bounds[i], bin_bounds[i + 1]]
                for i in range(self.bins_count)
            ]

            ranges = []
            for start in range(self.bins_count):
                for length in range(1, min(self.adjacent_bins_count, self.bins_count - start) + 1):
                    ranges.append(Range(
                        bins[start][0],
                        bins[start + length - 1][1],
                    ))

            # add inverse ranges
            for range1 in list(ranges):
                has_lower_complement = False
                has_upper_complement = False
                for range2 in ranges:
                    if range1 == range2:
                        continue

                    if range2.start == minf and range2.end == range1.start:
                        has_lower_complement = True
                    if range2.end == maxf and range2.start == range1.end:
                        has_upper_complement = True

                if has_lower_complement and has_upper_complement:
                    ranges.append(InverseRange(range1.start, range1.end, minf, maxf))

            # range pruning
            for bin in bins:
                if self._should_prune_bin(bin, feat_df, feat_name, resp_df, resp_name):
                    ranges = list(filter(
                        lambda r: 
                            isinstance(r, InverseRange) or
                            not (isclose(r.start, bin[0]) or isclose(r.end, bin[1])),
                        ranges))
                    
            ranges_map[feat_name] = ranges

        return ranges_map

    def _should_prune_bin(self, bin: list[float], feat_df: pd.DataFrame, feat_name: str, resp_df: pd.DataFrame, resp_name: str) -> bool:
        index = feat_df[(feat_df[feat_name] >= bin[0]) & (feat_df[feat_name] <= bin[1])].index
        
        return not (resp_df.loc[index, resp_name] == 1).any()


# BELOW ARE THE TESTS FOR THE DEFAULT DISCRETIZATION ALGORITHM
def run_tests():
    default_discretization_method_should_form_ranges_with_inverse_ranges_and_prune_bins()

def default_discretization_method_should_form_ranges_with_inverse_ranges_and_prune_bins():
    # arrange
    df = pd.DataFrame({'F': range(11), 'R': [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0]})
    feat_df = df[['F']]
    resp_df = df[['R']]

    bins_count = 3
    adjacent_bins_count = 2

    logger = logging.getLogger(__name__)
    sut = DefaultDiscretizationMethod(logger, bins_count, adjacent_bins_count)

    # act
    result = sut.discretize(feat_df, ['F'], resp_df, 'R')

    # assert
    actual = result['F']

    # [3.33, 6.67] should be pruned
    expected = [
        Range(0.0, 3.33),
        Range(6.67, 10.0),
        # inverse ranges
        InverseRange(3.33, 6.67, 0.0, 10.0),
    ]

    assert len(actual) == len(expected)

    for r in actual:
        r.start = round(r.start, 2)
        r.end = round(r.end, 2)

    for i, v in enumerate(actual):
        assert v == expected[i]

    print("✅ Passed")