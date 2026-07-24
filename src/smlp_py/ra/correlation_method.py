from abc import ABC, abstractmethod
from scipy import stats

class CorrelationMethod(ABC):
    @abstractmethod
    def compute_correlation(self, var1, var2) -> float:
        pass

class PearsonCorrelationMethod(CorrelationMethod):
    def compute_correlation(self, var1, var2) -> float:
        return stats.pearsonr(var1, var2).statistic