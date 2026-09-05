class RepresentativesSelectionAlgorithm(ABC):
    def __init__(self, logger: logging.Logger, representatives_threshold: float):
        self.logger = logger
        self.representatives_threshold = representatives_threshold

    @abstractmethod
    def select(self, feat_df: pd.DataFrame, feat_names: list[str]) -> list[str]:
        pass