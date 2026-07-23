from math import isclose

class Range:
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end

    def __eq__(self, other: 'Range') -> bool:            
        return isclose(self.start, other.start) and isclose(self.end, other.end)

    def __str__(self) -> str:
        return f"[{self.start}, {self.end}]"

    __repr__ = __str__

    def contains(self, value) -> bool:
        return (self.start <= value) and (self.end >= value)

class InverseRange(Range):
    def __init__(self, start: float, end: float, minf: float, maxf: float):
        super().__init__(end, start)
        self.minf = minf
        self.maxf = maxf

    def contains(self, value) -> bool:
        lower_complement = Range(self.minf, self.end)
        upper_complement = Range(self.start, self.maxf)

        return lower_complement.contains(value) | upper_complement.contains(value)