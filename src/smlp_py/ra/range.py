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

    def contains(self, value, is_left_inclusive: bool = True, is_right_inclusive: bool = True) -> bool:
        if is_left_inclusive and is_right_inclusive:
            return (self.start <= value) & (self.end >= value)
        elif is_left_inclusive and not is_right_inclusive:
            return (self.start <= value) & (self.end > value)
        elif not is_left_inclusive and is_right_inclusive:
            return (self.start < value) & (self.end >= value)
       
        return (self.start < value) & (self.end > value)

class InverseRange(Range):
    def __init__(self, start: float, end: float, minf: float, maxf: float):
        super().__init__(end, start)
        self.minf = minf
        self.maxf = maxf

    def contains(self, value, is_left_inclusive: bool = True, is_right_inclusive: bool = True) -> bool:
        lower_complement = Range(self.minf, self.end)
        upper_complement = Range(self.start, self.maxf)

        return lower_complement.contains(value, True, False) | upper_complement.contains(value, False, True)

# BELOW ARE THE TESTS FOR THE RANGE AND INVERSE RANGE CLASSES
def run_tests():
    range_should_not_contain_value_outside_of_its_bounds()
    range_should_contain_value_within_its_bounds()
    inverse_range_should_contain_value_inside_the_bounds_of_its_complements()
    inverse_range_should_not_contain_value_outside_of_its_complements()

def range_should_not_contain_value_outside_of_its_bounds():
    # arrange
    range = Range(1, 10)
    value = 11

    # act
    result = range.contains(value)

    # assert
    assert not result
    
    print("✅ Passed")

def range_should_contain_value_within_its_bounds():
    # arrange
    range = Range(1, 10)
    value = 5

    # act
    result = range.contains(value)

    # assert
    assert result

    print("✅ Passed")

def inverse_range_should_contain_value_inside_the_bounds_of_its_complements():
    # arrange
    inverse_range = InverseRange(3, 7, 0, 10)
    value = 2

    # act
    result = inverse_range.contains(value)

    # assert
    assert result

    print("✅ Passed")

def inverse_range_should_not_contain_value_outside_of_its_complements():
    # arrange
    inverse_range = InverseRange(3, 7, 0, 10)
    value = 4

    # act
    result = inverse_range.contains(value)

    # assert
    assert not result

    print("✅ Passed")