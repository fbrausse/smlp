from .ra.discretization import run_tests as run_discretization_tests
from .ra.range import run_tests as run_range_tests
from .ra.range_features import run_tests as run_range_features_tests

def run_ra_tests():
    run_discretization_tests()
    run_range_tests()
    run_range_features_tests()