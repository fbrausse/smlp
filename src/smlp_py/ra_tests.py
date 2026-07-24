from .ra.discretization import run_tests as run_discretization_tests
from .ra.representatives_selection import run_tests as run_representatives_selection_tests

def run_ra_tests():
    run_discretization_tests()
    run_representatives_selection_tests()