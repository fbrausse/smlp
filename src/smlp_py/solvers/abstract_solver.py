import functools
from abc import ABC, abstractmethod
import smlp

USE_CACHE = False


def conditional_cache(func):
    """Custom decorator to conditionally apply @functools.cache."""
    if USE_CACHE:
        # Apply caching
        return functools.cache(func)
    else:
        # Return the original function without caching
        return func


class ClassProperty:
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, instance, owner):
        return self.fget(owner)


class AbstractSolver(ABC):

    @abstractmethod
    def create_query(self, *args, **kwargs):
        pass

    @abstractmethod
    def create_query_and_beta(self, *args, **kwargs):
        pass

    @abstractmethod
    def substitute_objective_with_witness(self, *args, **kwargs):
        pass

    @abstractmethod
    def generate_rad_term(self, *args, **kwargs):
        pass

    @abstractmethod
    def create_theta_form(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_rad_term(self, *args, **kwargs):
        pass

    @abstractmethod
    def create_alpha_or_eta_form(self, *args, **kwargs):
        pass

    @abstractmethod
    def parse_ast(self, *args, **kwargs):
        pass

    @abstractmethod
    def create_solver(self, *args, **kwargs):
        pass

    @abstractmethod
    def add_formula(self, *args, **kwargs):
        pass

    @abstractmethod
    def check(self, *args, **kwargs):
        pass

    @abstractmethod
    def create_counter_example(self, *args, **kwargs):
        pass

    @abstractmethod
    def substitute(self, *args, **kwargs):
        pass

    @abstractmethod
    def handle_ite_formula(self, *args, **kwargs):
        pass

    @abstractmethod
    def calculate_eta_F_t(self, *args, **kwargs):
        pass

    @abstractmethod
    def apply_theta(self, *args, **kwargs):
        pass

    def get_witness(self, *args, **kwargs):
        result = kwargs["result"]
        witness = kwargs["witness"]
        interface = kwargs["interface"]

        condition = result == "sat"

        if condition:
            reduced_model = dict((k, v) for k, v in witness.items() if k in interface)
            return reduced_model
        else:
            return None

    def convert_results_to_string(self, res):
        if isinstance(res, smlp.sat):
            return "sat"
        elif isinstance(res, smlp.unsat):
            return "unsat"
        elif isinstance(res, smlp.unknown):
            return "unknown"
        elif type(res) == str:
            return res.lower()
        else:
            raise Exception("Unsupported result format")
