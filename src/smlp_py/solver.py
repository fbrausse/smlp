import functools
import types
from abc import ABC, abstractmethod
from enum import Enum

import pysmt
import smlp
from pysmt.shortcuts import Real

from src.smlp_py.NN_verifiers.verifiers import MarabouVerifier
from src.smlp_py.smtlib.text_to_sympy import TextToPysmtParser
import operator as op
from pysmt.shortcuts import Symbol, And
from src.smlp_py.smlp_operations import SMLPOperations, PYSMTOperations

class ClassProperty:
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, instance, owner):
        return self.fget(owner)

class AbstractSolver(ABC):

    # @abstractmethod
    # def true(self):
    #     pass

    # @abstractmethod
    # def GE(self, *args, **kwargs):
    #     pass

    # @abstractmethod
    # def LE(self, *args, **kwargs):
    #     pass

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



class Pysmt_Solver(AbstractSolver, PYSMTOperations):
    verifier = None

    def __init__(self, specs):
        super().__init__()
        self.specs = specs
        self.create_verifier()

    def create_verifier(self):
        symbols = []
        feat_names, resp_names, spec_domain_dict = self.specs

        for feature in feat_names:
            type = spec_domain_dict[feature]['range']
            symbols.append((feature, type, True))

        for response in resp_names:
            type = spec_domain_dict[response]['range']
            symbols.append((response, type, False))

        parser = TextToPysmtParser()
        parser.init_variables(symbols=symbols)

        self.verifier = MarabouVerifier(parser=parser)
        self.verifier.initialize(spec_domain_dict)

    @ClassProperty
    def smlp_true(self):
        return pysmt.shortcuts.TRUE()

    def smlp_var(self, var):
        return self.verifier.parser.get_symbol(var)

    def create_query(self, query_form=None):
        self.verifier.parser.handle_ite_formula(query_form, is_form2=True)

    def create_query_and_beta(self, query, beta):
        return self.verifier.parser.and_(query, beta)

    def substitute_objective_with_witness(self, *args, **kwargs):
        stable_witness_terms = kwargs["stable_witness_terms"]
        objv_term = kwargs["objv_term"]

        substitution = {}
        for symbol, value in stable_witness_terms.items():
            symbol = self.verifier.parser.get_symbol(symbol)
            substitution[symbol] = Real(value)
        # Apply the substitution
        return self.verifier.parser.simplify(objv_term.substitute(substitution))

    def generate_rad_term(self, **kwargs):
        rad = kwargs["rad"]
        return float(rad)

    def get_rad_term(self, **kwargs):
        rad = kwargs["rad"]
        return float(rad)

    def create_theta_form(self, **kwargs):
        witness = kwargs["witness"]
        var = kwargs["var"]
        rad_term = kwargs["rad_term"]
        theta_form = kwargs["theta_form"]

        value = float(witness)
        PYSMT_var = self.verifier.parser.get_symbol(var)
        type = pysmt.shortcuts.Int if str(PYSMT_var.get_type()) == "Int" else Real
        calc_type = int if str(PYSMT_var.get_type()) == "Int" else float
        lower = calc_type(value - rad_term)
        lower = type(lower)
        upper = calc_type(value + rad_term)
        upper = type(upper)
        theta_form = self.verifier.parser.and_(theta_form, PYSMT_var >= lower, PYSMT_var <= upper)
        return theta_form

    def create_alpha_or_eta_form(self, **kwargs):
        alpha_or_eta_form = kwargs["alpha_or_eta_form"]
        mx = kwargs["mx"]
        mn = kwargs["mn"]
        v = kwargs["v"]

        symbol_v = self.smlp_var(v)
        form = self.smlp_and(symbol_v >= mn, symbol_v <= mx)
        return self.simplify(self.smlp_and(alpha_or_eta_form, form))

    def simplify(self, expression):
        return self.verifier.parser.simplify(expression)

    def parse(self, expression):
        return self.verifier.parser.parse(expression)

    def GE(self, *args):
        return args[0] >= args[1]

    def parse_ast(self, *args, **kwargs):
        expression = kwargs['expression']
        return self.parse(expression)

    def create_solver(self, *args, **kwargs):
        self.verifier.reset()
        return self

    def add_formula(self, formula):
        self.verifier.apply_restrictions(formula)

    def check(self, *args, **kwargs):



class Form2_Solver(AbstractSolver, SMLPOperations):
    verifier = None
    smlp_term_instance = None
    terms = None

    def __init__(self):
        super().__init__()
        # self.verifier = verifier
        # self.smlp_term_instance = smlp_term_instance

    @property
    def smlp_true(self):
        return smlp.true

    def create_query(self, query_form=None):
        return query_form

    def create_query_and_beta(self, query, beta):
        return self.smlp_term_instance.smlp_and(query, beta)

    def substitute_objective_with_witness(self, *args, **kwargs):
        stable_witness_terms = kwargs["stable_witness_terms"]
        objv_term = kwargs["objv_term"]

        return smlp.subst(objv_term, stable_witness_terms);

    def generate_rad_term(self, *args, **kwargs):
        rad = kwargs["rad"]
        delta_rel = kwargs["delta_rel"]
        var_term = kwargs["var_term"]
        candidate = kwargs["candidate"]

        rad_term = smlp.Cnst(rad)
        if delta_rel is not None:  # radius for a lemma -- cex holds values of candidate counter-example
            rad_term = rad_term * abs(var_term)
        else:  # radius for excluding a candidate -- cex holds values of the candidate
            rad_term = rad_term * abs(candidate)

        return rad_term

    def create_theta_form(self, *args, **kwargs):
        theta_form = kwargs["theta_form"]
        witness = kwargs["witness"]
        var_term = kwargs["var_term"]
        rad_term = kwargs["rad_term"]
        smlp_and_ = kwargs["smlp_and_"]

        return smlp_and_(theta_form, ((abs(var_term - witness)) <= rad_term))

    def get_rad_term(self, *args, **kwargs):
        smlp_cnst = kwargs["smlp_cnst"]
        rad = kwargs["rad"]

        return smlp_cnst(rad)

    def create_alpha_or_eta_form(self, *args, **kwargs):
        alpha_or_eta_form = kwargs["alpha_or_eta_form"]
        is_in_spec = kwargs["is_in_spec"]
        is_disjunction = kwargs["is_disjunction"]
        is_alpha = kwargs["is_alpha"]
        mx = kwargs["mx"]
        mn = kwargs["mn"]
        v = kwargs["v"]


        if is_disjunction and is_alpha and is_in_spec:
            rng = self.smlp_or_multi([self.smlp_eq(self.smlp_var(v), self.smlp_cnst(i)) for i in range(mn, mx + 1)])
        else:
            rng = self.smlp_and(self.smlp_var(v) >= self.smlp_cnst(mn), self.smlp_var(v) <= self.smlp_cnst(mx))

        return self.smlp_and(alpha_or_eta_form, rng)

    def GE(self, *args):
        return args[0] >= args[1]

    def parse_ast(self, *args, **kwargs):
        expression = kwargs['expression']
        parser = kwargs['parser']
        return parser(expression)

    def create_solver(self, *args, **kwargs):
        create_solver = kwargs["create_solver"]
        domain = kwargs["domain"]
        model_full_term_dict = kwargs["model_full_term_dict"]
        incremental = kwargs["incremental"]
        solver_logic = kwargs["solver_logic"]
        formulas = kwargs["formulas"]

        self.verifier = create_solver(domain, model_full_term_dict, incremental, solver_logic)
        return self

    def add_formula(self, formula):
        self.verifier.add(formula)


class Solver:
    class Version(Enum):
        FORM2 = 0
        PYSMT = 1

    _instance = None
    version = None

    def __new__(cls, *args, **kwargs):
        version = kwargs["version"]
        if isinstance(version, cls.Version):
            cls.version = version
        else:
            raise ValueError("Must be a valid version")

        if cls._instance is None and isinstance(cls.version, cls.Version):
            if cls.version == cls.Version.PYSMT:
                specs = kwargs["specs"]
                cls._instance = Pysmt_Solver(specs)
            else:
                cls._instance = Form2_Solver()
            cls._map_instance_methods()
        return cls._instance

    @classmethod
    def _map_instance_methods(cls):
        """Automatically maps all methods from the instance to the SingletonFactory class."""
        # for name, method in cls._instance.__class__.__dict__.items():
        #     if isinstance(method, types.FunctionType):
        #         # Avoid overwriting existing methods in Solver class if not hasattr(cls, name):
        #         setattr(cls, name, cls._create_delegator(name))
        for base_class in cls._instance.__class__.__mro__:
            for name, method in base_class.__dict__.items():
                if isinstance(method, types.FunctionType):
                    # Avoid overwriting existing methods in Solver classifnothasattr(cls, name):
                    setattr(cls, name, cls._create_delegator(name))

    @classmethod
    def _create_delegator(cls, method_name):
        """Create a method that delegates the call to the _instance."""
        def delegator(*args, **kwargs):
            return getattr(cls._instance, method_name)(*args, **kwargs)
        return delegator




    #
    # @classmethod
    # def _map_instance_properties(cls):
    #     """Automatically maps all properties from the instance to the Solver class."""
    #     for name, attribute in cls._instance.__class__.__dict__.items():
    #         if isinstance(attribute, property):
    #             # Map property to Solver class
    #             if not hasattr(cls, name):
    #                 setattr(cls, name, cls._create_property_delegator(name))
    #
    # @classmethod
    # def _create_property_delegator(cls, property_name):
    #     """Create a property that delegates access to the _instance."""
    #     def getter(self):
    #         return getattr(self._instance, property_name)
    #
    #     def setter(self, value):
    #         setattr(self._instance, property_name, value)
    #
    #     def deleter(self):
    #         delattr(self._instance, property_name)
    #
    #     # Return a property with the mapped getter, setter, and deleter
    #     return property(getter, setter, deleter)
