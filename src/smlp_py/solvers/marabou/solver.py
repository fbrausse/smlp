import pysmt

from src.smlp_py.NN_verifiers.verifiers import MarabouVerifier
from src.smlp_py.smtlib.text_to_sympy import TextToPysmtParser
from src.smlp_py.solvers.abstract_solver import AbstractSolver, ClassProperty
from src.smlp_py.solvers.marabou.operations import PYSMTOperations
from pysmt.shortcuts import Real


class Pysmt_Solver(AbstractSolver, PYSMTOperations):
    verifier = None
    temp_solver = None

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

        rad_term = float(rad_term)
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
        temp = kwargs.get('temp', False)
        if temp:
            self.temp_solver = MarabouVerifier(parser=self.verifier.parser,
                                     variable_ranges=self.verifier.variable_ranges,
                                     is_temp=True)

        else:
            self.verifier.reset()
        return self

    def add_formula(self, formula, **kwargs):
        need_simplification = kwargs.get("need_simplification", False)
        self.verifier.apply_restrictions(formula, need_simplification=need_simplification)

    def check(self, *args, **kwargs):
        temp = kwargs.get("temp", False)
        if temp:
            result = self.temp_solver.solve()
            self.temp_solver = None
            return result
        else:
            return self.verifier.solve()

    def generate_theta(self, *args, **kwargs):
        pass

    def add_not_query(self, *args, **kwargs):
        query = kwargs["query"]
        temp = kwargs.get("temp", False)


    def create_counter_example(self, *args, **kwargs):
        formulas = kwargs["formulas"]
        query = kwargs["query"]

        self.temp_solver = MarabouVerifier( parser=self.verifier.parser,
                                            variable_ranges=self.verifier.variable_ranges,
                                            is_temp=True)
        for formula in formulas:
            self.temp_solver.apply_restrictions(formula)

        negation = self.temp_solver.parser.propagate_negation(query)
        z3_equiv = self.temp_solver.parser.handle_ite_formula(negation, handle_ite=False)
        self.temp_solver.apply_restrictions(negation, need_simplification=True)
        return self

    def substitute(self, *args, **kwargs):
        var = kwargs["var"]
        substitutions = kwargs["substitutions"]

        for x in list(substitutions.keys()):
            temp = substitutions[x]
            del substitutions[x]
            substitutions[self.smlp_var(x)] = temp

        return self.simplify(var.substitute(substitutions))