import smlp
from src.smlp_py.solvers.abstract_solver import AbstractSolver
from src.smlp_py.solvers.z3.operations import SMLPOperations


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

        return smlp.subst(objv_term, stable_witness_terms)

    def generate_rad_term(self, *args, **kwargs):
        rad = kwargs["rad"]
        delta_rel = kwargs["delta_rel"]
        var_term = kwargs["var_term"]
        candidate = kwargs["candidate"]

        rad_term = self.smlp_cnst(rad)
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

        return self.smlp_and(theta_form, ((abs(var_term - witness)) <= rad_term))

    def get_rad_term(self, *args, **kwargs):
        rad = kwargs["rad"]
        return self.smlp_cnst(rad)

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

        self.verifier = create_solver(domain, model_full_term_dict, incremental, solver_logic)
        return self

    def add_formula(self, *args, **kwargs):
        formula = kwargs["formula"]

        self.verifier.add(formula)

    def check(self):
        return self.verifier.check(), None

    def generate_theta(self, *args, **kwargs):
        pass

    def create_counter_example(self, *args, **kwargs):
        formulas = kwargs["formulas"]
        query = kwargs["query"]

        self.create_solver(*args, **kwargs)
        for formula in formulas:
            self.add_formula(formula)

        self.add_formula(self.smlp_not(query))
        return self

    def substitute(self, *args, **kwargs):
        var = kwargs["var"]
        substitutions = kwargs["substitutions"]

        return self.smlp_cnst_fold(var, substitutions)
