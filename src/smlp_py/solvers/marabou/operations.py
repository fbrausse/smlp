import pysmt.shortcuts
import smlp
from pysmt.fnode import FNode
from src.smlp_py.solvers.abstract_solver import ClassProperty, conditional_cache


class PYSMTOperations:

    @ClassProperty
    def smlp_true(cls):
        return pysmt.shortcuts.TRUE()

    @ClassProperty
    def smlp_false(cls):
        return pysmt.shortcuts.FALSE()

    @ClassProperty
    def smlp_real(cls):
        return pysmt.shortcuts.Real

    @ClassProperty
    def smlp_integer(cls):
        return pysmt.shortcuts.Int

    @conditional_cache  # @functools.cache
    def smlp_cnst(cls, const):
        if isinstance(const, FNode):
            return const
        return pysmt.shortcuts.Real(const)

    # logical not (logic negation)
    @conditional_cache  # @functools.cache
    def smlp_not(cls, form: FNode):
        return pysmt.shortcuts.Not(form)

    # logical and (conjunction)
    @conditional_cache  # @functools.cache
    def smlp_and(cls, form1: FNode, form2: FNode):
        return pysmt.shortcuts.And(form1, form2)  # form1 & form2

    def smlp_and_multi(cls, form_list: list[FNode]):
        return pysmt.shortcuts.And(*form_list)

    # logical or (disjunction)
    @conditional_cache  # @functools.cache
    def smlp_or(cls, form1: FNode, form2: FNode):
        return pysmt.shortcuts.Or(form1, form2)

    def smlp_or_multi(cls, form_list: list[FNode]):
        return pysmt.shortcuts.Or(*form_list)

    def smlp_eq(self, term1: smlp.term2, term2: smlp.term2):
        return pysmt.shortcuts.Equals(term1, term2)

    def smlp_q(self, const):
        return pysmt.shortcuts.Real(const)

    def smlp_mult(self, *args):
        return pysmt.shortcuts.Times(*args)