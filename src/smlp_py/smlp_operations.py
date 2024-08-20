import pysmt.shortcuts
import smlp
import functools
import operator as op

from pysmt.fnode import FNode

USE_CACHE = False


def conditional_cache(func):
    """Custom decorator to conditionally apply @functools.cache."""
    if USE_CACHE:
        # Apply caching
        return functools.cache(func)
    else:
        # Return the original function without caching
        return func


class SMLPOperations:
    @property
    @conditional_cache  # @functools.cache
    def smlp_true(self):
        return smlp.true

    @property
    @conditional_cache  # @functools.cache
    def smlp_false(self):
        return smlp.false

    @property
    @conditional_cache  # @functools.cache
    def smlp_real(self):
        return smlp.Real

    @property
    @conditional_cache  # @functools.cache
    def smlp_integer(self):
        return smlp.Integer

    @conditional_cache  # @functools.cache
    def smlp_var(self, var):
        return smlp.Var(var)

    @conditional_cache  # @functools.cache
    def smlp_cnst(self, const):
        return smlp.Cnst(const)

    # rationals
    @conditional_cache  # @functools.cache
    def smlp_q(self, const):
        return smlp.Q(const)

    # reals
    @conditional_cache  # @functools.cache
    def smlp_r(self, const):
        return smlp.R(const)

    # logical not (logic negation)
    @conditional_cache  # @functools.cache
    def smlp_not(self, form: smlp.form2):
        # res1 = ~form
        res2 = op.inv(form)
        # assert res1 == res2
        return res2  # ~form

    # logical and (conjunction)
    @conditional_cache  # @functools.cache
    def smlp_and(self, form1: smlp.form2, form2: smlp.form2):
        ''' test 83 gets stuck with this simplification
        if form1 == smlp.true:
            return form2
        if form2 == smlp.true:
            return form1
        '''
        res1 = op.and_(form1, form2)
        # res2 = form1 & form2
        # print('res1', res1, type(res1)); print('res2', res2, type(res2))
        # assert res1 == res2
        return res1  # form1 & form2

    # conjunction of possibly more than two formulas
    # @functools.cache -- error: unhashable type: 'list'
    def smlp_and_multi(self, form_list: list[smlp.form2]):
        res = self.smlp_true
        '''
        for i, form in enumerate(form_list):
            res = form if i == 0 else self.smlp_and(res, form)
        '''
        for form in form_list:
            res = form if res is self.smlp_true else self.smlp_and(res, form)
        return res

    # logical or (disjunction)
    @conditional_cache  # @functools.cache
    def smlp_or(self, form1: smlp.form2, form2: smlp.form2):
        res1 = op.or_(form1, form2)
        # res2 = form1 | form2
        # assert res1 == res2
        return res1  # form1 | form2

    # disjunction of possibly more than two formulas
    # @functools.cache -- error: unhashable type: 'list'
    def smlp_or_multi(self, form_list: list[smlp.form2]):
        res = self.smlp_false
        '''
        for i, form in enumerate(form_list):
            res = form if i == 0 else self.smlp_or(res, form)
        '''
        for form in form_list:
            res = form if res is self.smlp_false else self.smlp_or(res, form)
        return res

    # logical implication
    @conditional_cache  # @functools.cache
    def smlp_implies(self, form1: smlp.form2, form2: smlp.form2):
        return self.smlp_or(self.smlp_not(form1), form2)

    # addition
    @conditional_cache  # @functools.cache
    def smlp_add(self, term1: smlp.term2, term2: smlp.term2):
        return op.add(term1, term2)

    # sum of possibly more than two formulas
    # @functools.cache -- error: unhashable type: 'list'
    def smlp_add_multi(self, term_list: list[smlp.term2]):
        for i, term in enumerate(term_list):
            res = term if i == 0 else self.smlp_add(res, term)
        return res

    # subtraction
    @conditional_cache  # @conditional_cache #@functools.cache
    def smlp_sub(self, term1: smlp.term2, term2: smlp.term2):
        return op.sub(term1, term2)

    # multiplication
    @conditional_cache  # @functools.cache
    def smlp_mult(self, term1: smlp.term2, term2: smlp.term2):
        return op.mul(term1, term2)

        # TODO: !!!  check that term2 does not evaluate to term 0 ???

    # Do this before calling smlp_div, whenver possible?
    @conditional_cache  # @functools.cache
    def smlp_div(self, term1: smlp.term2, term2: smlp.term2):
        # return self.smlp_mult(self.smlp_cnst(self.smlp_q(1)) / term2, term1)
        return self.smlp_mult(op.truediv(self.smlp_cnst(self.smlp_q(1)), term2), term1)

    @conditional_cache  # @functools.cache
    def smlp_pow(self, term1: smlp.term2, term2: smlp.term2):
        return op.pow(term1, term2)

    # equality
    @conditional_cache  # @functools.cache
    def smlp_eq(self, term1: smlp.term2, term2: smlp.term2):
        res1 = op.eq(term1, term2)
        # res2 = term1 == term2; print('res1', res1, 'res2', res2)
        # assert res1 == res2
        return res1

    # operator != (not equal)
    @conditional_cache  # @functools.cache
    def smlp_ne(self, term1: smlp.term2, term2: smlp.term2):
        res1 = op.ne(term1, term2)
        # res2 = term1 != term2; print('res1', res1, 'res2', res2)
        # assert res1 == res2
        return res1

    # operator <
    @conditional_cache  # @functools.cache
    def smlp_lt(self, term1: smlp.term2, term2: smlp.term2):
        return op.lt(term1, term2)

        # operator <=

    @conditional_cache  # @functools.cache
    def smlp_le(self, term1: smlp.term2, term2: smlp.term2):
        return op.le(term1, term2)

    # operator >
    @conditional_cache  # @functools.cache
    def smlp_gt(self, term1: smlp.term2, term2: smlp.term2):
        return op.gt(term1, term2)

        # operator >=

    @conditional_cache  # @functools.cache
    def smlp_ge(self, term1: smlp.term2, term2: smlp.term2):
        return op.ge(term1, term2)

    # if-thne-else operation
    @conditional_cache  # @functools.cache
    def smlp_ite(self, form: smlp.form2, term1: smlp.term2, term2: smlp.term2):
        return smlp.Ite(form, term1, term2)

    # this function performs substitution of variables in term2:
    # it substitutes occurrences of the keys in subst_dict with respective values, in term2 term.
    # @functools.cache
    def smlp_subst(self, term: smlp.term2, subst_dict: dict):
        return smlp.subst(term, subst_dict)

    # simplifies a ground term to the respective constant; takes als a s
    # @functools.cache
    def smlp_cnst_fold(self, term: smlp.term2, subst_dict: dict):
        return smlp.cnst_fold(term, subst_dict)

class ClassProperty:
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, instance, owner):
        return self.fget(owner)

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
