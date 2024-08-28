# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

import operator
import numpy as np
import pandas as pd
import keras
from sklearn.tree import _tree
import json
import ast
import builtins
import operator as op
from fractions import Fraction
import time
import functools #for cacheing
from collections import defaultdict
import sys
from icecream import ic
ic.configureOutput(prefix=f'Debug | ', includeContext=True)
ic("Changes here and line 355")
from keras.models import Sequential
import smlp
from smlp_py.smlp_utils import (np_JSONEncoder, lists_union_order_preserving_without_duplicates, 
    list_subtraction_set, get_expression_variables, str_to_bool)
#from smlp_py.smlp_spec import SmlpSpec


# TODO !!! create a parent class for TreeTerms, PolyTerms, NNKerasTerms.
# setting logger, report_file_prefix, model_file_prefix can go to that class to work for all above three classes

# The classes in this module contain methods to generate terms from trained tree, polynomial or NN models.
# Further, some of the methods convert scaling and unscaling constraints to terms so that after composing
# generation of terms for models with feature and or responses scaling constraints the final term for each
# model is a term with features as inputs and responses as outputs -- that is, they are expressed in terms 
# of SMLP variables that are declared in the solver domain. To parse expressions for constraints, assertions,
# optimization objectives, etc. from command line or a spec file, AST (Abstract Syntax Trees) module is used,
# and it supports variables, constants, unary and binary operators that are supported in SMLP to build terms
# and formulas. Division is supported only when denominator is integer, and pow function is only supported 
# when the exponent is integer -- both are modelled by multiplication and fractions (in case of division).

# Model training parameter model_per_response controls whether one model is build that covers all responses
# or a separate model is built for each response. In the latter case, when MRMR option mrmr_pred is on,
# model for each response is built from the subset of features selected by MRMR algorithm for that 
# response -- these subsets of features might be different for different responses. Also, in this case
# (when model_per_response is true) result of training is a dictionary with response names as keys and
# the model trained for a given response as the corresponding value in the dictionary. When model_per_response
# is false, the trained model is not a dictionary, it is a model of the type that corresponds to the training
# algorithm; and in this case the features used for training are all features as specified in command line
# if MRMR is not used, and otherwise is the union of features selected by MRMR for at least one response.
# In model exploration modes (lie verification, querying, optimization) if SMLP terms and solver instances
# need to be built, each model in the dictionary of the models or a model trained for all responses is 
# converted to terms separately, the constraints and assertions built on the top of model responses are added
# to solver instance separately (as many as required, depending on whether all responses are analysed together).


USE_CACHE = False

def conditional_cache(func):
    """Custom decorator to conditionally apply @functools.cache."""
    if USE_CACHE:
        # Apply caching
        return functools.cache(func)
    else:
        # Return the original function without caching
        return func
'''

def conditional_cache(func):
    def wrapper(self, *args, **kwargs):
        if self._cache_terms:
            if not hasattr(self, '_cache'):
                self._cache = {}
            if func not in self._cache:
                self._cache[func] = func(self, *args, **kwargs)
            return self._cache[func]
        else:
            return func(self, *args, **kwargs)
    return wrapper
'''

# Class SmlpTerms has methods for generating terms, and classes TreeTerms, PolyTerms and NNKerasTerms are inherited
# from it but this inheritance is probably not implemented in the best way: TODO !!!: see if that can be improved.
class SmlpTerms:
    def __init__(self):
        self._smlp_terms_logger = None
        self.report_file_prefix = None
        self.model_file_prefix = None
        #self._cache_terms = False
        
        # supported operators in ast module for expression parsing and transformation
        # https://docs.python.org/3/library/ast.html -- AST (Abstract Syntax Trees)
        # OD version of AST documentation: https://docs.python.org/2/library/ast.html
        # Python operators: https://www.w3schools.com/python/python_operators.asp
        # https://docs.python.org/3/library/operator.html -- python operators documentation
        '''
        boolop = And | Or 
        operator = Add | Sub | Mult | Div | Mod | Pow | LShift 
                     | RShift | BitOr | BitXor | BitAnd | FloorDiv
        unaryop = Invert | Not | UAdd | USub
        cmpop = Eq | NotEq | Lt | LtE | Gt | GtE | Is | IsNot | In | NotIn
        comprehension = (expr target, expr iter, expr* ifs)
            more grops of operations are supported, such as expr_context (load | store | Del),
            excepthandler, arguments, arg, keyword, alias, withitem, match_case, pattern, 
            type_ignore, type_param
            ast.Is: op.is_, ast.IsNot: op.is_not, ast.In: op.in, ast.NotIn: op.not_in,
        '''
        # op.inv is equivalent to bitwise (= logocal) negation '~'
        # op.and_ and op.or_ are equivalent to bitvise (=logical) & and |, respectively. 
        # Python keywords 'and', 'or', 'mot' shpuld not be used as logic operators when
        # building smlp terms and formulas -- op.and_, op.or_, and op.inv should be used.
        self._ast_operators_map = {
            ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow,
            ast.BitXor: op.xor,
            ast.USub: op.neg,
            ast.Eq: op.eq, ast.NotEq: op.ne, ast.Lt: op.lt, ast.LtE: op.le, ast.Gt: op.gt, ast.GtE: op.ge, 
            ast.And: op.and_, ast.Or: op.or_, ast.Not: op.inv,
            ast.IfExp: smlp.Ite
        } 

        # maps python operators from module operator to correponding smlp/smlplib operations; 
        # used in recursive construction of smlp terms and formulas.
        # TODO !!! operations BitXor and USub are not used, they are not mapped to smlp operations
        self._ast_operators_smlp_map = {
            ast.Add: self.smlp_add, ast.Sub: self.smlp_sub, ast.Mult: self.smlp_mult, 
            ast.Div: self.smlp_div, ast.Pow: self.smlp_pow,
            ast.BitXor: op.xor,
            ast.USub: op.neg,
            ast.Eq: self.smlp_eq, ast.NotEq: self.smlp_ne, ast.Lt: self.smlp_lt, 
            ast.LtE: self.smlp_le, ast.Gt: self.smlp_gt, ast.GtE: self.smlp_ge, 
            ast.And: self.smlp_and, ast.Or: self.smlp_or, ast.Not: self.smlp_not,
            ast.IfExp: self.smlp_ite
        } 
        
    # set logger from a caller script
    def set_logger(self, logger):
        self._smlp_terms_logger = logger
    
    @property
    def ast_operators_smlp_map(self):
        return {
            ast.Add: self.smlp_add, ast.Sub: self.smlp_sub, ast.Mult: self.smlp_mult, 
            ast.Div: self.smlp_div, ast.Pow: self.smlp_pow,
            ast.BitXor: op.xor,
            ast.USub: op.neg,
            ast.Eq: self.smlp_eq, ast.NotEq: self.smlp_ne, ast.Lt: self.smlp_lt, 
            ast.LtE: self.smlp_le, ast.Gt: self.smlp_gt, ast.GtE: self.smlp_ge, 
            ast.And: self.smlp_and, ast.Or: self.smlp_or, ast.Not: self.smlp_not,
            ast.IfExp: self.smlp_ite
        } #self._ast_operators_smlp_map
    
    
    #def set_cache_terms(self, cache_terms):
    #    self._cache_terms = cache_tems
    
    @property
    @conditional_cache #@functools.cache
    def smlp_true(self):
        return smlp.true

    @property
    @conditional_cache #@functools.cache
    def smlp_false(self):
        return smlp.false

    @property
    @conditional_cache #@functools.cache
    def smlp_real(self):
        return smlp.Real
    
    @property
    @conditional_cache #@functools.cache
    def smlp_integer(self):
        return smlp.Integer

    @conditional_cache #@functools.cache
    def smlp_var(self, var):
        return smlp.Var(var)
    
    @conditional_cache #@functools.cache
    def smlp_cnst(self, const):
        return smlp.Cnst(const)
    
    # rationals
    @conditional_cache #@functools.cache
    def smlp_q(self, const):
        return smlp.Q(const)
    
    # reals
    @conditional_cache #@functools.cache
    def smlp_r(self, const):
        return smlp.R(const)
    
    # logical not (logic negation)
    @conditional_cache #@functools.cache
    def smlp_not(self, form:smlp.form2):
        #res1 = ~form
        res2 = op.inv(form)
        #assert res1 == res2
        return res2 #~form
    
    # logical and (conjunction)
    @conditional_cache #@functools.cache
    def smlp_and(self, form1:smlp.form2, form2:smlp.form2):
        ''' test 83 gets stuck with this simplification
        if form1 == smlp.true:
            return form2
        if form2 == smlp.true:
            return form1
        '''
        res1 = op.and_(form1, form2)
        ic("Here")
        #ic(res1)
        #ic(form1, form2)
        #print('res1', res1, type(res1)); print('res2', res2, type(res2))
        #assert res1 == res2
        return res1 # form1 & form2
    
    # conjunction of possibly more than two formulas
    #@functools.cache -- error: unhashable type: 'list'
    def smlp_and_multi(self, form_list:list[smlp.form2]):
        res = self.smlp_true
        '''
        for i, form in enumerate(form_list):
            res = form if i == 0 else self.smlp_and(res, form)
        '''
        for form in form_list:
            res = form if res is self.smlp_true else self.smlp_and(res, form)
        return res
    
    # logical or (disjunction)
    @conditional_cache #@functools.cache
    def smlp_or(self, form1:smlp.form2, form2:smlp.form2):
        res1 = op.or_(form1, form2)
        #res2 = form1 | form2
        #assert res1 == res2
        return res1 #form1 | form2
    
    # disjunction of possibly more than two formulas
    #@functools.cache -- error: unhashable type: 'list'
    def smlp_or_multi(self, form_list:list[smlp.form2]):
        res = self.smlp_false
        '''
        for i, form in enumerate(form_list):
            res = form if i == 0 else self.smlp_or(res, form)
        '''
        for form in form_list:
            res = form if res is self.smlp_false else self.smlp_or(res, form)
        return res
    
    # logical implication
    @conditional_cache #@functools.cache
    def smlp_implies(self, form1:smlp.form2, form2:smlp.form2):
        return self.smlp_or(self.smlp_not(form1), form2)

    
    # addition
    @conditional_cache #@functools.cache
    def smlp_add(self, term1:smlp.term2, term2:smlp.term2):
        return op.add(term1, term2)
    
    # sum of possibly more than two formulas
    #@functools.cache -- error: unhashable type: 'list'
    def smlp_add_multi(self, term_list:list[smlp.term2]):
        for i, term in enumerate(term_list):
            res = term if i == 0 else self.smlp_add(res, term)                
        return res
    
    # subtraction
    @conditional_cache #@conditional_cache #@functools.cache
    def smlp_sub(self, term1:smlp.term2, term2:smlp.term2):
        return op.sub(term1, term2)
    
    # multiplication
    @conditional_cache #@functools.cache
    def smlp_mult(self, term1:smlp.term2, term2:smlp.term2):
        return op.mul(term1, term2)    
    
    # TODO: !!!  check that term2 does not evaluate to term 0 ???
    # Do this before calling smlp_div, whenver possible?
    @conditional_cache #@functools.cache
    def smlp_div(self, term1:smlp.term2, term2:smlp.term2):
        #return self.smlp_mult(self.smlp_cnst(self.smlp_q(1)) / term2, term1)
        return self.smlp_mult(op.truediv(self.smlp_cnst(self.smlp_q(1)), term2), term1)
    
    @conditional_cache #@functools.cache
    def smlp_pow(self, term1:smlp.term2, term2:smlp.term2):
        return op.pow(term1, term2)
    
    # equality
    @conditional_cache #@functools.cache
    def smlp_eq(self, term1:smlp.term2, term2:smlp.term2):
        res1 = op.eq(term1, term2)
        #res2 = term1 == term2; print('res1', res1, 'res2', res2)
        #assert res1 == res2
        return res1
    
    # operator != (not equal)
    @conditional_cache #@functools.cache
    def smlp_ne(self, term1:smlp.term2, term2:smlp.term2):
        res1 = op.ne(term1, term2)
        #res2 = term1 != term2; print('res1', res1, 'res2', res2)
        #assert res1 == res2
        return res1
        
    # operator <
    @conditional_cache #@functools.cache
    def smlp_lt(self, term1:smlp.term2, term2:smlp.term2):
        return op.lt(term1, term2)    
    
    # operator <=
    @conditional_cache #@functools.cache
    def smlp_le(self, term1:smlp.term2, term2:smlp.term2):
        return op.le(term1, term2)
    
    # operator >
    @conditional_cache #@functools.cache
    def smlp_gt(self, term1:smlp.term2, term2:smlp.term2):
        return op.gt(term1, term2)    
    
    # operator >=
    @conditional_cache #@functools.cache
    def smlp_ge(self, term1:smlp.term2, term2:smlp.term2):
        return op.ge(term1, term2)
    
    # if-thne-else operation
    @conditional_cache #@functools.cache
    def smlp_ite(self, form:smlp.form2, term1:smlp.term2, term2:smlp.term2):
        return smlp.Ite(form, term1, term2)
    
    # this function performs substitution of variables in term2:
    # it substitutes occurrences of the keys in subst_dict with respective values, in term2 term.
    #@functools.cache
    def smlp_subst(self, term:smlp.term2, subst_dict:dict):
        return smlp.subst(term, subst_dict)
    
    # simplifies a ground term to the respective constant; takes als a s
    #@functools.cache
    def smlp_cnst_fold(self, term:smlp.term2, subst_dict:dict):
        return smlp.cnst_fold(term, subst_dict)
    
    '''
    destruct(e: smlp.libsmlp.form2 | smlp.libsmlp.term2) -> dict
    Destructure the given term2 or form2 instance `e`. The result is a dict
    with the following entries:
    - 'id': always, one of:
      - term2: 'var', 'add', 'sub', 'mul', 'uadd', 'usub', 'const', 'ite'
      - form2: 'prop', 'and', 'or', 'not'
    - 'args': operands to this operation as a tuple (or list in case of
              'and' and 'or') of term2 and/or form2 objects (all except
              'var', 'const'),
    - 'name': name of symbol ('var' only)
    - 'type': type of term2 constant, one of: 'Z', 'Q', 'A' ('const' only)
    - 'cmp': comparison predicate, one of: '<=', '<', '>=', '>', '==', '!='
             ('prop' only)
    '''
    def smlp_destruct(self, term2_or_form2): #:smlp.term2|smlp.form2
        #ic("Changes here...")
        #sys.setrecursionlimit(15000)
        return smlp.destruct(term2_or_form2)
    
    # this function traverses an object of type smlp.libsmlp.form2 or smlp.libsmlp.term2 
    # and returns a dictionary with the counts of each operator encountered during the traversal.
    def smlp_count_operators(self, e):
        #return {}
        # Initialize a dictionary to store the counts of operators
        operator_counts = defaultdict(int)

        # Define a helper function to traverse the object
        def traverse(obj):
            # Destructure the given object
            # ic("Changes here...")
            sys.setrecursionlimit(20000)
            destructure_result = self.smlp_destruct(obj)

            # Increment the count of the current operator
            operator_counts[destructure_result['id']] += 1

            # If there are arguments, recursively traverse them
            if 'args' in destructure_result:
                for arg in destructure_result['args']:
                    traverse(arg)

        # Start the traversal with the input object
        traverse(e)
        
        return dict(operator_counts)
    
    # Example usage:
    # Assuming operator_counts_list is a list of dictionaries with operator counts
    # summed_operator_counts = sum_operator_counts(operator_counts_list)
    # print(summed_operator_counts)
    def sum_operator_counts(self, operator_counts_list):
        # Initialize a dictionary to store the sum of operator counts
        summed_counts = {}

        # Iterate over each operator_counts dictionary in the list
        for operator_counts in operator_counts_list:
            # Iterate over each key-value pair in the current dictionary
            for operator, count in operator_counts.items():
                # Add the count to the summed_counts dictionary
                if operator in summed_counts:
                    summed_counts[operator] += count
                else:
                    summed_counts[operator] = count

        return summed_counts


    # this function doesn't take substitutions, but in addition to whatever cnst_fold() is doing, 
    # it simplifies arithmetics and con-/disjunctions and negations:
    # simplify() uses the following axioms:
    # - variables remain variables
    # - rationals get canonicalized (that means common divisor of numerator and denominator
    #         removed); and then: if denominator is 1, replace by integer
    # - addition/subtraction by zero
    # - multiplication by zero or one
    # - negation of zero or constants get simplified (folded into the constant)
    # - if-then-else gets simplified if condition can be simplified to a constant
    # - comparisons between constants get simplified
    # - (and ...) with one constant false operand
    # - (or ...) with one true operand
    # - empty (and) and (or)
    # - logical negation of constants
    #@functools.cache
    def smlp_simplify(self, term:smlp.term2):
        #print('term to simplify\n', term); print('result\n', smlp.simplify(term))
        return smlp.simplify(term)
    
    # https://stackoverflow.com/questions/68390248/ast-get-the-list-of-only-the-variable-names-in-an-expression
    def get_expression_variables(self, expression):
        tree = ast.parse(expression)
        variables = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                variables.append(node.id)
        return tuple(v for v in set(variables) if v not in vars(builtins))
    
    # compute smlp term for strings that represent python expressions, based on code from
    # https://stackoverflow.com/questions/2371436/evaluating-a-mathematical-expression-in-a-string
    # modified to generate SMLP terms.
    # reference to tres in python (not used currently) 
    # https://www.tutorialspoint.com/python_data_structure/python_binary_tree.htm
    # https://docs.python.org/3/library/operator.html -- python operators documentation
    def ast_expr_to_term2(self, expr):
        #print('evaluating AST expression ====', expr)
        assert isinstance(expr, str)
        # recursion
        def eval_(node):
            if isinstance(node, ast.Num): # <number>
                #print('node Num', node.n, type(node.n))
                return smlp.Cnst(node.n)
            elif isinstance(node, ast.BinOp): # <left> <operator> <right>
                #print('node BinOp', node.op, type(node.op))
                if type(node.op) not in [ast.Div, ast.Pow]:
                    return self._ast_operators_map[type(node.op)](eval_(node.left), eval_(node.right))
                elif type(node.op) == ast.Div:
                    if type(node.right) == ast.Constant:
                        if node.right.n == 0:
                            raise Exception('Division by 0 in parsed expression ' + expr)
                        elif not isinstance(node.right.n, int):
                            raise Exception('Division in parsed expression is only supported for integer constants; got ' + expr)
                        else:
                            #print('node.right.n', node.right.n, type(node.right.n))
                            return self._ast_operators_map[ast.Mult](smlp.Cnst(smlp.Q(1) / smlp.Q(node.right.n)), eval_(node.left))
                    else: 
                        raise Exception('Opreator ' + str(self._ast_operators_map[type(node.op)]) + 
                            ' with non-constant demominator within ' + str(expr) + ' is not supported in ast_expr_to_term')
                elif type(node.op) == ast.Pow:
                    if type(node.right) == ast.Constant:
                        if type(node.right.n) == int:
                            #print('node.right.n', node.right.n)
                            if node.right.n == 0:
                                return smlp.Cnst(1)
                            elif node.right.n > 0:
                                left_term = res_pow = eval_(node.left)
                                for i in range(1, node.right.n):
                                    res_pow = op.mul(res_pow, left_term)
                                #print('res_pow', res_pow)
                                return res_pow
                    raise Exception('Opreator ' + str(self._ast_operators_map[type(node.op)]) + 
                                    ' with non-constant or negative exponent within ' + 
                                    str(expr) + 'is not supported in ast_expr_to_term')
                else:
                    raise Exception('Implementation error in function ast_expr_to_term')
            elif isinstance(node, ast.UnaryOp): # <operator> <operand> e.g., -1
                #print('unary op', node.op, type(node.op)); 
                return self._ast_operators_map[type(node.op)](eval_(node.operand))
            elif isinstance(node, ast.Name): # variable
                #print('node Var', node.id, type(node.id))
                return smlp.Var(node.id)
            elif isinstance(node, ast.BoolOp):
                # Say if BoolOp is op.And, whne there is a (sub-)formula that is conjunction of more than two
                # conjuncts, say a > 5 and b < 3 and b > 0, then this is detected by AST parser as conjunction
                # with three arguments given as list node.values [a > 5, b < 3, b > 0]. We build the 
                # corresponding smlp formula by applying two-argument conjunction in relevant number of times.
                #print('node BoolOp', node.op, type(node.op), 'values', node.values, type(node.values));
                res_boolop = self._ast_operators_map[type(node.op)](eval_(node.values[0]), eval_(node.values[1]))
                if len(node.values) > 2:
                    for i in range(2, len(node.values)):
                        res_boolop = self._ast_operators_map[type(node.op)](res_boolop, eval_(node.values[i]))
                #print('res_boolop', res_boolop)
                return res_boolop
            elif isinstance(node, ast.Compare):
                #print('node Compare', node.ops, type(node.ops), 'left', node.left, 'comp', node.comparators);
                #print('len ops', len(node.ops), 'len comparators', len(node.comparators))
                assert len(node.ops) == len(node.comparators)
                left_term_0 = eval_(node.left)
                right_term_0 = eval_(node.comparators[0])
                res_comp = self._ast_operators_map[type(node.ops[0])](left_term_0, right_term_0); #print('res_comp_0', res_comp)
                if len(node.ops) > 1:
                    #print('enum', list(range(1, len(node.ops))))
                    left_term_i = right_term_0
                    for i in range(1, len(node.ops)):
                        right_term_i = eval_(node.comparators[i])
                        #print('i', i, 'left', left_term_i, 'right', right_term_i)
                        res_comp_i = self._ast_operators_map[type(node.ops[i])](left_term_i, right_term_i)
                        res_comp = op.and_(res_comp, res_comp_i) # self._ast_operators_map[type(node.op.And)]
                        # for the next iteration (if any):
                        left_term_i = right_term_i
                #print('res_comp', res_comp)
                return res_comp 
            elif isinstance(node, ast.List):
                self._smlp_terms_logger.error('Parsing expressions with lists is not supported')
                #print('node List', 'elts', node.elts, type(node.elts), 'expr_context', node.expr_context);
                raise Exception('Parsing expressions with lists is not supported')
            elif isinstance(node, ast.Constant):
                if node.n == True:
                    return smlp.true
                if node.n == False:
                    return smlp.false
                raise Exception('Unsupported comstant ' + str(node.n) + ' in funtion ast_expr_to_term')
            elif isinstance(node, ast.IfExp):
                res_test = eval_(node.test)
                res_body = eval_(node.body)
                res_orelse = eval_(node.orelse)
                #res_ifexp = smlp.Ite(res_test, res_body, res_orelse)
                res_ifexp = self._ast_operators_map[ast.IfExp](res_test, res_body, res_orelse)
                #print('res_ifexp',res_ifexp)
                return res_ifexp
            else:
                self._smlp_terms_logger.error('Unexpected node type ' + str(type(node)))
                #print('node type', type(node))
                raise TypeError(node)

        return eval_(ast.parse(expr, mode='eval').body)

    def ast_expr_to_term(self, expr):
        #print('evaluating AST expression ====', expr)
        assert isinstance(expr, str)
        # recursion
        def eval_(node):
            if isinstance(node, ast.Num): # <number>
                #print('node Num', node.n, type(node.n))
                return self.smlp_cnst(node.n)
            elif isinstance(node, ast.BinOp): # <left> <operator> <right>
                #print('node BinOp', node.op, type(node.op))
                if type(node.op) not in [ast.Div, ast.Pow]:
                    return self.ast_operators_smlp_map[type(node.op)](eval_(node.left), eval_(node.right))
                elif type(node.op) == ast.Div:
                    if type(node.right) == ast.Constant:
                        if node.right.n == 0:
                            raise Exception('Division by 0 in parsed expression ' + expr)
                        elif not isinstance(node.right.n, int):
                            raise Exception('Division in parsed expression is only supported for integer constants; got ' + expr)
                        else:
                            #print('node.right.n', node.right.n, type(node.right.n))
                            return self.ast_operators_smlp_map[ast.Mult](self.smlp_cnst(self.smlp_q(1) / self.smlp_q(node.right.n)), eval_(node.left))
                    else: 
                        raise Exception('Opreator ' + str(self.ast_operators_smlp_map[type(node.op)]) + 
                            ' with non-constant demominator within ' + str(expr) + ' is not supported in ast_expr_to_term')
                elif type(node.op) == ast.Pow:
                    if type(node.right) == ast.Constant:
                        if type(node.right.n) == int:
                            #print('node.right.n', node.right.n)
                            if node.right.n == 0:
                                return self.smlp_cnst(1)
                            elif node.right.n > 0:
                                left_term = res_pow = eval_(node.left)
                                for i in range(1, node.right.n):
                                    res_pow = op.mul(res_pow, left_term)
                                #print('res_pow', res_pow)
                                return res_pow
                    raise Exception('Opreator ' + str(self.ast_operators_smlp_map[type(node.op)]) + 
                                    ' with non-constant or negative exponent within ' + 
                                    str(expr) + 'is not supported in ast_expr_to_term')
                else:
                    raise Exception('Implementation error in function ast_expr_to_term')
            elif isinstance(node, ast.UnaryOp): # <operator> <operand> e.g., -1
                #print('unary op', node.op, type(node.op)); 
                return self.ast_operators_smlp_map[type(node.op)](eval_(node.operand))
            elif isinstance(node, ast.Name): # variable
                #print('node Var', node.id, type(node.id))
                return self.smlp_var(node.id)
            elif isinstance(node, ast.BoolOp):
                # Say if BoolOp is op.And, whne there is a (sub-)formula that is conjunction of more than two
                # conjuncts, say a > 5 and b < 3 and b > 0, then this is detected by AST parser as conjunction
                # with three arguments given as list node.values [a > 5, b < 3, b > 0]. We build the 
                # corresponding smlp formula by applying two-argument conjunction in relevant number of times.
                #print('node BoolOp', node.op, type(node.op), 'values', node.values, type(node.values));
                res_boolop = self.ast_operators_smlp_map[type(node.op)](eval_(node.values[0]), eval_(node.values[1]))
                if len(node.values) > 2:
                    for i in range(2, len(node.values)):
                        res_boolop = self.ast_operators_smlp_map[type(node.op)](res_boolop, eval_(node.values[i]))
                #print('res_boolop', res_boolop)
                return res_boolop
            elif isinstance(node, ast.Compare):
                #print('node Compare', node.ops, type(node.ops), 'left', node.left, 'comp', node.comparators);
                #print('len ops', len(node.ops), 'len comparators', len(node.comparators))
                assert len(node.ops) == len(node.comparators)
                left_term_0 = eval_(node.left)
                right_term_0 = eval_(node.comparators[0])
                res_comp = self.ast_operators_smlp_map[type(node.ops[0])](left_term_0, right_term_0); #print('res_comp_0', res_comp)
                if len(node.ops) > 1:
                    #print('enum', list(range(1, len(node.ops))))
                    left_term_i = right_term_0
                    for i in range(1, len(node.ops)):
                        right_term_i = eval_(node.comparators[i])
                        #print('i', i, 'left', left_term_i, 'right', right_term_i)
                        res_comp_i = self.ast_operators_smlp_map[type(node.ops[i])](left_term_i, right_term_i)
                        res_comp = op.and_(res_comp, res_comp_i) # self._ast_operators_smlp_map[type(node.op.And)]
                        # for the next iteration (if any):
                        left_term_i = right_term_i
                #print('res_comp', res_comp)
                return res_comp 
            elif isinstance(node, ast.List):
                self._smlp_terms_logger.error('Parsing expressions with lists is not supported')
                #print('node List', 'elts', node.elts, type(node.elts), 'expr_context', node.expr_context);
                raise Exception('Parsing expressions with lists is not supported')
            elif isinstance(node, ast.Constant):
                if node.n == True:
                    return self.smlp_true
                if node.n == False:
                    return self.smlp_false
                raise Exception('Unsupported comstant ' + str(node.n) + ' in funtion ast_expr_to_term')
            elif isinstance(node, ast.IfExp):
                res_test = eval_(node.test)
                res_body = eval_(node.body)
                res_orelse = eval_(node.orelse)
                #res_ifexp = smlp.Ite(res_test, res_body, res_orelse)
                res_ifexp = self.ast_operators_smlp_map[ast.IfExp](res_test, res_body, res_orelse)
                #print('res_ifexp',res_ifexp)
                return res_ifexp
            else:
                self._smlp_terms_logger.error('Unexpected node type ' + str(type(node)))
                #print('node type', type(node))
                raise TypeError(node)

        return eval_(ast.parse(expr, mode='eval').body)
    
    # Compute numeric values of smlp ground terms, returns a faction (rational number), of type <class 'fractions.Fraction'> or a float.  
    # Enhencement !!!: intend to extend to ground formulas as well. Currently an assertion prevents this usage:
    # assertion checks that the constant expression is rational Q or algebraic A (not a transcendental Real), and also
    # nothing else like Boolean/formula type
    def ground_smlp_expr_to_value(self, ground_term:smlp.term2, approximate=False, precision=64):
        # evaluate to constant term or formula (evaluate all operations in ground_term) -- should
        # succeed because the assumption is that ground_term does not contain variables (is a ground term).
        # The input ground_term and the result smlp_const of smlp.const_fold() are of type <class 'smlp.libsmlp.term2'>.
        #print('ground_term', type(ground_term), ground_term)
        smlp_const = smlp.cnst_fold(ground_term); #print('smlp_const', type(smlp_const), smlp_const)
        assert isinstance(self.smlp_cnst(smlp_const), smlp.libsmlp.Q) or isinstance(self.smlp_cnst(smlp_const), smlp.libsmlp.A) 
        if isinstance(self.smlp_cnst(smlp_const), smlp.libsmlp.A) or isinstance(self.smlp_cnst(smlp_const), smlp.libsmlp.R): 
            #print('algebraic', 'approximate', approximate, 'precision', precision)
            # algebraic number, solution of a polynomial, need to specify precision for the case
            # value_type is not float (for float, precison is always 64); the result var is of type <class 'fractions.Fraction'>
            val = smlp.approx(smlp.Cnst(smlp_const), precision=precision)
        elif isinstance(smlp.Cnst(smlp_const), smlp.libsmlp.Q):
            #print('smlp.libsmlp.Q', 'approximate', approximate, 'precision', precision)
            if approximate:
                val = smlp.approx(smlp.Cnst(smlp_const), precision=precision)
            else:
                try:
                    if self.smlp_cnst(smlp_const).denominator is not None and self.smlp_cnst(smlp_const).numerator is not None:
                        val = Fraction(self.smlp_cnst(smlp_const).numerator, self.smlp_cnst(smlp_const).denominator)
                except Exception as err:
                    self._smlp_terms_logger.error(f"Unexpected {err=}, {type(err)=}")
                    raise
        else:
            raise Exception('Failed to compute value for smlp expression ' + str(expr) + ' of type ' + str(type(expr)))
        
        #print('smlp expr val', type(val), val)
        assert isinstance(val, Fraction) or isinstance(val, float)
        return val
    
    # Converts values in sat assignmenet (witness) from terms to python fractions using function 
    # self.ground_smlp_expr_to_value() -- see the description of that function for more detail.
    # Can also be applied to a dictionary where values are terms.
    def witness_term_to_const(self, witness, approximate=False, precision=64):
        witness_vals_dict = {}
        for k,t in witness.items():
            witness_vals_dict[k] = self.ground_smlp_expr_to_value(t, approximate, precision)
        return witness_vals_dict

    # computes and returns sat assignment witness_approx which approximates input witness/sat assignment 
    # witness with precision lemma_precision. Both in witness and witness_approx, values assigned to
    # model interface variables (inputs, knobs, outputs) are smlp terms (type term2).
    def approximate_witness_term(self, witness, lemma_precision:int, approximate=False, precision=64):
        approx_ca = lemma_precision != 0
        assert lemma_precision >= 0
        if not approx_ca:
            return witness
        witness_approx = {}
        for k, v in witness.items():
            #print('k', k, v)
            v_approx = self.ground_smlp_expr_to_value(v, False, None)
            #print('v_approx', v_approx, type(v_approx), round(v_approx, precision), 'precision', precision, 'approximate', approximate)
            v_round = round(v_approx, lemma_precision)
            assert isinstance(v_round, Fraction)
            witness_approx[k] = self.smlp_cnst(round(v_approx, lemma_precision))
            #witness_approx[k] = self.smlp_div(self.smlp_cnst(v_approx.numerator), self.smlp_cnst(v_approx.denominator))
            #print('witness_approx[k]', witness_approx[k])
        #print('witness_approx', witness_approx)
        return witness_approx
                    
    # Converts values in sat assignmenet (witness) from python fractions to terms.
    def witness_const_to_term(self, witness):
        witness_vals_dict = {}
        for k,t in witness.items():
            witness_vals_dict[k] = self.smlp_cnst(t)
        return witness_vals_dict
    
    # Converts values assugnments in sat assignmenet (witness) from to smlp formula
    # equal to conjucnction of equalities smlp.Var(k) = smlp.Cnst(t)
    def witness_to_formula(self, witness):
        witness_formula = []
        for k,t in witness.items():
            witness_formula.append(self.smlp_eq(self.smlp_var(k), self.smlp_cnst(t)))
        return self.smlp_and_multi(witness_formula)
    
# Methods to generate smlp term and formula from rules associated to branches of an sklearn
# (or caret) regression tree model (should work for classification trees as well, not tested)
class TreeTerms:
    def __init__(self):
        self._smlp_terms_logger = None
        self.model_file_prefix = None
        self.report_file_prefix = None
        self._inequality_ops_dict = {
            '>':op.gt,
            '>=':op.ge,
            '<':op.lt,
            '<=':op.le,
            '==':op.eq,
            '!=':op.ne
        }
        self._inequality_ops_ast_dict = {
            '>':ast.Gt,
            '>=':ast.GtE,
            '<':ast.Lt,
            '<=':ast.LtE,
            '==':ast.Eq,
            '!=':ast.NotEq
        }
        self.instSmlpTerms = SmlpTerms()
        self._compress_rules = None
    
    # set logger from a caller script
    def set_logger(self, logger):
        self._smlp_terms_logger = logger

    def set_compress_rules(self, compress_rules:bool):
        self._compress_rules = compress_rules
    
    def set_tree_encoding(self, tree_encoding:str):
        self._tree_encoding = tree_encoding
        
    def _tree_model_id(self, algo:str, tree_number:int, resp_name=None):
        assert algo is not None
        res1 = '_'.join([algo, 'tree', str(tree_number)]) if tree_number is not None else algo
        res = '_'.join([res1, str(resp_name)]) if resp_name is not None else res1
        return res
    
    def _tree_resp_id(self, tree_number:int, resp_name:str):
        return '_'.join(['tree', str(tree_number), resp_name])
    
    # generate rules from a single decision or regression tree that predicts a single response
    def _get_abstract_rules(self, tree, feature_names, resp_names, class_names, rounding=-1):
        #print('_get_abstract_rules: tree', tree, '\nresp_names', resp_names)
        tree_ = tree.tree_
        #print(tree_.feature) ; print(_tree.TREE_UNDEFINED) ; print(feature_names)
        feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature]
                
        paths = [] # antecedents
        path = [] # current antecedent and consequent (the last element of path : list
        cons = [] # consequents
        #samples = []

        def recurse_tuple(node, path, paths):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                p1, p2 = list(path), list(path)
                p1 += [(name, '<=', threshold)]
                recurse_tuple(tree_.children_left[node], p1, paths)
                p2 += [(name, '>', threshold)]
                recurse_tuple(tree_.children_right[node], p2, paths)
            else:
                #print('node value', tree_.value[node]); print('node samples', tree_.n_node_samples[node])
                path += [(tree_.value[node], tree_.n_node_samples[node])]
                paths += [path]

        recurse_tuple(0, path, paths); #print('paths\n', paths)
        
        # sort by samples count
        samples_count = [p[-1][1] for p in paths]
        ii = list(np.argsort(samples_count))
        paths = [paths[i] for i in reversed(ii)]

        rules = []
        for path in paths:
            antecedent = path[:-1]
            if class_names is None:
                # path is a tuple of the form ['(feature1 > 0.425)', '(feature2 > 0.875)', (array([[0.19438973],[0.28151123]]), 1)]
                # where first n-1 elements of the list describe a branch in the tree and the last element has the array of response
                # values for that branch of the tree; the length of that array must coincide with the number of the responses.
                #print('path', path, 'path[-1][0]', path[-1][0], resp_names)
                assert len(path[-1][0]) == len(resp_names)
                #print('+++++++++ path :\n', path, path[-1][0]); 
                responses_values = [path[-1][0][i][0] for i in range(len(resp_names))]
                #print('responses_values', responses_values)
                if rounding > 0:
                    responses_values = np.round(responses_values, rounding)
                #consequent = []
                consequent_dict = {}
                for i, rn in enumerate(resp_names):
                    #consequent.append((rn, '=', responses_values[i]))
                    consequent_dict[rn] = responses_values[i]
            else:
                # TODO: this branch has not been tested; likely will not work with multiple responses as classes
                # in the next line is defined as hard coded for resp_names[0] -- meaning, the last index [0].
                classes = path[-1][0][0]
                l = np.argmax(classes)
                # was hard-coded rounding: class_probability = np.round(100.0*classes[l]/np.sum(classes),2)
                class_probability = 100.0*classes[l]/np.sum(classes)
                if rounding > 0:
                    class_probability = np.round(class_probability, rounding)
                #consequent = []
                consequent_dict = {}
                for i, rn in enumerate(resp_names):
                    for cls in class_names:
                        #consequent.append((rn, cls, class_names[l], class_probability))
                        consequent_dict[rn] = (cls, class_names[l], class_probability)
                #rule += f"class: {class_names[l]} (proba: {class_probability}%)"
            #rule += f" | based on {path[-1][1]:,} samples"
            coverage = path[-1][1]
            #rule = {'antecedent': antecedent, 'consequent':consequent, 'coverage':coverage}
            rule = {'antecedent': antecedent, 'consequent':consequent_dict, 'coverage':coverage}
            rules.append(rule); 
        #print('rules\n', rules)

        return rules

    def _rule_to_str(self, rule):
        if isinstance(rule, dict):
            # convert consequent from dictionary into list of triplets that are more convenient to log tree formulas
            consequent_list = []
            for tup in rule['consequent'].items():
                #print('tup', tup)
                consequent_list.append((tup[0], '=', tup[1]))
            #print('consequent_list', consequent_list)
            
            if len(rule['antecedent']) == 0:
                return ' and '.join(['('+' '.join([str(e) for e in list(tup)])+')' for tup in consequent_list]) + \
                    f" | based on {rule['coverage']:,} samples"
            else:
                return 'if ' + ' and '.join(['('+' '.join([str(e) for e in list(tup)])+')' for tup in rule['antecedent']]) + \
                    ' then ' + ' and '.join(['('+' '.join([str(e) for e in list(tup)])+')' for tup in consequent_list]) + \
                    f" | based on {rule['coverage']:,} samples"
        elif isinstance(rule, str):
            return rule
        else:
            raise exception('Implementation error in function rule_to_str')
                
                
    # Print on standard output or in file rules that describe a set of decision or regression trees that
    # predict a single response, by extracting the rules from each individual tree using get_abstract_rules().
    # Argument tree_estimators is a set of objects tree_est that as *.tree_ contain trees trained for 
    # predicting the response. Argument class_names specifies levels of the response in case of classification. 
    # The representation of the input tree, tree_estimators, is that of sklearn DecisionTreeRegressor.
    # The returned value is a list of lists of rules, where each inner list correponds to a tree, and such an
    # inner list is a list of rules where each rule is represented as a dicteonary with three keys: 'antecedent', 
    # 'consequent' and 'coverage'. The 'antecedent feild reprethat describe the input tree as a set of rules path_cond --> leaf_value,
    # where path_cond is a condition along a full path in the tree and leaf_value is the value at the leaf of this 
    # path in the tree. This is how the returned value looks like when we have one tree:
    # [[{'antecedent': [('p3', '>', 0.4000000134110451), ('FMAX_abc', '<=', 0.75), ('FMAX_xyz', '>', 0.5000000149011612)], 
    #    'consequent': {'num1': 0.0, 'num2': 0.0}, 'coverage': 2}, 
    #   {'antecedent': [('p3', '>', 0.4000000134110451), ('FMAX_abc', '<=', 0.75), ('FMAX_xyz', '<=', 0.5000000149011612), 
    #     ('p3', '>', 0.7000000178813934)], 'consequent': {'num1': 1.0, 'num2': 0.0}, 'coverage': 2}, 
    #   {'antecedent': [('p3', '>', 0.4000000134110451), ('FMAX_abc', '>', 0.75)], 'consequent': {'num1': 1.0, 'num2': 1.0}, 
    #    'coverage': 1}, 
    #   {'antecedent': [('p3', '>', 0.4000000134110451), ('FMAX_abc', '<=', 0.75), ('FMAX_xyz', '<=', 0.5000000149011612), 
    #    ('p3', '<=', 0.7000000178813934)], 'consequent': {'num1': 0.0, 'num2': 0.0}, 'coverage': 1}, 
    #   {'antecedent': [('p3', '<=', 0.4000000134110451), ('FMAX_xyz', '>', 0.6666666716337204)], 
    #    'consequent': {'num1': 1.0, 'num2': 1.0}, 'coverage': 1}, 
    #   {'antecedent': [('p3', '<=', 0.4000000134110451), ('FMAX_xyz', '<=', 0.6666666716337204)], 
    #    'consequent': {'num1': 0.0, 'num2': 1.0}, 'coverage': 1}]]
    # The first rule (as an example) in that tree is: 
    #   {'antecedent': [('p3', '>', 0.4000000134110451), ('FMAX_abc', '<=', 0.75), ('FMAX_xyz', '>', 0.5000000149011612)], 
    #    'consequent': {'num1': 0.0, 'num2': 0.0}, 'coverage': 2};
    # The 'coverage' feild reports how many samples are covered by the path condition in training data; this info is not
    # used say for model configuration optimization.
    def trees_to_rules(self, tree_estimators, feature_names, response_names, class_names, log, rules_filename):
        if rules_filename is not None:
            save = True
            self._smlp_terms_logger.info('Writing tree rules into file ' + rules_filename)
            rules_file = open(rules_filename, "w")
        else:
            save = False
        #print('trees_to_rules: tree_estimators:', tree_estimators, 'response_names', response_names)
        # write a preamble: number of trees and tree semantics (how responses are computed using many trees)
        if log:
            print('#Forest semantics: {}\n'.format('majority vote'))
            print('#Number of trees: {}\n\n'.format(len(tree_estimators)))
        if save:
            rules_file.write('#Forest semantics: {}\n'.format('majority vote'))
            rules_file.write('#Number of trees: {}\n\n'.format(len(tree_estimators)))

        trees_as_rules = []
        # traverse trees, generate and print rules per tree (each rule correponds to a full branch in the tree)
        for indx, tree_est in enumerate(tree_estimators):
            #rules = self._get_rules(tree_est, feature_names, response_names, class_names)
            rules = self._get_abstract_rules(tree_est, feature_names, response_names, class_names);
            trees_as_rules.append(rules)
            
            if log:
                print('#TREE {}\n'.format(indx))
                for rule in rules:
                    print(self._rule_to_str(rule))
                    #self._rule_to_solver(None, rule)
                print('\n')
            if save:
                rules_file.write('#TREE {}\n'.format(indx))
                for rule in rules:
                    rules_file.write(self._rule_to_str(rule))
                    rules_file.write('\n')
        if save:
            rules_file.close()
        #print('trees_as_rules', trees_as_rules)
        return trees_as_rules

    # building terms (and formulas?) from terms left and right.
    # The argument op is a python operator from operator module, e.g.,
    # operator.gt which stands for the greater than operator >.
    def _apply_op(self, op, left, right):
        return self._inequality_ops_dict[op](left, right)

    # p is a triplet defininig a range, e.g. ('p3', '>', 0.40000001) or a triplet defining
    # an equality or assignment, e.g., ('num1', '==', 0.0).  So in these triplets p[0]
    # is a variable (or variable identifier) -- from which a term is built; e.g. smlp.Var('p3').
    # p[1] is the operator to be applied, and p[2] is a constant (int or real or rational (?).
    def _rule_triplet_to_term(self, p):
        #!!!!return self._apply_op(p[1], smlp.Var(p[0]), smlp.Cnst(p[2]))
        return (self.instSmlpTerms._ast_operators_smlp_map[self._inequality_ops_ast_dict[p[1]]])(
            self.instSmlpTerms.smlp_var(p[0]), self.instSmlpTerms.smlp_cnst(p[2]))
    

    # Convert rules predicting the same responses into SMLP terms. These rules are generated using 
    # method trees_to_rules() of this class directly from sklearn DecisionTreeRegressor model.
    # Currently only models with one tree only, such as in DecisionTreeRegressor, are supported.
    # Return a dictionary with reponse names as keys and corresponding smlp terms as the values.
    # This function is supposed to be applied to rules that correpond to all branches of a tree
    # (there is no check for this condition within the function itself). A rule has the following form:
    # {'antecedent': [('p3', '>', 0.4000000134110451), ('FMAX_abc', '<=', 0.75), ('FMAX_xyz', '<=', 0.5), 
    #     ('p3', '>', 0.7000000178813934)], 'consequent': {'num1': 1.0, 'num2': 0.0}, 'coverage': 2}.
    # rules is a list of rules. It is computed from a tree model using method trees_to_rules of the same
    # class TreeTerms.
    def compress_antecedent(self, antecedent):
        if not self._compress_rules:
            return antecedent, len(antecedent), len(antecedent)
        ant_dict = {}
        ant_reduced = []
        for trp in antecedent:
            #print('trp', trp, type(trp[0]), type(trp[1]), type(trp[2]))
            ant_dict[trp[0]] = {'lo':[], 'lo_cl':[], 'up':[], 'up_cl':[]}

        for trp in antecedent:
            if trp[1] == '<':
                ant_dict[trp[0]]['up'].append(trp[2])
                #ant_dict[trp[0]]['op_op'].append(trp[2])
            elif trp[1] == '<=':
                ant_dict[trp[0]]['up'].append(trp[2])
                ant_dict[trp[0]]['up_cl'].append(trp[2])
            elif trp[1] == '>':
                ant_dict[trp[0]]['lo'].append(trp[2])
                #ant_dict[trp[0]]['lo_op'].append(trp[2])
            elif trp[1] == '>=':
                ant_dict[trp[0]]['lo'].append(trp[2])
                ant_dict[trp[0]]['lo_cl'].append(trp[2])
            else:
                raise Exception('Unexpected binop ' + str(trp[1]) + ' in function reduce_antecedent')

        #print('ant_dict', ant_dict)
        for k,v in ant_dict.items():
            #print('k', k, 'v', v)
            if ant_dict[k]['up'] != []:
                mx = min(ant_dict[k]['up'])
                op = '<=' if mx in ant_dict[k]['up_cl'] else '<'
                ant_reduced.append([k,op, mx])
            if ant_dict[k]['lo'] != []:
                mn = max(ant_dict[k]['lo'])
                op = '>=' if mn in ant_dict[k]['lo_cl'] else '>'
                ant_reduced.append([k,op, mn])
        #print('antecedent', len(antecedent), antecedent, '\nant_reduced', len(ant_reduced), ant_reduced)
        #print('antecedent size: ', len(antecedent), ' --> ', len(ant_reduced), flush=True)
        return ant_reduced, len(antecedent), len(ant_reduced)

    def rules_to_term(self, algo, tree_number:int, rules:list, ant_reduction_stats:dict):
        #print('rules_to_term start', flush=True)
        # Convert the antecedent and consequent of a rule (corresponding to a full branch in a tree)
        # into smlp terms and return a dictionary with response names as the keys and pairs of terms
        # (antecdent_term, consequent_term) as the values. This function should be used for a rule
        # within a list of rules that represent a tree model.
        def rule_to_term(rule):
            antecedent = rule['antecedent']; #print('antecedent', antecedent)
            consequent = rule['consequent']; #print('consequent', consequent)
            antecedent, ant_befor, ant_after = self.compress_antecedent(antecedent)
            if len(antecedent) == 0:
                ant = self.instSmlpTerms.smlp_true
            else:
                ant = self._rule_triplet_to_term(antecedent[0])
            for i, p in enumerate(antecedent):
                if i > 0:
                    ant = ant & self._rule_triplet_to_term(p)
            res_dict = {}
            for resp, val in consequent.items():
                #term = smlp.Ite(ant, smlp.Cnst(val), smlp.Var('SMLP_UNDEFINED'))
                #res_dict[resp] = term
                #!!!!res_dict[resp] = (ant, smlp.Cnst(val))
                res_dict[resp] = (ant, self.instSmlpTerms.smlp_cnst(val))
                            
            return res_dict, ant_befor, ant_after
        
        def rule_to_form(rule, tree_number):
            res_dict, _, _ = rule_to_term(rule)
            rhs = self.instSmlpTerms.smlp_true
            for resp, (ant, val) in res_dict.items():
                resp_rhs = self.instSmlpTerms.smlp_eq(self.instSmlpTerms.smlp_var(self._tree_resp_id(tree_number, resp)), val) #   '_'.join([resp, 'tree', str(tree_number)]))
                rhs = resp_rhs if rhs == self.instSmlpTerms.smlp_true else self.instSmlpTerms.smlp_and(rhs, resp_rhs) 
            form = self.instSmlpTerms.smlp_implies(ant, rhs); #print('rule formula', rule, form)
            return form
        
        # returned value
        rules_dict = {}
        if self._tree_encoding == 'flat' and algo in ['dt_sklearn', 'rf_sklearn', 'et_sklearn', 'dt_caret', 'rf_caret', 'et_caret']:
            rules_dict[self._tree_model_id(algo, tree_number)] = [] # 'Tree_0_dt_sklearn_model
        #print('numer of rules', len(rules), flush=True)
        
        for i, rule in enumerate(rules):
            # this is an example of a rule (corresponds to a branch in a decision/regression tree):
            # {'antecedent': [('p3', '>', 0.4000000134110451), ('FMAX_abc', '<=', 0.75), ('FMAX_xyz', '>', 
            #     0.5000000149011612)], 'consequent': {'num1': 0.0, 'num2': 0.0}, 'coverage': 2}
            #print('\n ====== i', i, 'rule', rule)
            rule_dict, ant_befor, ant_after = rule_to_term(rule)
            ant_reduction_stats['before'].append(ant_befor)
            ant_reduction_stats['after'].append(ant_after)
            # here is how the corresponding rule_dict looks like (it contains smlp / smt2 terms):
            # {'num1': (<smlp.libsmlp.form2 (and (and (> p3 (/ 53687093 134217728)) (<= FMAX_abc (/ 3 4))) 
            #     (> FMAX_xyz (/ 33554433 67108864)))>, <smlp.libsmlp.term2 0>), 
            #  'num2': (<smlp.libsmlp.form2 (and (and (> p3 (/ 53687093 134217728)) (<= FMAX_abc (/ 3 4))) 
            #     (> FMAX_xyz (/ 33554433 67108864)))>, <smlp.libsmlp.term2 0>)}
            #print('rule_dict', rule_dict)
            #print('number of rule terms', len(rule_dict), flush=True)
            if self._tree_encoding == 'flat' and algo in ['dt_sklearn', 'rf_sklearn', 'et_sklearn', 'dt_caret', 'rf_caret', 'et_caret']:
                rules_dict[self._tree_model_id(algo, tree_number)].append(rule_to_form(rule, tree_number)) # 'Tree_0_dt_sklearn_model
            else:
                for resp, (ant_term, con_term) in rule_dict.items():            
                    if i == 0:
                        # The condition along a branch of a decision tree is implied by disjunction 
                        # of conditions along the rest of the branches, thus can be omitted. In this
                        # implementation, we choose do omit the condition along the first branch.
                        # TODO: Ideally, we could implement a sanity check that the condition along 
                        # the forst branch is implied by the conjunction of conditions along the rest
                        # of the branches -- using a solver like Z3.
                        rules_dict[resp] = con_term #self.instSmlpTerms.smlp_ite(ant_term, con_term, smlp.Var('SMLP_UNDEFINED'))
                    else:
                        rules_dict[resp] = self.instSmlpTerms.smlp_ite(ant_term, con_term, rules_dict[resp])
        #print('rules_dict', rules_dict)

        #print('rules_to_term end', flush=True)
        return rules_dict, ant_reduction_stats
    
    def tree_model_to_term(self, tree_model, algo, feat_names, resp_names):
        if algo in ['dt_caret', 'dt_sklearn']:
            tree_estimators = [tree_model]
        elif algo in ['rf_caret', 'rf_sklearn', 'et_caret', 'et_sklearn']:
            tree_estimators = tree_model.estimators_
        else:
            raise Exception('Model trained using algorithm ' + str(algo) + ' is currently not supported in smlp_opt')
        trees = self.trees_to_rules(tree_estimators, feat_names, resp_names, None, False, None)
        def count_occurrences(int_list):
            counts = {}
            for element in int_list:
                if element in counts:
                    counts[element] += 1
                else:
                    counts[element] = 1
            return counts
        #print('------- trees ---------\n', trees); 
        #print('tree_term_dict_dict start', flush=True)
        tree_term_dict_dict = {} 
        ant_reduction_stats = {'before':[], 'after':[]}
        branches_count_per_tree = []
        for i, tree_rules in enumerate(trees):
            #print('====== tree_rules ======\n', len(tree_rules), tree_rules)
            branches_count_per_tree.append(len(tree_rules))
            tree_term_dict, ant_reduction_stats = self.rules_to_term(algo, i, tree_rules, ant_reduction_stats); #print('tree term_dict', tree_term_dict); 
            if self._tree_encoding == 'flat' and algo in ['dt_sklearn', 'rf_sklearn', 'et_sklearn', 'dt_caret', 'rf_caret', 'et_caret']:
                assert list(tree_term_dict.keys()) == [self._tree_model_id(algo, i)]
            else:
                #print(list(tree_term_dict.keys()), resp_names)
                assert list(tree_term_dict.keys()) == resp_names
            tree_term_dict_dict['tree_'+str(i)] = tree_term_dict

        if self._compress_rules:
            trees_count = len(trees)
            branches_count = sum(branches_count_per_tree); #print('branches_count', branches_count)
            ant_len_befor = sum(ant_reduction_stats['before']); #print('ant_len_befor', ant_len_befor)
            ant_len_after = sum(ant_reduction_stats['after'])
            unique_conj_befor = count_occurrences(ant_reduction_stats['before'])  #sum(list(set(ant_reduction_stats['before'])))
            unique_conj_after = count_occurrences(ant_reduction_stats['after']) # sum(list(set(ant_reduction_stats['after'])))
            tree_max_depth_befor = max(ant_reduction_stats['before'])
            tree_max_depth_after = max(ant_reduction_stats['after'])
            assert branches_count >= trees_count
            #assert ant_len_befor >= branches_count
            assert ant_len_befor >= ant_len_after
            #assert ant_len_befor >= unique_conj_befor
            #assert ant_len_after >= unique_conj_after
            #assert unique_conj_befor >= unique_conj_after
            assert tree_max_depth_befor >= tree_max_depth_after
            self._smlp_terms_logger.info(
                'Tree rules (branches) antecedent compression statistics for response(s) {}:'.format(','.join(resp_names)) + \
                '\n\ttrees count in the model   ' + str(trees_count) + \
                '\n\ttree branches/rules count  ' + str(branches_count) + \
                '\n\tantecedent lengths before  ' + str(ant_len_befor) + \
                '\n\tantecedent lengths after   ' + str(ant_len_after) + \
                '\n\tbranch length counts before ' + str(unique_conj_befor) + \
                '\n\tbranch length counts after  ' + str(unique_conj_after) + \
                '\n\ttree max depth before      ' + str(tree_max_depth_befor) + \
                '\n\ttree max depth after       ' + str(tree_max_depth_after))

        #print(tree_term_dict_dict\n', tree_term_dict_dict)
        #print('tree_term_dict_dict end', flush=True)
        number_of_trees = len(trees); #print('number_of_trees (trees)', number_of_trees)
        tree_model_term_dict = {}
        #print('tree_model_term_dict start', flush=True)
        for j, tree_rules in enumerate(trees):
            #print('j', j, flush=True)
            if self._tree_encoding == 'flat' and algo in ['dt_sklearn', 'rf_sklearn', 'et_sklearn', 'dt_caret', 'rf_caret', 'et_caret']:
                curr_resp = resp_names[0] if len(resp_names) == 1 else None # TODO !!! might require model_per_response to decide instead of using len(resp_names) == 1 
                tree_model_term_dict[self._tree_model_id(algo, j, curr_resp)] = tree_term_dict_dict['tree_'+str(j)][self._tree_model_id(algo, j)]
                #print('tree_model_term_dict', tree_model_term_dict)
                if j == number_of_trees - 1:
                    model_formulas = []
                    for formulas in tree_model_term_dict.values():
                        model_formulas = model_formulas + formulas
                    tree_model_term_dict = {}
                    tree_model_term_dict[self._tree_model_id(algo, None, curr_resp)] = model_formulas
                    #print('tree_model_term_dict w/o resp', tree_model_term_dict);                     
                    for resp_name in resp_names:
                        sum_term = self.instSmlpTerms.smlp_add_multi([ self.instSmlpTerms.smlp_var(self._tree_resp_id(i, resp_name)) for i in range(number_of_trees)])
                        mean_term = self.instSmlpTerms.smlp_mult(self.instSmlpTerms.smlp_cnst(self.instSmlpTerms.smlp_q(1) / self.instSmlpTerms.smlp_q(int(number_of_trees))), sum_term)
                        resp_j_form = self.instSmlpTerms.smlp_eq(self.instSmlpTerms.smlp_var(resp_name), mean_term)
                        #print('sum_term', sum_term); print('mean_term', mean_term)
                        #print('resp_j_form', resp_name, resp_j_form)
                        tree_model_term_dict[self._tree_model_id(algo, None, curr_resp)].append(resp_j_form)
            else: 
                for resp_name in resp_names:
                    if j == 0:
                        tree_model_term_dict[resp_name] = tree_term_dict_dict['tree_'+str(j)][resp_name]
                    else:                            
                        tree_model_term_dict[resp_name] = self.instSmlpTerms.smlp_add(tree_model_term_dict[resp_name], tree_term_dict_dict['tree_'+str(j)][resp_name])
                        if j == number_of_trees - 1: # the last tree -- compute the mean by dividing the sum on number_of_trees
                            #tree_model_term_dict[resp_name] = smlp.Div(tree_model_term_dict[resp_name], smlp.Cnst(int(number_of_trees)))
                            #!!!!tree_model_term_dict[resp_name] = self.instSmlpTerms.smlp_mult(smlp.Cnst(smlp.Q(1) / smlp.Q(int(number_of_trees))), tree_model_term_dict[resp_name])
                            tree_model_term_dict[resp_name] = self.instSmlpTerms.smlp_mult(
                                self.instSmlpTerms.smlp_cnst(self.instSmlpTerms.smlp_q(1) / self.instSmlpTerms.smlp_q(int(number_of_trees))), tree_model_term_dict[resp_name])

        #print('tree_model_term_dict', tree_model_term_dict); print('tree_model_term_dict end', flush=True)
        return tree_model_term_dict

    def tree_models_to_term(self, model, algo, feat_names, resp_names):
        #print('tree_models_to_term start', flush=True)
        #print('tree_models_to_term: feat_names', feat_names, 'resp_names', resp_names)
        #print('tree_models_to_term: ', model, '\ntype',  type(model))
        if isinstance(model, dict):
            # case when model is per response
            tree_model_term_dict = {}
            for resp_name in resp_names:
                tree_model = model[resp_name]
                tree_model_term_dict[resp_name] = self.tree_model_to_term(tree_model, algo, 
                    feat_names, [resp_name])[resp_name]
        else:
            # there is one model covering all responses (one or multiple responses)
            tree_model_term_dict = self.tree_model_to_term(model, algo, feat_names, resp_names)
        #print('tree_models_to_term end', flush=True)
        #print('tree_models_to_term: ', tree_model_term_dict)
        return tree_model_term_dict

    def get_tree_model_estimator_count2(self, algo, model):
        if algo in ['dt_caret', 'dt_sklearn']:
            n_trees = [1]
        elif algo in ['rf_caret', 'rf_sklearn', 'et_caret', 'et_sklearn']:
            if isinstance(model, dict):
                #m = next(iter(model.values()))
                n_trees = [len(m.estimators_) for m in model.values()]             
            else:
                #m = model
                n_trees = [len(model.estimators_)]
            #n_trees = len(m.estimators_)
        else:
            raise Exception('Unexpected tree model ' + str(algo))
        return n_trees
    
    def get_tree_model_estimator_count(self, algo, model):
        def get_estimators(algo, model):
            if algo in ['rf_caret', 'rf_sklearn', 'et_caret', 'et_sklearn']:
                est = model.estimators_
            elif algo in ['dt_caret', 'dt_sklearn']:
                est = [model]
            else:
                raise Exception('Unexpected tree model ' + str(algo))
            return est
        
        if isinstance(model, dict):
            #m = next(iter(model.values()))
            n_trees = [len(get_estimators(algo, m)) for m in model.values()]             
        else:
            #m = model
            n_trees = [len(get_estimators(algo, model))]

        return n_trees
    
    
# Method to generate smlp term from sklearn (and caret) representation of a polynomial model    
class PolyTerms: #(SmlpTerms):
    def __init__(self):
        self._smlp_terms_logger = None
    
    # set logger from a caller script
    def set_logger(self, logger):
        self._smlp_terms_logger = logger
    
    # Create smlp term from polynomial model, returns dictionary with response names as keys
    # and the correponding smlp terms as values, works for single response only
    def poly_model_to_term_single_response(self, feat_names, resp_names, coefs, powers, resp_id, log, formula_filename):
        #print('Polynomial model coef\n', coefs.shape, '\n', coefs)
        #print('Polynomial model terms\n', powers.shape, '\n', powers)
        #print('Polynomial model inps\n', len(feat_names), feat_names)
        #print('Polynomial model outp\n', len(resp_names), resp_names)
        if len(feat_names) != powers.shape[1]:
            raise Exception('Error in poly_model_to_term_single_response')
        term_str = ''
        for r in range(powers.shape[0]):
            #print('r', powers[r], 'coef', coefs[0][r])
            if coefs[resp_id][r] == 0:
                continue
            curr_term_str = str(coefs[resp_id][r])
            curr_term = smlp.Cnst(coefs[resp_id][r])
            for i in range(len(feat_names)):
                #print('i', feat_names[i], coefs[resp_id][r], powers[r][i])
                if powers[r][i] == 0:
                    continue
                elif powers[r][i] == 1:
                    curr_term_str = curr_term_str + ' * ' + feat_names[i]
                    curr_term = curr_term * smlp.Var(feat_names[i])
                else:
                    curr_term_str = curr_term_str + ' * ' + feat_names[i] + '^' + str(powers[r][i])
                    #print('power coeff', powers[r][i]); 
                    # power/exponentiation operation ** is not supported in smlp / libsmlp, we therefore perform 
                    # repeated multiplication in required number of times
                    curr_term = curr_term * smlp.Var(feat_names[i]) #** smlp.Cnst(int(powers[r][i])))
                    for p in range(2, powers[r][i]+1):
                        curr_term = curr_term * smlp.Var(feat_names[i])
                    #print('curr_term', curr_term)

            #print('curr_term_str', curr_term_str); print('curr_term', curr_term)

            if term_str == '':
                term_str = curr_term_str
                term = curr_term
            else:
                term_str = term_str + ' + ' + curr_term_str
                term = term + curr_term

        # add the response name
        formula_str = resp_names[resp_id] + ' == ' + term_str

        if log:
            print('formula', formula_str)

        # save formula into file
        if formula_filename is not None:
            model_file = open(formula_filename, "w")
            model_file.write(formula_str)
            model_file.close()

        return {resp_names[resp_id]:term}

    # Create smlp term from polynomial model, returns dictionary with model response names as keys
    # and the correponding smlp terms as values. Arguments model_feat_names and model_resp_names are
    # feature and response names, respectively possibly suffixed by '_scaled' in case features and/or 
    # responses have been scaled prior to training
    def poly_model_to_term(self, model_feat_names, model_resp_names, coefs, powers, log, formula_filename):
        poly_model_terms_dict = {}
        for resp_id, resp_name in enumerate(model_resp_names):
            poly_model_terms_dict[resp_name] = self.poly_model_to_term_single_response(
                model_feat_names, model_resp_names, coefs, powers, 
                resp_id, log, formula_filename)[resp_name]; #print('poly_model_terms_dict', poly_model_terms_dict)
        return poly_model_terms_dict
            
    
    
# Method to generate smlp term from a Tensorflow Keras model built using Sequential or Functional API 
class NNKerasTerms: #(SmlpTerms):
    def __init__(self):
        self._smlp_terms_logger = None
     
    # set logger from a caller script
    def set_logger(self, logger):
        self._smlp_terms_logger = logger
    
    # Creates smlp term for an internal node in NN based on terms built for the preceding layer 
    # (the argument called last_layer_terms); as well as the weights and bias for that node with
    # respect to the preceding layer last_layer_terms.
    def _nn_dense_layer_node_term(self, last_layer_terms, node_weights, node_bias):
        #print('node_weights', node_weights.shape, type(node_weights), '\n', node_weights)
        #print('node_bias', node_bias.shape, type(node_bias), '\n', node_bias);
        layer_term = None
        for i,t in enumerate(last_layer_terms):
            if i == 0:
                layer_term = last_layer_terms[0] * smlp.Cnst(float(node_weights[0]))
            else:
                layer_term = layer_term + last_layer_terms[i] * smlp.Cnst(float(node_weights[i]))
        layer_term = layer_term + smlp.Cnst(float(node_bias)) 

        return layer_term

    # Creates smlp term that represents an application of an activation function relu or linear
    # on smlp term representing an NN node
    def _nn_activation_term(self, activation_func, input_term):
        if activation_func == 'relu':
            relu_term = smlp.Ite(input_term >= smlp.Cnst(0), input_term, smlp.Cnst(0))
            return relu_term
        elif activation_func == 'linear':
            return input_term
        else:
            raise Exception('Unsupported activation function ' + str(activation_func))

    # Creates a list of smlp terms curr_layer_terms for an internal layer of a sequential NN.
    # Argument last_layer_terms is a list of smlp terms correponding to a preceding layer
    # (the input layer or an internal layer preceding to the current layer). The arguments
    # layer_weights and layer_biases np arrays of weights and biases to compute current layer
    # nodes from the bodes of last (preceding) layer; and argument activation_func is the
    # activation function for the current layer. This function is called both for sequential
    # and functional API models from function nn_keras_model_to_formula(), which explicitly 
    # genrates last_layer_terms for the input layer (and subsequent layers are generated using
    # _nn_dense_layer_terms, both for sequential and functional API models).
    def _nn_dense_layer_terms(self, last_layer_terms, layer_weights, layer_biases, activation_func):
        #print('-------start computing next layer')
        #print('layer_weights', layer_weights.shape, '\n', layer_weights)
        #print('layer_biases', layer_biases.shape, '\n', layer_biases)
        #print('last_layer_terms', len(last_layer_terms)) #, last_layer_terms)
        assert layer_weights.shape[1] == len(last_layer_terms)
        assert layer_biases.shape[0] == layer_weights.shape[0]
        curr_layer_terms = [self._nn_activation_term(activation_func, self._nn_dense_layer_node_term(
            last_layer_terms, layer_weights[i], layer_biases[i])) for i in range(layer_weights.shape[0])]
        #print('curr_layer_terms', len(curr_layer_terms))
        #print('+++++++done computing next layer')
        #print('curr_layer_terms', curr_layer_terms)
        assert layer_biases.shape[0] == len(curr_layer_terms)
        return curr_layer_terms

    def _keras_is_sequential(self, model):
        ic("Changes here ...")
        try:
            # v2.9 has this API
            cl = keras.engine.sequential.Sequential
        except AttributeError:
            # v2.14+ has this API
            #cl = keras.src.engine.sequential.Sequential
            cl = Sequential
        return isinstance(model, cl)

    def _keras_is_functional(self, model):
        try:
            # v2.9 has this API
            cl = keras.engine.functional.Functional
        except AttributeError:
            # v2.14+ has this API
            cl = keras.src.engine.functional.Functional
        return isinstance(model, cl)
    
    # determine the model type -- sequential vs functional
    def _get_nn_keras_model_type(self, model):
        #print('keras model', model, type(model))
        if self._keras_is_sequential(model):
            model_type = 'sequential'
        elif self._keras_is_functional(model):
            model_type = 'functional'
        else:
            raise Exception('Unsupported Keras NN type (neither sequential nor functional)')
            assert False
        return model_type
    
    # Create SMLP terms from NN Keras model. Returns a dictionary with response names from model_resp_names as keys
    # and respective model terms as the values.
    def nn_keras_model_to_term(self, model, model_feat_names, model_resp_names, feat_names, resp_names):
        #from pprint import pprint
        #import inspect
        #print('model', model, type(model), model.summary())
        model_type = self._get_nn_keras_model_type(model)
        assert model_type in ['sequential', 'functional']
        model_terms_dict = {}
        # input variables layer as list of terms
        last_layer_terms = [smlp.Var(v) for v in model_feat_names]; #print('input layer terms', last_layer_terms)
        for layer in model.layers:
            #print('layer type', type(layer).__name__ )
            #print('layer config', layer.get_config())
            if type(layer).__name__ == 'InputLayer':
                assert model_type == 'functional'
                continue 
            assert isinstance(layer, keras.layers.Dense)
            #print('current layer', layer) 
            #pprint(inspect.getmembers(layer)); 
            #print('units', layer.units, 'activation', layer.activation, 'use_bais', layer.use_bias, 'kernel_initializer', layer.kernel_initializer, 'bias_initializer', layer.bias_initializer, 'kernel_regularizer', layer.kernel_regularizer, 'bias_regularizer', layer.bias_regularizer, 'activity_regularizer', layer.activity_regularizer, 'kernel_constraint', layer.kernel_constraint, 'bias_constraint', layer.bias_constraint)
            layer_activation = layer.get_config()["activation"]; #print('layer_activation', layer_activation)
            weights, biases = layer.get_weights(); 
            #print('t_weights', weights.transpose().shape, '\n', weights.transpose()); 
            #print('t_biases', biases.transpose().shape, '\n', biases.transpose())
            curr_layer_terms = self._nn_dense_layer_terms(last_layer_terms, weights.transpose(), 
                biases.transpose(), layer_activation)
            if model_type == 'functional' and layer.get_config()['name'] in resp_names:
                #assert model_type == 'functional'
                #print('layer.get_config()[name]', layer.get_config()['name'])
                resp_index = resp_names.index(layer.get_config()['name'])
                #print('response', layer.get_config()['name'])
                # we have an output layer -- do not update last_layer_terms
                #model_terms_dict[layer.get_config()['name']] = curr_layer_terms
                model_terms_dict[model_resp_names[resp_index]] = curr_layer_terms[0]
            else:
                last_layer_terms = curr_layer_terms

        if model_type == 'functional':
            return model_terms_dict
        else:
            return dict(zip(model_resp_names, last_layer_terms))


# NOTE on terminology: The two most discussed scaling methods are Normalization and Standardization. 
# Normalization typically means rescaling the values into a range of [0,1]. Standardization typically 
# means rescaling data to have a mean of 0 and a standard deviation of 1 (unit variance).
class ScalerTerms(SmlpTerms):
    def __init__(self):
        pass
        #self._SCALED_TERMS_SUFFIX = '_scaled'
    
    @property
    def scaled_term_suffix(self):
        return '_scaled'
    
    # Convention to generate scaled feature, response objective's names from respective original names.
    # Models are generated using the original feature and response names, independently from whether 
    # the features and/or responses were scaled prior to traing the models. However, then building model
    # terms from models, we use scaled feature and/or response names as inputs and outputs of the model
    # term, respectively, in case features and/or responses were scaled before trainng the model. In case
    # optimization objectives are scaled prior to optimization procedure, the results will be un-scaled 
    # before reproting to user so that user can see results in riginal scale (and in scaled form as well).
    def _scaled_name(self, name):
        return name+self.scaled_term_suffix #self._SCALED_TERMS_SUFFIX
    
    def _unscaled_name(self, name):
        if name.endswith(self.scaled_term_suffix): # self._SCALED_TERMS_SUFFIX
            return name[:-len(self.scaled_term_suffix)] # self._SCALED_TERMS_SUFFIX
        else:
            return name
    
    # Computes term x_scaled for column x using expression x_scaled = 1 / (max(x) - min(x)) * (x - min(x).
    # Argument orig_feat_name is name for column x, argument scaled_feat_name is the name of scaled column 
    # x_scaled obtained from x using min_max scaler to range [0, 1] (which is the same as normalizin x),
    # orig_min stands for min(x) and orig_max stands for max(x). Note that 1 / (max(x) - min(x)) is a
    # rational constant, it is defined to smlp instance as a fraction (thus there is no loss of precision).
    def feature_scaler_to_term(self, orig_feat_name, scaled_feat_name, orig_min, orig_max): 
        #print('feature_scaler_to_term', 'orig_min', orig_min, type(orig_min), 'orig_max', orig_max, type(orig_max), flush=True)
        if orig_min == orig_max:
            return self.smlp_cnst(0) #smlp.Cnst(0) # same as returning smlp.Cnst(smlp.Q(0))
        else:
            return self.smlp_mult(
                self.smlp_cnst(self.smlp_q(1) / self.smlp_q(orig_max - orig_min)),
                (self.smlp_var(orig_feat_name) - self.smlp_cnst(orig_min)))
            ####return self.smlp_div(self.smlp_var(orig_feat_name) - self.smlp_cnst(orig_min), self.smlp_cnst(orig_max) - self.smlp_cnst(orig_min))
            ####return smlp.Cnst(smlp.Q(1) / smlp.Q(orig_max - orig_min)) * (smlp.Var(orig_feat_name) - smlp.Cnst(orig_min))
    
    # Computes dictionary with features as keys and scaler terms as values
    def feature_scaler_terms(self, data_bounds, feat_names): 
        return dict([(self._scaled_name(feat), self.feature_scaler_to_term(feat, self._scaled_name(feat), 
            data_bounds[feat]['min'], data_bounds[feat]['max'])) for feat in feat_names])
        
    # Computes term x from column x_scaled using expression x = x_scaled * (max_x - min_x) + x_min.
    # Argument orig_feat_name is name for column x, argument scaled_feat_name is the name of scaled column 
    # x_scaled obtained earlier from x using min_max scaler to range [0, 1] (same as normalization of x),
    # orig_min stands for min_x and orig_max stands for max_x, where min_x amd max_x there computed and 
    # stored diring scaling of x to x_scaled, and loaded for cmputed original, unscaled x.
    def feature_unscaler_to_term(self, orig_feat_name, scaled_feat_name, orig_min, orig_max): 
        ####unscaled_term = (smlp.Var(scaled_feat_name) * smlp.Cnst(orig_max - orig_min)) + smlp.Cnst(orig_min)
        unscaled_term = self.smlp_add(
            self.smlp_mult(self.smlp_var(scaled_feat_name), self.smlp_cnst(orig_max - orig_min)), 
            self.smlp_cnst(orig_min))
        #print('unscaled_term', unscaled_term)
        return unscaled_term
    
    # Compute dictionary with features as keys and unscaler terms as values.
    # This function actually is applied to responses (is called on resp_names as feat_names).
    def feature_unscaler_terms(self, data_bounds, feat_names): 
        return dict([(feat, self.feature_unscaler_to_term(feat, self._scaled_name(feat), 
            data_bounds[feat]['min'], data_bounds[feat]['max'])) for feat in feat_names])

    # unscale constant const converted to smlp.term2 with respect to max and min values of 
    # feat_name specified in data_bounds dictionary
    def unscale_constant_term(self, data_bounds:dict, feat_name:str, const):
        orig_max = data_bounds[feat_name]['max']
        orig_min = data_bounds[feat_name]['min']
        return self.smlp_add(
            self.smlp_mult(self.smlp_cnst(const), self.smlp_cnst(orig_max - orig_min)), 
            self.smlp_cnst(orig_min))
        #return (smlp.Cnst(const) * smlp.Cnst(orig_max - orig_min)) + smlp.Cnst(orig_min)


class ModelTerms(ScalerTerms):
    def __init__(self):
        self._scalerTermsInst = ScalerTerms()
        self._treeTermsInst = TreeTerms()
        self._polyTermsInst = PolyTerms()
        #self._smlpTermsInst = SmlpTerms
        self._nnKerasTermsInst = NNKerasTerms()
        
        #self._cache_terms = False
        
        self.report_file_prefix = None
        self.model_file_prefix = None
        self._smlp_terms_logger = None

        # control parameter to decide whether to add input and knob variable ranges as part
        # of solver domain declaration or to only add variable type (int, real) declaration.
        # This option is for experiemntation, as this choice can affect solver performance.
        self._declare_domain_interface_only = True
        
        # in order to use QF_LRA instead of QF_LIRA theory solver, we want to incode integer
        # variables as reals and add an integer grid constraint to make it range on integers.
        self._declare_integer_as_real_with_grid = False
        
        # for integer inputs, instead of declaring their range, disclare all their values as disjunction
        # say if 1 <= x <= 5, we generate constrint x == 1 | x == 2 | ... | x == 5 (and do not generate
        # constraints 1 <= x and x <= 5. Note this applies to (free) inputs, not knobs (control inputs).
        self._encode_input_range_as_disjunction = False
        
        self._SPEC_DOMAIN_RANGE_TAG = 'range'
        self._SPEC_DOMAIN_INTERVAL_TAG = 'interval'
        self._DEF_COMPRESS_RULES = True
        self._DEF_SIMPLIFY_TERMS = False
        self._DEF_TREE_ENCODING = 'nested' # 'flat' #  
        #self._DEF_CACHE_TERMS = False
        self.model_term_params_dict = {
            'compress_rules': {'abbr':'compress_rules', 'default':str(self._DEF_COMPRESS_RULES), 'type':str_to_bool,
                'help':'Should rules that represent tree branches be compressed to eliminate redundant repeated splitting ' +
                'of ranges of model features after training tree based models, in order to build smaller model terms? ' +
                '[default {}]'.format(str(self._DEF_COMPRESS_RULES))},
            'simplify_terms': {'abbr':'simplify_terms', 'default':str(self._DEF_SIMPLIFY_TERMS), 'type':str_to_bool,
                'help':'Should terms be simplified using before building solver instance in model exploration modes? ' +
                '[default {}]'.format(str(self._DEF_SIMPLIFY_TERMS))},
            'tree_encoding': {'abbr':'tree_encoding', 'default':str(self._DEF_TREE_ENCODING), 'type':str,
                'help':'Strategy to encode tree model to solvers. Flat encoding cretea a formula from ' +
                'each branch of a tree, while nested encoding builds formula from branches using nested ' +
                'if-thn-else (ite) exoressions [default {}]'.format(str(self._DEF_COMPRESS_RULES))},
            #'cache_terms': {'abbr':'cache_terms', 'default':str(self._DEF_CACHE_TERMS), 'type':str_to_bool,
            #    'help':'Should terms be cached along building terms and formulas in model exploration modes? ' +
            #    '[default {}]'.format(str(self._DEF_CACHE_TERMS))}
        }
        
    # set logger from a caller script
    def set_logger(self, logger):
        self._smlp_terms_logger = logger
        self._treeTermsInst.set_logger(logger)
        self._polyTermsInst.set_logger(logger)
        self._nnKerasTermsInst.set_logger(logger)
    
    # set tracer from a caller script
    # logs solver statistics in a trace log file
    def set_tracer(self, tracer, trace_runtime, trace_prec, trace_anonym):
        self._smlp_terms_tracer = tracer
        self._trace_runtime = trace_runtime
        self._trace_precision = trace_prec
        self._trace_anonymize = trace_anonym
    
    # model_file_prefix is a string used as prefix in all outut files of SMLP that are used to 
    # save a trained ML model and to re-run the model on new data (without need for re-training)
    def set_model_file_prefix(self, model_file_prefix):
        self.model_file_prefix = model_file_prefix

    # report_file_prefix is a string used as prefix in all report files of SMLP
    def set_report_file_prefix(self, report_file_prefix):
        self.report_file_prefix = report_file_prefix
    
    def set_spec_file(self, spec_file):
        self._specInst.set_spec_file(spec_file)
    
    def set_smlp_spec_inst(self, spec_inst):
        self._specInst = spec_inst
    
    def set_compress_rules(self, compress_rules:bool):
        self._compress_rules = compress_rules
        self._treeTermsInst.set_compress_rules(compress_rules)
    
    def set_simplify_terms(self, simplify_terms:bool):
        self._simplify_terms = simplify_terms
    
    def set_tree_encoding(self, tree_encoding:str):
        self._tree_encoding = tree_encoding
        self._treeTermsInst.set_tree_encoding(tree_encoding)
    
    #def set_cache_terms(self, cache_terms:bool):
    #    self._cache_terms = cache_terms
    
    # file to dump tree model converted to SMLP term
    def smlp_model_term_file(self, resp:str, full:bool):
        assert self.model_file_prefix is not None
        filename_suffix = 'smlp_full_model_term.json' if full else 'smlp_model_term.json'
        if resp is None:
            return '_'.join([self.model_file_prefix, filename_suffix])
        else:
            return '_'.join([self.model_file_prefix, str(resp), filename_suffix])
    '''
    # file to dump tree model converted to SMLP term
    @property
    def smlp_full_model_term_file(self, resp:str=None):
        assert self.model_file_prefix is not None
        if resp is None:
            return self.model_file_prefix + '_smlp_full_model_term.json'
        else:
            return 
    '''
    # This function computes a dictionary with response names as keys and an smlp term corresponding to the 
    # model for that response as the value for that key. The response names used as keys are suffixed with 
    # suffix '_scaled' in case reponses were scaled prior to the training, and feature names that serve as 
    # variables (leaves) in the model terms are also suffixed with '_scaled' in case features were scaled 
    # prior to the model training; the reason for that is further elaboarted below. This function ignores
    # (that is, does not model) the scaling and/or unscaling constraints even if features and/or responses 
    # were scaled prior to training the model(s). These scailing / unscaling constraints for features and 
    # responses are handled in function _compute_model_terms_dict() which first computs only the "pure"   
    # model terms using this function (hence the "pure" in the name of this function) and then adds feature  
    # and response scaling/un-scaling constraints when it is relevant.
    #
    # The argument 'model' is an ML model, and can be a model trained for one response (out of one or more 
    # responses) or for all the responses in training data. (Parameter model_per_response controls whether 
    # one model is trained for all responses or individual models are trained for each response.) 
    # As said above, this function will return a dictionary of model terms with response names as keys and 
    # the respective model terms as the values, does not matter whether the model was trained for one response 
    # or multiple ones.  Also in the latter case, when building respective model terms per resposne, the
    # common sub-terms of the resulting model terms for different responses are shared thanks to smlp term 
    # and formula construction and storage mechanisms, and these common sub-terms are represented / stored
    # as one smlp (sub-)term. So generating an smlp term per response will not make the problem instance 
    # larger than it should be.
    #  
    # As mentioned above, while a model is trained with original feature and response names (even if 
    # features and/or responses were scaled prior to training), the model terms are built with feature names 
    # suffixed with '_scaled' in case features were scaled, and with response names suffixed with '_scaled' 
    # in case responses were scaled, prior to model training; these feature and response names are precomputed 
    # and passed to this function using arguments model_feat_names and model_resp_names. Thus if say features 
    # were not scaled prior to tarining, model_feat_names is identical to feat_names, and similarly for 
    # model_resp_names and resp_names. Function _compute_model_terms_dict(), when taking care of modelling 
    # scaling constraints, substutes the variables with suffix '_scaled' with the corresponding terms capturing \
    # the scaling constraints, so at the end the model terms computed by _compute_model_terms_dict() use only
    # the original feature and response names feat_names and resp_names as the variables, and scaling  
    # constraints are modeled as part of the final computed model term.
    # 
    # Only a subset of all model algorithms supported in training and prediction modes are currently
    # supported in _compute_pure_model_terms(). For each supported model algorithm, a dedicated function
    # computing a dictionary of model terms from the model / dictionary of mdoels is called. 
    def _compute_pure_model_terms(self, algo, model, model_feat_names, model_resp_names, feat_names, resp_names):
        assert not isinstance(model, dict)
        if algo == 'nn_keras': 
            model_term_dict = self._nnKerasTermsInst.nn_keras_model_to_term(model, model_feat_names, 
                model_resp_names, feat_names, resp_names) 
        elif algo == 'poly_sklearn':
            model_term_dict = self._polyTermsInst.poly_model_to_term(model_feat_names, model_resp_names, 
                model[0].coef_, model[1].powers_, False, None)
        elif algo in ['dt_sklearn', 'dt_caret', 'rf_sklearn', 'rf_caret', 'et_sklearn', 'et_caret']:
            model_term_dict = self._treeTermsInst.tree_models_to_term(model, algo, model_feat_names, model_resp_names)
        #elif algo in ['rf_sklearn', 'rf_caret', 'et_sklearn', 'et_caret']:
        #    model_term_dict = self._treeTermsInst.tree_models_to_term(model, algo, model_feat_names, model_resp_names)
        else:
            raise Exception('Algo ' + str(algo) + ' is currently not suported in model exploration modes')
        #print('model_term_dict', model_term_dict)
        resp_name = resp_names[0] if len(resp_names) == 1 else None
        with open(self.smlp_model_term_file(resp_name, False), 'w') as f:
            json.dump(str(model_term_dict), f, indent='\t', cls=np_JSONEncoder)
        return model_term_dict
    
    
    # This function takes an ML model (the argument 'model) as well as features and responses scaling info
    # as inputs and for each response in the model generates a term that encodes the "pure" model constraints
    # as well as constraints relating to scaling of features and/or responses if they were scaled prior to
    # the model training. By "pure" model we mean a model trained with the original feature and response names
    # independently from whether the features and/or responses were scaled prior to the training. Term for a
    # pure model is built using function _compute_pure_model_terms() (see its description for more details on
    # how this is done), and the remaining parts in this function take care of generating terms for feature 
    # and/or response scaler expressions and substitute them inside the model terms which is built using
    # feature and response names suffixed with '_scaled' in case they were scaled prior to training. 
    # Description to function _compute_pure_model_terms() also gives some details on usage of feature and 
    # response names suffixed with '_scaled', which are generated in this function (variables model_feat_names
    # and model_resp_names) and passed to _compute_pure_model_terms()).
    def _compute_model_terms_dict(self, algo, model, feat_names, resp_names, data_bounds, data_scaler,
            scale_features, scale_responses):
        assert not isinstance(model, dict)
        # were features and / or responses scaled prior building the model?
        feat_were_scaled = scale_features and data_scaler != 'none'
        resp_were_scaled = scale_responses and data_scaler != 'none'
        #print('computing model terms dict for response ' + str(resp_names))
        #assert len(resp_names) == 1
        # compute model terms
        model_feat_names = [self._scaled_name(feat) for feat in feat_names] if feat_were_scaled else feat_names #._scalerTermsInst
        model_resp_names = [self._scaled_name(resp) for resp in resp_names] if resp_were_scaled else resp_names #._scalerTermsInst
        #print('adding model terms: model_feat_names', model_feat_names, 'model_resp_names', model_resp_names, flush=True)

        model_term_dict = self._compute_pure_model_terms(algo, model, model_feat_names, model_resp_names, 
            feat_names, resp_names); #print('model_term_dict', model_term_dict)
        
        model_full_term_dict = model_term_dict;
        
        # compute features scaling term (skipped when it is identity);
        # substitute them instead of scaled feature variables in the model
        # terms (where the latter variables are inputs to the model)
        if feat_were_scaled:
            #print('adding feature scaler terms', data_bounds, feat_names, flush=True)
            #scaled_feat_names = [self._scalerTermsInst._scaled_name(feat)for feat in feat_names]
            feature_scaler_terms_dict = self.feature_scaler_terms(data_bounds, feat_names) #._scalerTermsInst
            #print('feature_scaler_terms_dict', feature_scaler_terms_dict, flush=True)

            for resp_name, model_term in model_term_dict.items():
                for feat_name, feat_term in feature_scaler_terms_dict.items():
                    #print('feat_name', feat_name, 'feat_term', feat_term, flush=True)
                    if self._tree_encoding == 'flat' and algo in ['dt_sklearn', 'rf_sklearn', 'et_sklearn', 'dt_caret', 'rf_caret', 'et_caret']: # rule_form:
                        model_term = [self.smlp_cnst_fold(form, {feat_name: feat_term}) for form in model_term]
                    else:
                        model_term = self.smlp_cnst_fold(model_term, {feat_name: feat_term}) #self.smlp_subst
                #print('model term after', model_term, flush=True)
                model_term_dict[resp_name] = model_term
            #print('model_term_dict', model_term_dict, flush=True)
            model_full_term_dict = model_term_dict
            
        # compute responses in original scale from scaled responses that are
        # the outputs of the modes, compose models with unscaled responses
        tree_flat_encoding = self._tree_encoding == 'flat' and algo in ['dt_sklearn', 'rf_sklearn', 'et_sklearn', 'dt_caret', 'rf_caret', 'et_caret']
        if resp_were_scaled and not tree_flat_encoding:
            responses_unscaler_terms_dict = self.feature_unscaler_terms(data_bounds, resp_names)           
            # substitute scaled response variables with scaled response terms (the model outputs)
            # in original response terms within responses_unscaler_terms_dict
            for resp_name, resp_term in responses_unscaler_terms_dict.items():
                #print('resp_name', resp_name, resp_term, flush=True)
                responses_unscaler_terms_dict[resp_name] = self.smlp_cnst_fold(resp_term, #self.smlp_subst 
                    {self._scaled_name(resp_name): model_term_dict[self._scaled_name(resp_name)]})
            #print('responses_unscaler_terms_dict full model', responses_unscaler_terms_dict, flush=True)
            model_full_term_dict = responses_unscaler_terms_dict
        
        if resp_were_scaled and tree_flat_encoding:
            responses_scaler_terms_dict = self.feature_scaler_terms(data_bounds, resp_names)
            #print('responses_scaler_terms_dict', responses_scaler_terms_dict)
            tree_counts = self._treeTermsInst.get_tree_model_estimator_count(algo, model)
            resp_names_numbered_trees = []
            data_bounds_numbered_trees = {}
            for j, resp in enumerate(resp_names):
                tree_count = tree_counts[j] if isinstance(model, dict) else tree_counts[0]
                for i in range(tree_count):
                    resp_unscaled = self._scalerTermsInst._unscaled_name(resp)
                    resp_names_numbered_trees.append(self._treeTermsInst._tree_resp_id(i, resp))
                    data_bounds_numbered_trees[self._treeTermsInst._tree_resp_id(i, resp)] = data_bounds[resp_unscaled]
            #print('resp_names_numbered_trees', resp_names_numbered_trees); print('data_bounds_numbered_trees', data_bounds_numbered_trees) ; 
            responses_scaler_terms_dict_numbered_trees = self.feature_scaler_terms(data_bounds_numbered_trees, resp_names_numbered_trees)
            #print('responses_scaler_terms_dict_numbered_trees', responses_scaler_terms_dict_numbered_trees)
            #print('model_full_term_dict befor', model_full_term_dict)
            assert len(model_full_term_dict) == 1
            key = list(model_full_term_dict.keys())[0]
            new_key = self._scalerTermsInst._unscaled_name(key); #print('key', key, 'new_key', new_key)
            model_full_term_dict_new = {}
            model_full_term_dict_new[new_key] = [self.smlp_cnst_fold(rule_formula, responses_scaler_terms_dict | responses_scaler_terms_dict_numbered_trees)
                for rule_formula in model_full_term_dict[key]]
            model_full_term_dict = model_full_term_dict_new
        #print('model_full_term_dict', model_full_term_dict, flush=True)
        resp_name = resp_names[0] if len(resp_names) == 1 else None
        with open(self.smlp_model_term_file(resp_name, True), 'w') as f:
            json.dump(str(model_full_term_dict), f, indent='\t', cls=np_JSONEncoder)
        return model_full_term_dict

    # Computes a dictionary with response names as keys and an smlp term corresponding to the model for that
    # response as the value for that key. Argument 'model_or_model_dict' is an ML model if one model was trained
    # for all the responses in training data, or is a dictionary of models if a separate model was trained for
    # each response (parameter model_per_response controls whether one model is trained for all responses or
    # individual models are trained for each response). This function simply iterates over the ML models
    # within model_or_model_dict and for each response calls _compute_model_terms_dict() to generate a term 
    # that emcodes the ML model as well as the scaling / unscaling constraints in cases features and / or 
    # reponses have been scaled prior to training.
    def compute_models_terms_dict(self, algo, model_or_model_dict, model_features_dict, feat_names, resp_names, 
            data_bounds, data_scaler,scale_features, scale_responses):

        #print('model_features_dict', model_features_dict); print('feat_names', feat_names, 'resp_names', resp_names)
        assert lists_union_order_preserving_without_duplicates(list(model_features_dict.values())) == feat_names
        #print('model_or_model_dict', model_or_model_dict)
        if isinstance(model_or_model_dict, dict):
            models_full_terms_dict = {}
            for i, resp_name in enumerate(model_or_model_dict.keys()):
                curr_feat_names = model_features_dict[resp_name]
                curr_model_full_term_dict = self._compute_model_terms_dict(algo, model_or_model_dict[resp_name], 
                    feat_names, [resp_name], data_bounds, data_scaler, scale_features, scale_responses)
                #print('join 1', models_full_terms_dict, '----2 -', curr_model_full_term_dict)
                models_full_terms_dict = models_full_terms_dict | curr_model_full_term_dict
        else:
            models_full_terms_dict = self._compute_model_terms_dict(algo, model_or_model_dict, 
                feat_names, resp_names, data_bounds, data_scaler, scale_features, scale_responses)
        #print('models_full_terms_dict', models_full_terms_dict); 
        for key, m in models_full_terms_dict.items():
            #print('m', m); print(self.smlp_destruct(m)); 
            if isinstance(m, list): # case where tree rules are coded as formulas
                assert key.startswith(algo)  #'flat_dt_sklearn_model' 'all_responses'
                ops = [self.smlp_count_operators(form) for form in m]
                ops = self.sum_operator_counts(ops)
            else:
                ops = self.smlp_count_operators(m); #print('ops', ops)
            self._smlp_terms_logger.info('Model operator counts for ' + str(key) + ': ' + str(ops))
        #print('compute_models_terms_dict', models_full_terms_dict)
        return models_full_terms_dict
    
    # This function computes orig_objv_terms_dict with names of objectives as keys and smlp terms
    # for objectives' expressions (spcified by used through command line or spec file) as the values; the  
    # inputs of objectives' expressions can only be feature and response names as declared in the domain. 
    # If scale_objv is true, then the objectives are min_max scaled to [0,1] and the correponding 
    # terms for scaled objectives in terms of original (unscaled) objectives are computed, and they are
    # returned as values in scaled objectives dictionary scaled_objv_terms_dict (which has names
    # of scaled objectives as the keys; these names as default have the form 'scaled_'+objv_name).
    # Finally, objv_terms_dict is computed which is the same as orig_objv_terms_dict if
    # scale_objv is False, and otherwise is a dictionary with scaled objectives' names as keys and
    # scaled objectives terms built only from smlp terms for original responses and features in the domain.
    # Dict objv_terms_dict is used in solver instance, while orig_objv_terms_dict and
    # scaled_objv_terms_dict are used to recover values of objectives and scaled objectives,
    # repectively, from sat assignments in counter examples.
    # TODO !!!: it might make sense to create a SmlpObjectives class and move this function there
    # maybe other functions like get_objectives() 
    def compute_objectives_terms(self, objv_names, objv_exprs, objv_bounds, scale_objv):
        #print('objv_exprs', objv_exprs)
        if objv_exprs is None:
            return None, None, None, None
        orig_objv_terms_dict = dict([(objv_name, self.ast_expr_to_term(objv_expr)) \
            for objv_name, objv_expr in zip(objv_names, objv_exprs)]) #self._smlpTermsInst.
        #print('orig_objv_terms_dict', orig_objv_terms_dict)
        if scale_objv:
            scaled_objv_terms_dict = self.feature_scaler_terms(objv_bounds, objv_names) #._scalerTermsInst
            #print('scaled_objv_terms_dict', scaled_objv_terms_dict)
            objv_terms_dict = {}
            for i, (k, v) in enumerate(scaled_objv_terms_dict.items()):
                #print('k', k, 'v', v, type(v)); 
                x = list(orig_objv_terms_dict.keys())[i]; 
                #print('x', x); print('arg', orig_objv_terms_dict[x])
                objv_terms_dict[k] = self.smlp_cnst_fold(v, {x: orig_objv_terms_dict[x]}) #self.smlp_subst
            #objv_terms_dict = scaled_objv_terms_dict
        else:
            objv_terms_dict = orig_objv_terms_dict
            scaled_objv_terms_dict = None
        
        if scaled_objv_terms_dict is not None:
            assert list(scaled_objv_terms_dict.keys()) == [self._scaled_name(objv_name) #._scalerTermsInst
                for objv_name in objv_names]
        #print('objv_terms_dict', objv_terms_dict)
        return objv_terms_dict, orig_objv_terms_dict, scaled_objv_terms_dict
    
    
    # Compute stability region theta; used also in generating lemmas during search for a stable solution. 
    # cex is assignement of values to knobs. Even if cex contains assignements to inputs, such assignements
    # are ignored as only variables which occur as keys in radii_dict are used for building theta.
    def compute_stability_formula_theta(self, cex, delta_dict:dict, radii_dict, universal=True): 
        #print('generate stability constraint theta')
        if delta_dict is not None:
            delta_abs = delta_dict['delta_abs']
            delta_rel = delta_dict['delta_rel']
            if delta_abs is not None:
                assert delta_abs >= 0 
            if delta_rel is not None:
                assert delta_rel >= 0
        else:
            delta_rel = delta_abs = None
            
        theta_form = self.smlp_true
        #print('radii_dict', radii_dict)
        radii_dict_local = radii_dict.copy() 
        knobs = radii_dict_local.keys(); #print('knobs', knobs); print('cex', cex); print('delta', delta_dict)
        
        # use inputs in theta computation, by setting radii to 0, and use delta if specified (not None)
        if not universal and delta_rel is not None:
            for cex_var in cex.keys():
                if cex_var not in knobs:
                    radii_dict_local[cex_var] = {'rad-abs':0, 'rad-rel': None} # delta
        
        for var,radii in radii_dict_local.items():
            var_term = self.smlp_var(var)
            # either rad-abs or rad-rel must be None -- for each var wr declare only one of these
            if radii['rad-abs'] is not None:
                rad = radii['rad-abs']; #print('rad', rad); 
                if delta_rel is not None: # we are generating a lemma
                    rad = rad * (1 + delta_rel) + delta_abs 
                rad_term = self.smlp_cnst(rad)
            elif radii['rad-rel'] is not None:
                rad = radii['rad-rel']; #print('rad', rad)
                if delta_rel is not None: # we are generating a lemma
                    rad = rad * (1 + delta_rel) + delta_abs
                rad_term = self.smlp_cnst(rad)
                
                # TODO !!!  issue a warning when candidates become closer and closer
                # TODO !!!!!!! warning when distance between previous and current candidate
                # TODO !!!!!! warning when FINAL rad + delta is 0, as part of sanity checking options
                # when rad and delta are both 0, then exclude at least this candidate  
                # global control on warning messages
                # abs(!delta_dict ? e : nm); !delta_dict means the argument cex is sat-model for candidate, we use constant from  
                # the model; delta_dict means we want lemma to exclude region around cex to candidate, we want lemma to restrict 
                # the next candidate; the circle to rule out from search is around cex to counter-example, but the radius is 
                # calculated based on candidate values, not the values in the counter-example to candidate.
                #
                # Condition "delta_dict is None" (= !delta_dict) means the argument cex is a sat-model defining a candidate, 
                # and in this case we use constants from the cex as the factors to compute absolute radius from relative radius, 
                # for each cex variable. Condition "delata is not None" means the argument cex is a counter-example to candidate, 
                # and we want a lemma to exclude a circle-shaped region around cex to candidate, while we want the radious of 
                # this circle to be computed based on the values in the candidate and not values in the counter-example to the 
                # candidate; this is a matter of definition of relative radius, and seems cleaner than computing actual radius 
                # from relative radius based on variable values in the counter-exaples to candidate rather than variable values 
                # in the candidate itself.
                if delta_rel is not None: # radius for a lemma -- cex holds values of candidate counter-example
                    rad_term = rad_term * abs(var_term)
                else: # radius for excluding a candidate -- cex holds values of the candidate 
                    rad_term = rad_term * abs(cex[var])
            elif delta_dict is not None: 
                raise exception('When delta dictionary is provided, either absolute or relative or delta must be specified') 
            theta_form = self.smlp_and(theta_form, ((abs(var_term - cex[var])) <= rad_term))
        #print('theta_form', theta_form)
        return theta_form
    
    # Creates eta constraints on control parameters (knobs) from the spec.
    # Covers grid as well as range/interval constraints.
    def compute_grid_range_formulae_eta(self):
        #print('generate eta constraint')
        eta_grid_form = self.smlp_true
        eta_grids_dict = self._specInst.get_spec_eta_grids_dict; #print('eta_grids_dict', eta_grids_dict)
        for var,grid in eta_grids_dict.items():
            eta_grid_disj = self.smlp_false
            var_term = self.smlp_var(var)
            for gv in grid: # iterate over grid values
                if eta_grid_disj == self.smlp_false:
                    eta_grid_disj = var_term == self.smlp_cnst(gv)
                else:
                    eta_grid_disj = self.smlp_or(eta_grid_disj, var_term == self.smlp_cnst(gv))
            if eta_grid_form == self.smlp_true:
                eta_grid_form = eta_grid_disj
            else:
                eta_grid_form = self.smlp_and(eta_grid_form, eta_grid_disj)
        #print('eta_grid_form', eta_grid_form); 
        return eta_grid_form
                

    # Compute formulae alpha, beta, eta from respective expression string.
    def compute_input_ranges_formula_alpha(self, model_inputs):
        alpha_form = self.smlp_true
        alpha_dict = self._specInst.get_spec_alpha_bounds_dict; #print('alpha_dict', alpha_dict)
        for v,b in alpha_dict.items():
            if v not in model_inputs:
                continue
            mn = b['min']
            mx = b['max']
            #print('mn', mn, 'mx', mx)
            if mn is not None and mx is not None:
                if self._declare_domain_interface_only:
                    if self._encode_input_range_as_disjunction:
                        rng = self.smlp_or_multi([self.smlp_eq(self.smlp_var(v), self.smlp_cnst(i)) for i in range(mn, mx-1)])
                    else:
                        rng = self.smlp_and(self.smlp_var(v) >= self.smlp_cnst(mn), self.smlp_var(v) <= self.smlp_cnst(mx))
                    alpha_form = self.smlp_and(alpha_form, rng)
            elif mn is not None:
                rng = self.smlp_var(v) >= self.smlp_cnst(mn)
                alpha_form = self.smlp_and(alpha_form, rng)
            elif mx is not None:
                rng = self.smlp_var(v) <= self.smlp_cnst(mx)
                alpha_form = self.smlp_and(alpha_form, rng)
            else:
                assert False
        return alpha_form
    
    def compute_input_ranges_formula_alpha_eta(self, alpha_vs_eta, model_inputs):
        alpha_or_eta_form = self.smlp_true
        if alpha_vs_eta == 'alpha':
            alpha_or_eta_ranges_dict = self._specInst.get_spec_alpha_bounds_dict
        elif alpha_vs_eta == 'eta':
            alpha_or_eta_ranges_dict = self._specInst.get_spec_eta_bounds_dict
        else:
            raise Exception('Unsupported value ' + str(alpha_vs_eta) + \
                ' of argument alpha_vs_eta in function compute_input_ranges_formula_alpha_eta')
        #print(alpha_vs_eta, 'alpha_or_eta_ranges_dict', alpha_or_eta_ranges_dict)
        for v,b in alpha_or_eta_ranges_dict.items():
            if v not in model_inputs:
                continue
            mn = b['min']
            mx = b['max']
            #print('mn', mn, 'mx', mx)
            if mn is not None and mx is not None:
                if self._declare_domain_interface_only:
                    if self._encode_input_range_as_disjunction and alpha_vs_eta == 'alpha' and v in self._specInst.get_spec_inputs:
                        rng = self.smlp_or_multi([self.smlp_eq(self.smlp_var(v), self.smlp_cnst(i)) for i in range(mn, mx+1)])
                    else:
                        rng = self.smlp_and(self.smlp_var(v) >= self.smlp_cnst(mn), self.smlp_var(v) <= self.smlp_cnst(mx))
                    alpha_or_eta_form = self.smlp_and(alpha_or_eta_form, rng)
            elif mn is not None:
                rng = self.smlp_var(v) >= self.smlp_cnst(mn)
                alpha_or_eta_form = self.smlp_and(alpha_or_eta_form, rng)
            elif mx is not None:
                rng = self.smlp_var(v) <= self.smlp_cnst(mx)
                alpha_or_eta_form = self.smlp_and(alpha_or_eta_form, rng)
            else:
                assert False
        return alpha_or_eta_form
    
    # alph_expr is alpha constraint specified in command line. If it is not None 
    # it overrides alpha constraint defined in spec file through feild "alpha".
    # Otherwise we get definition of alpha if it is specified in spec, and otherwise
    # global alha is None and global alpha constraint formula is smlp.true
    # This function computes alpha constraints originated from spec input variable 
    # "bounds" escription using compute_input_ranges_formula_alpha and combines
    # with global alpha constraint (retunrns the conjunction).
    def compute_global_alpha_formula(self, alph_expr, model_inputs):
        #alph_form = self.compute_input_ranges_formula_alpha(model_inputs) 
        #alph_form = self.smlp_true
        if alph_expr is None:
            alpha_expr = self._specInst.get_spec_alpha_global_expr
        if alph_expr is None:
            return self.smlp_true
        else:
            alph_expr_vars = get_expression_variables(alph_expr)
            dont_care_vars = list_subtraction_set(alph_expr_vars, model_inputs)
            if len(dont_care_vars) > 0:
                raise Exception('Variables ' + str(dont_care_vars) + 
                    ' in input constraints (alpha) are not part of the model')
            alph_glob = self.ast_expr_to_term(alph_expr)
            return alph_glob #self._smlpTermsInst.smlp_and(alph_form, alph_glob)
    
    # The argument model_inps_outps is the union of model input and output varaiables.
    # We want to make sure that the global beta constraint beta_expr contains only variables 
    # in model_inps_outps, so that the (global) beta constraint on the model is well defined.
    # Besides the above sanity check, this function simply converts beta constraint expression
    # into smlp formula. Reminder: input variables that occur in globas alpha, beta and eta
    # constrints are added to variables that shoud be kept as model inputs during model training,
    # and are not dropped during data processing (see function SmlpData._prepare_data_for_modeling).
    def compute_beta_formula(self, beta_expr, model_inps_outps):
        if beta_expr is None:
            return self.smlp_true
        else:
            beta_expr_vars = get_expression_variables(beta_expr)
            dont_care_vars = list_subtraction_set(beta_expr_vars, model_inps_outps)
            if len(dont_care_vars) > 0:
                raise Exception('Variables ' + str(dont_care_vars) + 
                    ' in optimization constraints (beta) are not part of the model')
            return self.ast_expr_to_term(beta_expr)
    
    def compute_eta_formula(self, eta_expr, model_inputs):
        if eta_expr is None:
            return self.smlp_true
        else:
            # eta_expr can only contain knobs (control inputs), not free inputs or outputs (responses)
            eta_expr_vars = get_expression_variables(eta_expr)
            dont_care_vars = list_subtraction_set(eta_expr_vars, model_inputs)
            if len(dont_care_vars) > 0:
                raise Exception('Variables ' + str(dont_care_vars) + 
                    ' in knob constraints (eta) are not part of the model')
            return self.ast_expr_to_term(eta_expr)

    def var_domain(self, var, spec_domain_dict):
        interval = spec_domain_dict[var][self._SPEC_DOMAIN_INTERVAL_TAG]; #self._specInst.get_spec_interval_tag
        if interval is None:
            interval_has_none = True
        elif interval[0] is None or interval[1] is None:
            interval_has_none = True
        else:
            interval_has_none = False
        if spec_domain_dict[var][self._SPEC_DOMAIN_RANGE_TAG] == self._specInst.get_spec_integer_tag: # self._specInst.get_spec_range_tag
            if self._declare_integer_as_real_with_grid and not interval_has_none:
                var_component = smlp.component(self.smlp_real, grid=list(range(interval[0], interval[1]+1)))
                assert False
            if self._declare_domain_interface_only or interval_has_none:
                var_component = smlp.component(self.smlp_integer)
            else:
                var_component = smlp.component(self.smlp_integer, 
                    interval=spec_domain_dict[var][self._SPEC_DOMAIN_INTERVAL_TAG]) #self._specInst.get_spec_interval_tag
        elif spec_domain_dict[var][self._SPEC_DOMAIN_RANGE_TAG] == self._specInst.get_spec_real_tag: #self._specInst.get_spec_range_tag
            if self._declare_domain_interface_only or interval_has_none:
                var_component = smlp.component(self.smlp_real)
            else:
                var_component = smlp.component(self.smlp_real, 
                    interval=spec_domain_dict[var][self._SPEC_DOMAIN_INTERVAL_TAG]) #self._specInst.get_spec_interval_tag
        return var_component
    
    # this function builds terms and formulas for constraints, system description and the models
    def create_model_exploration_base_components(self, syst_expr_dict:dict, algo, model, model_features_dict:dict, feat_names:list, resp_names:list, 
            alph_expr:str, beta_expr:str, eta_expr:str, data_scaler, scale_feat, scale_resp, 
            float_approx=True, float_precision=64, data_bounds_json_path=None):
        self._smlp_terms_logger.info('Creating model exploration base components: Start')
        #print('data_bounds_json_path', data_bounds_json_path)
        self._smlp_terms_logger.info('Parsing the SPEC: Start')
        if data_bounds_json_path is not None:
            with open(data_bounds_json_path, 'r') as f:
                data_bounds = json.load(f, parse_float=Fraction)
        else:
            raise Exception('Data bounds file cannot be loaded')
        self._smlp_terms_logger.info('Parsing the SPEC: End') 
        
        # get variable domains dictionary; certain sanity checks are performrd within this function.
        spec_domain_dict = self._specInst.get_spec_domain_dict; #print('spec_domain_dict', spec_domain_dict)
        
        # contraints on features used as control variables and on the responses
        alph_ranges = self.compute_input_ranges_formula_alpha_eta('alpha', feat_names); #print('alph_ranges')
        alph_global = self.compute_global_alpha_formula(alph_expr, feat_names); #print('alph_global')
        alpha = self.smlp_and(alph_ranges, alph_global); #print('alpha')
        beta = self.compute_beta_formula(beta_expr, feat_names+resp_names); #print('beta')
        eta_ranges = self.compute_input_ranges_formula_alpha_eta('eta', feat_names); #print('eta_ranges')
        eta_grids = self.compute_grid_range_formulae_eta(); #print('eta_grids')
        eta_global = self.compute_eta_formula(eta_expr, feat_names); #print('eta_global', eta_global)
        eta = self.smlp_and_multi([eta_ranges, eta_grids, eta_global]); #print('eta', eta)
        
        self._smlp_terms_logger.info('Alpha global   constraints: ' + str(alph_global))
        self._smlp_terms_logger.info('Alpha ranges   constraints: ' + str(alph_ranges))
        self._smlp_terms_logger.info('Alpha combined constraints: ' + str(alpha))
        self._smlp_terms_logger.info('Beta  global   constraints: ' + str(beta))
        self._smlp_terms_logger.info('Eta   ranges   constraints: ' + str(eta_ranges))
        self._smlp_terms_logger.info('Eta   grid     constraints: ' + str(eta_grids))
        self._smlp_terms_logger.info('Eta   global   constraints: ' + str(eta_global))
        self._smlp_terms_logger.info('Eta   combined constraints: ' + str(eta))
        self._smlp_terms_logger.info('Creating model exploration base components: End')

        # Create solver domain from the dictionary of varibale types, range and grid specificaton.
        # First we create solver domain that includes declarations of inputs and knobs only, 
        # in order to check consistency of alpha and eta constraints (without model constraints). 
        # Then we create solver domain that includes declarations od inputs, knobs and outputs,
        # and check consistency of alapha, eta and together with constraints that define the model.
        domain_dict = {}
        #print('model_features_dict', model_features_dict, 'feat_names', feat_names)
        
        # define domain from inputs and knobs only and check alpha and eta constraints are consistent
        for var in feat_names:
            domain_dict[var] = self.var_domain(var, spec_domain_dict)
        domain_features = smlp.domain(domain_dict)
        interface_consistent = self.check_alpha_eta_consistency(domain_features, None, alpha, eta, 'ALL')
        if not interface_consistent:
            return None, None, None, eta, alpha, beta, False, False
        #print('interface_consistent', interface_consistent)
        # now define solver donain that includes input, knob and output declarations from spec file.
        for var in resp_names:
            domain_dict[var] = self.var_domain(var, spec_domain_dict)
        
        # when we use flat encoding for tree models, we define tree_i_resp variables that represent
        # responses resp computed by the i-th tree, and we need to declare them within domain
        if algo in ['dt_sklearn', 'rf_sklearn', 'et_sklearn', 'dt_caret', 'rf_caret', 'et_caret'] and self._tree_encoding == 'flat':
            tree_counts = self._treeTermsInst.get_tree_model_estimator_count(algo, model)
            model_per_resp = isinstance(model, dict)
            assert tree_counts is not None 
            #print('resp_names', resp_names, 'tree_counts', tree_counts)
            for j, resp in enumerate(resp_names):
                tree_count = tree_counts[j] if isinstance(model, dict) else tree_counts[0]
                for i in range(tree_count):
                    tree_resp_name = self._treeTermsInst._tree_resp_id(i, resp)
                    domain_dict[tree_resp_name] = self.var_domain(resp, spec_domain_dict)

        #print('domain_dict', domain_dict)
        domain = smlp.domain(domain_dict)
        
        if syst_expr_dict is not None:
            self._smlp_terms_logger.info('Building system terms: Start')
            for resp, syst_expr in syst_expr_dict.items():
                feat = self.get_expression_variables(syst_expr)
                if set(feat) != set(model_features_dict[resp]):
                    #print('resp', resp, 'syst_feat', feat, 'model_feat', model_features_dict[resp])
                    raise Exception('System and model features do not match for response ' + str(resp))
            #print('syst_expr_dict', syst_expr_dict)
            system_term_dict = dict([(resp_name, self.ast_expr_to_term(resp_expr)) \
                for resp_name, resp_expr in syst_expr_dict.items()]); #print('system_term_dict', system_term_dict); 
            self._smlp_terms_logger.info('System terms dictionary: ' + str(system_term_dict))
            self._smlp_terms_logger.info('Building system terms: End')
        else:
            system_term_dict = None
        
        self._smlp_terms_logger.info('Building model terms: Start')
        # model terms with terms for scaling inputs and / or unscaling responses are all composed to 
        # build model terms with inputs and outputs in the original scale. 
        #print('algo', algo, flush=True)
        if algo == 'system':
            if syst_expr_dict is None:
                raise Exception('System must be specified when model training algorithm is "system"')
            model_full_term_dict = system_term_dict
        else:
            #print('model', model, flush=True)
            model_full_term_dict = self.compute_models_terms_dict(algo, model, 
                model_features_dict, feat_names, resp_names, data_bounds, data_scaler, scale_feat, scale_resp)
        self._smlp_terms_logger.info('Building model terms: End')
        
        model_consistent = self.check_alpha_eta_consistency(domain, model_full_term_dict, alpha, eta, 'ALL')
        if not model_consistent:
            return domain, system_term_dict, model_full_term_dict, eta, alpha, beta, True, False

        if self._simplify_terms:
            if model_full_term_dict is not None:
                for r, m in model_full_term_dict.items():
                    model_full_term_dict[r] = self.smlp_simplify(m)
            if system_term_dict is not None:
                for r, m in system_term_dict.items():
                    system_term_dict[r] = self.smlp_simplify(m)
            if eta is not None:
                eta = self.smlp_simplify(eta)
            if alpha is not None:
                alpha = self.smlp_simplify(alpha)
            if beta is not None:
                beta = self.smlp_simplify(beta)
        
        return domain, system_term_dict, model_full_term_dict, eta, alpha, beta, interface_consistent, model_consistent
    
    # create base solver instance with model constraints, declare logic and (non/)incremental mode
    def create_model_exploration_instance_from_smlp_components(self, domain, model_full_term_dict, incremental, logic):
        # create base solver -- it has model and other constraints that are common for all model
        # exloration modes in SMLP: querying, assertion verification, configuration optimization
        if logic is not None and logic != 'ALL':
            base_solver = smlp.solver(incremental, logic)
        else:
            #base_solver = smlp.solver(incremental=incremental)
            base_solver = smlp.solver(incremental, 'ALL')
        base_solver.declare(domain)
        
        if model_full_term_dict is not None :
            if self._tree_encoding == 'flat' and isinstance(list(model_full_term_dict.values())[0], list): # algo == 'dt_sklearn':
                for k, v in model_full_term_dict.items():
                    for rule_formula in v:
                        #print('rule_formula', rule_formula)
                        base_solver.add(rule_formula)
            else:
                # let solver know definition of responses (the model's function)
                for resp_name, resp_term in model_full_term_dict.items():
                    eq_form = self.smlp_eq(self.smlp_var(resp_name), resp_term)
                    base_solver.add(eq_form)
        return base_solver
    
    # wrapper function on solver.check to measure runtime and return status in a convenient way
    def smlp_solver_check(self, solver, call_name:str, lemma_precision:int=0):
        approx_lemmas =  lemma_precision > 0
        start = time.time()
        #print('solver chack start', flush=True)
        res = solver.check()
        #print('solver chack end', flush=True)
        end = time.time()
        if  isinstance(res, smlp.unknown):
            #print('smlp_unknown', smlp.unknown)
            status = 'unknown'
            sat_model = {}
        elif isinstance(res, smlp.sat):
            #print('smlp_sat', smlp.sat)
            status = 'sat'
            sat_model = self.witness_term_to_const(res.model, approximate=False, precision=None)
            if approx_lemmas:
                sat_model_approx = self.approximate_witness_term(res.model, lemma_precision)
            #print('res.model', res.model, 'sat_model', sat_model)
        elif isinstance(res, smlp.unsat):
            #print('smlp_unsat', smlp.unsat)
            status = 'unsat'
            sat_model = {}
        else:
            raise Exception('Unexpected solver result ' + str(res))
        
        anonym_interface_dict = self._specInst.get_anonymized_interface; #print('anonym_interface_dict', anonym_interface_dict)
        
        # genrate columns for trace file to enable viewing candidates and counter-example in a convenient way
        if call_name == 'interface_consistency':
            if self._trace_anonymize:
                interface_column_names = list(anonym_interface_dict['knobs'].values()) +  list(anonym_interface_dict['inputs'].values()) \
                    + list(anonym_interface_dict['outputs'].values())
            else:
                interface_column_names = list(anonym_interface_dict['knobs'].keys()) +  list(anonym_interface_dict['inputs'].keys()) \
                    + list(anonym_interface_dict['outputs'].keys())
            if self._trace_runtime == 0:
                self._smlp_terms_tracer.info(','.join(['stage', 'solver'] + interface_column_names))
            else:
                self._smlp_terms_tracer.info(','.join(['stage', 'solver', 'runtime'] + interface_column_names))
        
        if status == 'sat':
            assignment = {}
            assignment_approx = {}
            for k in sat_model.keys():
                if k in anonym_interface_dict['knobs'].keys():
                    name = k if not self._trace_anonymize else anonym_interface_dict['knobs'][k]
                elif k in anonym_interface_dict['inputs'].keys():
                    name = k if not self._trace_anonymize else anonym_interface_dict['inputs'][k]
                elif k in anonym_interface_dict['outputs'].keys():
                    name = k if not self._trace_anonymize else anonym_interface_dict['outputs'][k]
                else:
                    # TODO !!!!!!!!!!!: maybe expose tree values per response? will need to enhance function get_anonymized_interface
                    name = None # covers case say when we have tree models and domain contain response declaration per model
                if name is not None:
                    assignment[name] = sat_model[k] if self._trace_precision == 0 else round(float(sat_model[k]), self._trace_precision)
                    if approx_lemmas:
                        assignment_approx[name] = sat_model_approx[k] if self._trace_precision == 0 else round(float(sat_model[k]), self._trace_precision)
            
            #print('sat_model', sat_model); print('assignment', assignment, self._trace_anonymize)
            #print('anonym_interface_dict', anonym_interface_dict)
            if self._trace_anonymize:
                knob_values = [str(assignment[e]) for e in list(anonym_interface_dict['knobs'].values()) if e in assignment.keys()]
                input_values = [str(assignment[e]) for e in list(anonym_interface_dict['inputs'].values()) if e in assignment.keys()]
                output_values = [str(assignment[e]) for e in list(anonym_interface_dict['outputs'].values()) if e in assignment.keys()]
                if approx_lemmas:
                    knob_values_approx = [str(assignment_approx[e]) for e in list(anonym_interface_dict['knobs'].values()) if e in assignment.keys()]
                    input_values_approx = [str(assignment_approx[e]) for e in list(anonym_interface_dict['inputs'].values()) if e in assignment.keys()]
                    output_values_approx = [str(assignmentv[e]) for e in list(anonym_interface_dict['outputs'].values()) if e in assignment.keys()]
            else:
                knob_values = [str(assignment[e]) for e in list(anonym_interface_dict['knobs'].keys()) if e in assignment.keys()]
                input_values = [str(assignment[e]) for e in list(anonym_interface_dict['inputs'].keys()) if e in assignment.keys()]
                output_values = [str(assignment[e]) for e in list(anonym_interface_dict['outputs'].keys()) if e in assignment.keys()]
                if approx_lemmas:
                    knob_values_approx = [str(assignment_approx[e]) for e in list(anonym_interface_dict['knobs'].keys()) if e in assignment.keys()]
                    input_values_approx = [str(assignment_approx[e]) for e in list(anonym_interface_dict['inputs'].keys()) if e in assignment.keys()]
                    output_values_approx = [str(assignment_approx[e]) for e in list(anonym_interface_dict['outputs'].keys()) if e in assignment.keys()]
        else:
            knob_values = input_values = output_values = knob_values_approx = input_values_approx = output_values_approx = []
        #print('knob_values', [type(kv) for kv in knob_values], 'input_values', [type(iv) for iv in input_values], 'output_values', [type(ov) for ov in output_values])
        if self._trace_runtime == 0:
            self._smlp_terms_tracer.info(','.join([call_name, status] + knob_values + input_values + output_values))
            if approx_lemmas and status == 'sat':
                self._smlp_terms_tracer.info(','.join([call_name+'_approx', status] + knob_values_approx + input_values_approx + output_values_approx))
        else:
            elapsed = round(end - start, self._trace_runtime)
            self._smlp_terms_tracer.info(','.join([call_name, status, str(elapsed)] + knob_values + input_values + output_values))
            if approx_lemmas and status == 'sat':
                self._smlp_terms_tracer.info(','.join([call_name+'_approx', status, str(elapsed)] + knob_values_approx + input_values_approx + output_values_approx))
        #if status == 'sat' and approx_lemmas:
            #print('res', type(res), res)
            #print('res.mode;', res.model, 'assignment', assignment, 'assignment_approx', assignment_approx); 
            #return res, assignment_approx
        #print('exit smlp_solver_check', flush=True)
        return res
    
    def solver_status_sat(self, res):
        return isinstance(res, smlp.sat)
        
    def solver_status_unsat(self, res):
        return isinstance(res, smlp.unsat)
        
    def solver_status_unknown(self, res):
        return isinstance(res, smlp.unknown)
        
    # we return value assignmenets to interface (input, knob, output) variables defined in the Spec file
    # (and not values assigned to any other variables that might be defined additionally as part of solver domain,
    # like variables tree_i_resp that we decalre as part of domain for tree models with flat encoding).
    def get_solver_model(self, res):
        if self.solver_status_sat(res):
            reduced_model = dict((k,v) for k,v in res.model.items() if k in self._specInst.get_spec_interface)
            return reduced_model
        else:
            return None
    
    # function to check that alpha and eta constraints on inputs and knobs are consistent.
    # TODO: model_full_term_dict is not required here but omiting it causes z3 error 
    # result smlp::z3_solver::check(): Assertion `m.num_consts() == size(symbols)' failed.
    # This is likely because the domain declares model outputs as well and without 
    # model_full_term_dict these outputs have no logic (no definition). This function is
    # not a performance bottleneck, but if one wants to speed it up one solution could be
    # to create alpha_eta domain without declaring the outputs and feed it to this function 
    # instead of the domain that contains output declarations as well (the argument 'domain').
    def check_alpha_eta_consistency(self, domain:smlp.domain, model_full_term_dict:dict, 
            alpha:smlp.form2, eta:smlp.form2, solver_logic:str):
        #print('create solver: model', model_full_term_dict, flush=True)
        solver = self.create_model_exploration_instance_from_smlp_components(
            domain, model_full_term_dict, False, solver_logic)
        #print('add alpha', alpha, flush=True)
        solver.add(alpha); #print('alpha', alpha, flush=True)
        solver.add(eta); #print('eta', eta)
        #print('create check', flush=True)
        #res = solver.check(); print('res', res, flush=True)
        res = self.smlp_solver_check(solver, 'interface_consistency' if model_full_term_dict is None else 'model_consistency')
        consistency_type = 'Input and knob' if model_full_term_dict is None else 'Model'
        if isinstance(res, smlp.sat):
            self._smlp_terms_logger.info(consistency_type + ' interface constraints are consistent')
            interface_consistent = True
        elif isinstance(res, smlp.unsat):
            self._smlp_terms_logger.info(consistency_type + ' interface constraints are inconsistent')
            interface_consistent = False
        else:
            raise Exception('alpha and eta cosnsistency check failed to complete')
        return interface_consistent

