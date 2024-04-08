# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

import operator
import numpy as np
import pandas as pd
#from keras.models import load_model as keras_load_model
#from tensorflow import keras
import keras
from sklearn.tree import _tree
import json
import ast
import builtins
import operator as op
from fractions import Fraction

import smlp
from smlp_py.smlp_utils import (np_JSONEncoder, lists_union_order_preserving_without_duplicates, 
    list_subtraction_set, get_expression_variables)
#from smlp_py.smlp_spec import SmlpSpec

def _keras_is_sequential(model):
    try:
        # v2.9 has this API
        clz = keras.engine.sequential.Sequential
    except AttributeError:
        # v2.14+ has this API
        clz = keras.src.engine.sequential.Sequential
    return isinstance(model, clz)

def _keras_is_functional(model):
    try:
        # v2.9 has this API
        clz = keras.engine.functional.Functional
    except AttributeError:
        # v2.14+ has this API
        clz = keras.src.engine.functional.Functional
    return isinstance(model, clz)

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

# Class SmlpTerms has methods for generating terms, and classes TreeTerms, PolyTerms and NNKerasTerms are inherited
# from it but this inheritance is probably not implemented in the best way: TODO !!!: see if that can be improved.
class SmlpTerms:
    def __init__(self):
        self._smlp_terms_logger = None
        self.report_file_prefix = None
        self.model_file_prefix = None

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

    # set logger from a caller script
    def set_logger(self, logger):
        self._smlp_terms_logger = logger

    def smlp_not(self, form:smlp.form2):
        #res1 = ~form
        res2 = op.inv(form)
        #assert res1 == res2
        return res2 #~form
    
    def smlp_and(self, form1:smlp.form2, form2:smlp.form2):
        ''' test 83 gets stuck with this simplification
        if form1 == smlp.true:
            return form2
        if form2 == smlp.true:
            return form1
        '''
        res1 = op.and_(form1, form2)
        #res2 = form1 & form2
        #print('res1', res1, type(res1)); print('res2', res2, type(res2))
        #assert res1 == res2
        return res1 # form1 & form2
    
    def smlp_and_multi(self, form_list:list[smlp.form2]):
        res = smlp.true
        for form in form_list:
            res = form if res is smlp.true else self.smlp_and(res, form)
        return res
    
    def smlp_or(self, form1:smlp.form2, form2:smlp.form2):
        res1 = op.or_(form1, form2)
        #res2 = form1 | form2
        #assert res1 == res2
        return res1 #form1 | form2
    
    def smlp_eq(self, term1:smlp.term2, term2:smlp.term2):
        res1 = op.eq(term1, term2)
        #res2 = term1 == term2; print('res1', res1, 'res2', res2)
        #assert res1 == res2
        return res1
    
    def smlp_add(self, term1:smlp.term2, term2:smlp.term2):
        return term1 + term2

    def smlp_mult(self, term1:smlp.term2, term2:smlp.term2):
        return term1 * term2    
    
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
    def ast_expr_to_term(self, expr):
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

    # Compute numeric values of smlp ground terms.  
    # TODO !!!: intend to extend to ground formulas as well. Currently an assertion prevents this usage"
    # assertion checks that the constant expression is rational Q, not real R or algebraic A and also
    # nothing else like Boolean/formula type
    def ground_smlp_expr_to_value(self, expr, approximate=False, precision=64):
        # evaluate to constant term or formula (evaluate all operations in the expressions) -- should
        # succeed because the assumption is that expr does not contain variables
        #print('ground_smlp_expr_to_value: expr', expr)
        #print('applying smlp.cnst_fold to ', expr)
        smlp_const = smlp.cnst_fold(expr); #print('smlp_const', smlp_const)
        assert isinstance(smlp.Cnst(smlp_const), smlp.libsmlp.Q) or isinstance(smlp.Cnst(smlp_const), smlp.libsmlp.A) 
        if isinstance(smlp.Cnst(smlp_const), smlp.libsmlp.A) or isinstance(smlp.Cnst(smlp_const), smlp.libsmlp.R): 
            # algebraic number, solution of a polynomial, need to specify precision for the case
            # value_type is not float (for float, precison is always 64)
            val = smlp.approx(smlp.Cnst(smlp_const), precision=precision)
        elif isinstance(smlp.Cnst(smlp_const), smlp.libsmlp.Q): 
            if approximate:
                val = smlp.approx(smlp.Cnst(smlp_const), precision=precision)
            else:
                try:
                    if smlp.Cnst(smlp_const).denominator is not None and smlp.Cnst(smlp_const).numerator is not None:
                        val = Fraction(smlp.Cnst(smlp_const).numerator, smlp.Cnst(smlp_const).denominator)
                except Exception as err:
                    self._smlp_terms_logger.error(f"Unexpected {err=}, {type(err)=}")
                    #print(f"Unexpected {err=}, {type(err)=}")
                    raise
        else:
            raise Exception('Failed to computel value for smlp expression ' + str(expr) + 
                ' of type ' + str(type(expr)))
        #print('smlp expr val', val)
        return val
    
    # Converts values in sat assignmenet (witness) from terms to python fractions when the value
    # itself is of type smlp.libsmlp.Q (same as smlp.Q) and the argument approximate is set to False; 
    # and otherwise, as default converts into approximate float value if value_type is float or 
    # Note that the 'precision' parameter is ignored for type=float -- precision is defaulted to 64.
    def witness_term_to_const(self, witness, approximate=False, precision=64, value_type=float):
        assert value_type in [float, smlp.Q, smlp.R]
        witness_vals_dict = {}
        for k,t in witness.items():
            witness_vals_dict[k] = self.ground_smlp_expr_to_value(t, approximate, precision)
        return witness_vals_dict

    # Converts values in sat assignmenet (witness) from python fractions to terms.
    def witness_const_to_term(self, witness):
        witness_vals_dict = {}
        for k,t in witness.items():
            witness_vals_dict[k] = smlp.Cnst(t)
        return witness_vals_dict
    
    # Converts values assugnments in sat assignmenet (witness) from to smlp formula
    # equal to conjucnction of equalities smlp.Var(k) = smlp.Cnst(t)
    def witness_to_formula(self, witness):
        witness_formula = []
        for k,t in witness.items():
            witness_formula.append(self.smlp_eq(smlp.Var(k), smlp.Cnst(t)))
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
        self.instSmlpTerms = SmlpTerms()
    
    # set logger from a caller script
    def set_logger(self, logger):
        self._smlp_terms_logger = logger

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
                # in th enext line is defined as hard coded for resp_names[0] -- meaning, the last index [0].
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
        return self._apply_op(p[1], smlp.Var(p[0]), smlp.Cnst(p[2]))

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
    def rules_to_term(self, rules):
        # Convert the antecedent and consequent of a rule (corresponding to a full branch in a tree)
        # into smlp terms and return a dictionary with response names as the keys and pairs of terms
        # (antecdent_term, consequent_term) as the values. This function should be used for a rule
        # within a list of rules that represent a tree model.
        def rule_to_term(rule):
            antecedent = rule['antecedent']; #print('antecedent', antecedent)
            consequent = rule['consequent']; #print('consequent', consequent)
            if len(antecedent) == 0:
                ant = smlp.true
            else:
                ant = self._rule_triplet_to_term(antecedent[0])
            for i, p in enumerate(antecedent):
                if i > 0:
                    ant = ant & self._rule_triplet_to_term(p)
            res_dict = {}
            for resp, val in consequent.items():
                #term = smlp.Ite(ant, smlp.Cnst(val), smlp.Var('SMLP_UNDEFINED'))
                #res_dict[resp] = term
                res_dict[resp] = (ant, smlp.Cnst(val))
            return res_dict
        
        # returned value
        rules_dict = {}
        for i, rule in enumerate(rules):
            # this is an example of a rule (corresponds to a branch in a decision/regression tree):
            # {'antecedent': [('p3', '>', 0.4000000134110451), ('FMAX_abc', '<=', 0.75), ('FMAX_xyz', '>', 
            #     0.5000000149011612)], 'consequent': {'num1': 0.0, 'num2': 0.0}, 'coverage': 2}
            #print('\n ====== i', i, 'rule', rule)
            rule_dict = rule_to_term(rule)
            # here is how the corresponding rule_dict looks like (it contains smlp / smt2 terms):
            # {'num1': (<smlp.libsmlp.form2 (and (and (> p3 (/ 53687093 134217728)) (<= FMAX_abc (/ 3 4))) 
            #     (> FMAX_xyz (/ 33554433 67108864)))>, <smlp.libsmlp.term2 0>), 
            #  'num2': (<smlp.libsmlp.form2 (and (and (> p3 (/ 53687093 134217728)) (<= FMAX_abc (/ 3 4))) 
            #     (> FMAX_xyz (/ 33554433 67108864)))>, <smlp.libsmlp.term2 0>)}
            #print('rule_dict', rule_dict)
            for resp, (ant_term, con_term) in rule_dict.items():            
                if i == 0:
                    # The condition along a branch of a decision tree is implied by disjunction 
                    # of conditions along the rest of the branches, thus can be omitted. In this
                    # implementation, we choose do omit the condition along the first branch.
                    # TODO: Ideally, we could implement a sanity check that the condition along 
                    # the forst branch is implied by the conjunction of conditions along the rest
                    # of the branches -- using a solver like Z3.
                    rules_dict[resp] = con_term #smlp.Ite(ant_term, con_term, smlp.Var('SMLP_UNDEFINED'))
                else:
                    rules_dict[resp] = smlp.Ite(ant_term, con_term, rules_dict[resp])
        #print('rules_dict', rules_dict)
        return rules_dict
    
    def tree_model_to_term(self, tree_model, algo, feat_names, resp_names):
        if algo in ['dt_caret', 'dt_sklearn']:
            tree_estimators = [tree_model]
        elif algo in ['rf_caret', 'rf_sklearn', 'et_caret', 'et_sklearn']:
            tree_estimators = tree_model.estimators_
        else:
            raise Exception('Model trained using algorithm ' + str(algo) + ' is currently not supported in smlp_opt')
        rules = self.trees_to_rules(tree_estimators, feat_names, resp_names, None, False, None) # rules_filename
        #print('------- rules ---------\n', rules); 
        tree_term_dict_dict = {} 
        for i, tree_rules in enumerate(rules):
            #print('====== tree_rules ======\n', tree_rules)
            tree_term_dict = self.rules_to_term(tree_rules); #print('tree term_dict', tree_term_dict); 
            assert list(tree_term_dict.keys()) == resp_names
            tree_term_dict_dict['tree_'+str(i)] = tree_term_dict
        #print('**********tree_term_dict_dict\n', tree_term_dict_dict)
        
        number_of_trees = len(rules)
        tree_model_term_dict = {}
        for j, tree_rules in enumerate(rules):
            for resp_name in resp_names:
                if j == 0:
                    tree_model_term_dict[resp_name] = tree_term_dict_dict['tree_'+str(j)][resp_name]
                else:
                    #tree_model_term_dict[resp_name] = smlp.ADD(tree_model_term_dict[resp_name], tree_term_dict_dict['tree_'+str(j)][resp_name])
                    #tree_model_term_dict[resp_name] = tree_model_term_dict[resp_name] + tree_term_dict_dict['tree_'+str(j)][resp_name]
                    tree_model_term_dict[resp_name] = self.instSmlpTerms.smlp_add(tree_model_term_dict[resp_name], tree_term_dict_dict['tree_'+str(j)][resp_name])
                    if j == number_of_trees - 1: # the last tree -- compute the mean by dividing the sum on number_of_trees
                        #tree_model_term_dict[resp_name] = smlp.Div(tree_model_term_dict[resp_name], smlp.Cnst(int(number_of_trees)))
                        tree_model_term_dict[resp_name] = self.instSmlpTerms.smlp_mult(smlp.Cnst(smlp.Q(1) / smlp.Q(int(number_of_trees))), tree_model_term_dict[resp_name])
                    #print('tree_model_term_dict', tree_model_term_dict)
        
        return tree_model_term_dict

    def tree_models_to_term(self, model, algo, feat_names, resp_names):
        #print('tree_models_to_term: feat_names', feat_names, 'resp_names', resp_names)
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
        return tree_model_term_dict

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


    # determine the model type -- sequential vs functional
    def _get_nn_keras_model_type(self, model):
        #print('keras model', model, type(model))
        #print('keras terms', keras)
        if _keras_is_sequential(model):
            model_type = 'sequential'
        elif _keras_is_functional(model):
            model_type = 'functional'
        else:
            raise Exception('Unsupported Keras NN type (neither sequential nor functional)')
        return model_type
    
    # Create SMLP terms from polynomial model. Returns a dictionary with response names from model_resp_names as keys
    # and respective model terms as the values.
    def nn_keras_model_to_term(self, model, model_feat_names, model_resp_names, feat_names, resp_names):
        #print('nn_keras_model_to_term: model_feat_names', model_feat_names, 'model_resp_names', model_resp_names)
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
            if layer.get_config()['name'] in resp_names:
                assert model_type == 'functional'
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
class ScalerTerms:
    def __init__(self):
        self._SCALED_TERMS_SUFFIX = '_scaled'
    
    # Convention to generate scaled feature, response objective's names from respective original names.
    # Models are generated using the original feature and response names, independently from whether 
    # the features and/or responses were scaled prior to traing the models. However, then building model
    # terms from models, we use scaled feature and/or response names as inputs and outputs of the model
    # term, respectively, in case features and/or responses were scaled before trainng the model. In case
    # optimization objectives are scaled prior to optimization procedure, the results will be un-scaled 
    # before reproting to user so that user can see results in riginal scale (and in scaled form as well).
    def _scaled_name(self, name):
        return name+self._SCALED_TERMS_SUFFIX
    
    def _unscaled_name(self, name):
        if name.endswith(self._SCALED_TERMS_SUFFIX):
            return name[:-len(self._SCALED_TERMS_SUFFIX)]
        else:
            return name
    
    # Computes term x_scaled for column x using expression x_scaled = 1 / (max(x) - min(x)) * (x - min(x).
    # Argument orig_feat_name is name for column x, argument scaled_feat_name is the name of scaled column 
    # x_scaled obtained from x using min_max scaler to range [0, 1] (which is the same as normalizin x),
    # orig_min stands for min(x) and orig_max stands for max(x). Note that 1 / (max(x) - min(x)) is a
    # rational constant, it is defined to smlp instance as a fraction (thus there is no loss of precision).
    def feature_scaler_to_term(self, orig_feat_name, scaled_feat_name, orig_min, orig_max): 
        #print('feature_scaler_to_term', 'orig_min', orig_min, type(orig_min), 'orig_max', orig_max, type(orig_max))
        return smlp.Cnst(smlp.Q(1) / smlp.Q(orig_max - orig_min)) * (smlp.Var(orig_feat_name) - smlp.Cnst(orig_min))
    
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
        unscaled_term = (smlp.Var(scaled_feat_name) * smlp.Cnst(orig_max - orig_min)) + smlp.Cnst(orig_min)
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
        return (smlp.Cnst(const) * smlp.Cnst(orig_max - orig_min)) + smlp.Cnst(orig_min)


class ModelTerms(SmlpTerms):
    def __init__(self):
        self._scalerTermsInst = ScalerTerms()
        self._treeTermsInst = TreeTerms()
        self._polyTermsInst = PolyTerms()
        self._smlpTermsInst = SmlpTerms()
        self._nnKerasTermsInst = NNKerasTerms()
        
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
        
        self._SPEC_DOMAIN_RANGE_TAG = 'range'
        self._SPEC_DOMAIN_INTERVAL_TAG = 'interval'
        
        
    # set logger from a caller script
    def set_logger(self, logger):
        self._smlp_terms_logger = logger
    
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
    
    # file to dump tree model converted to SMLP term
    @property
    def smlp_model_term_file(self):
        assert self.model_file_prefix is not None
        return self.model_file_prefix + '_smlp_model_term.json'
    
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
    # computing a dictionary of model terms from the model / dictionary of mdels is called. 
    def _compute_pure_model_terms(self, algo, model, model_feat_names, model_resp_names, feat_names, resp_names):
        assert not isinstance(model, dict)
        if algo == 'nn_keras': 
            model_term_dict = self._nnKerasTermsInst.nn_keras_model_to_term(model, model_feat_names, 
                model_resp_names, feat_names, resp_names) 
        elif algo == 'poly_sklearn':
            model_term_dict = self._polyTermsInst.poly_model_to_term(model_feat_names, model_resp_names, 
                model[0].coef_, model[1].powers_, False, None)
        elif algo in ['dt_sklearn', 'dt_caret']:
            model_term_dict = self._treeTermsInst.tree_models_to_term(model, algo, model_feat_names, model_resp_names)
        elif algo in ['rf_sklearn', 'rf_caret', 'et_sklearn', 'et_caret']:
            model_term_dict = self._treeTermsInst.tree_models_to_term(model, algo, model_feat_names, model_resp_names)
        else:
            raise Exception('Algo ' + str(algo) + ' is currently not suported in model exploration modes')
        #print('model_term_dict', model_term_dict)
        with open(self.smlp_model_term_file, 'w') as f:
            json.dump(str(model_term_dict), f, indent='\t', cls=np_JSONEncoder)
        return model_term_dict
    
    
    # This function takes an ML model (the argument 'model) as well as features and responses scaling info
    # as inputs and for each response in the model generates a term that encodes the "pure" model constraints
    # as well as constraints relating to scaling of features and/or responses if they were scaled prior to
    # the model training. By "pure" moel we mean a model trained with the original feature and response names
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
        model_feat_names = [self._scalerTermsInst._scaled_name(feat) for feat in feat_names] if feat_were_scaled else feat_names
        model_resp_names = [self._scalerTermsInst._scaled_name(resp) for resp in resp_names] if resp_were_scaled else resp_names
        #print('adding model terms: model_feat_names', model_feat_names, 'model_resp_names', model_resp_names)

        model_term_dict = self._compute_pure_model_terms(algo, model, model_feat_names, model_resp_names, 
            feat_names, resp_names)
        #print('model_term_dict', model_term_dict.keys())
        
        model_full_term_dict = model_term_dict; #print('model_full_term_dict', model_full_term_dict)
        
        # compute features scaling term (skipped when it is identity);
        # substitute them instead of scaled feature variables in the model
        # terms (where the latter variables are inputs to the model)
        if feat_were_scaled:
            #print('adding feature scaler terms')
            #scaled_feat_names = [self._scalerTermsInst._scaled_name(feat)for feat in feat_names]
            feature_scaler_terms_dict = self._scalerTermsInst.feature_scaler_terms(data_bounds, feat_names)
            #print('feature_scaler_terms_dict', feature_scaler_terms_dict)

            for resp_name, model_term in model_term_dict.items():
                #print('model term before', model_term)
                for feat_name, feat_term in feature_scaler_terms_dict.items():
                    #print('feat_name', feat_name)
                    model_term = smlp.subst(model_term, {feat_name: feat_term})
                #print('model term after', model_term)
                model_term_dict[resp_name] = model_term
            #print('model_term_dict', model_term_dict)
            model_full_term_dict = model_term_dict
            
        # compute responses in original scale from scaled responses that are
        # the outputs of the modes, compose models with unscaled responses
        if resp_were_scaled:
            responses_unscaler_terms_dict = self._scalerTermsInst.feature_unscaler_terms(data_bounds, resp_names)
            
            # substitute scaled response variables with scaled response terms (the model outputs)
            # in original response terms within responses_unscaler_terms_dict
            for resp_name, resp_term in responses_unscaler_terms_dict.items():
                #print('resp_name', resp_name, resp_term)
                responses_unscaler_terms_dict[resp_name] = smlp.subst(resp_term, 
                    {self._scalerTermsInst._scaled_name(resp_name): model_term_dict[self._scalerTermsInst._scaled_name(resp_name)]})
            #print('responses_unscaler_terms_dict full model', responses_unscaler_terms_dict)
            model_full_term_dict = responses_unscaler_terms_dict
            
        #print('model_full_term_dict', model_full_term_dict);
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
        
        if isinstance(model_or_model_dict, dict):
            models_full_terms_dict = {}
            for i, resp_name in enumerate(model_or_model_dict.keys()):
                curr_feat_names = model_features_dict[resp_name]
                curr_model_full_term_dict = self._compute_model_terms_dict(algo, model_or_model_dict[resp_name], 
                    feat_names, [resp_name], data_bounds, data_scaler, scale_features, scale_responses)
                models_full_terms_dict = models_full_terms_dict | curr_model_full_term_dict
        else:
            models_full_terms_dict = self._compute_model_terms_dict(algo, model_or_model_dict, 
                feat_names, resp_names, data_bounds, data_scaler, scale_features, scale_responses)
        #print('models_full_terms_dict', models_full_terms_dict)
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
        orig_objv_terms_dict = dict([(objv_name, self._smlpTermsInst.ast_expr_to_term(objv_expr)) \
            for objv_name, objv_expr in zip(objv_names, objv_exprs)])
        #print('orig_objv_terms_dict', orig_objv_terms_dict)
        if scale_objv:
            scaled_objv_terms_dict = self._scalerTermsInst.feature_scaler_terms(objv_bounds, objv_names)
            #print('scaled_objv_terms_dict', scaled_objv_terms_dict)
            objv_terms_dict = {}
            for i, (k, v) in enumerate(scaled_objv_terms_dict.items()):
                #print('k', k, 'v', v, type(v)); 
                x = list(orig_objv_terms_dict.keys())[i]; 
                #print('x', x); print('arg', orig_objv_terms_dict[x])
                objv_terms_dict[k] = smlp.subst(v, {x: orig_objv_terms_dict[x]})
            #objv_terms_dict = scaled_objv_terms_dict
        else:
            objv_terms_dict = orig_objv_terms_dict
            scaled_objv_terms_dict = None
        
        if scaled_objv_terms_dict is not None:
            assert list(scaled_objv_terms_dict.keys()) == [self._scalerTermsInst._scaled_name(objv_name) 
                for objv_name in objv_names]
        #print('objv_terms_dict', objv_terms_dict)
        return objv_terms_dict, orig_objv_terms_dict, scaled_objv_terms_dict
    
    # Compute stability region theta; used also in generating lemmas during search for a stable solution. 
    # cex is assignement of values to knobs. Even if cex contains assignements to inputs, such assignements
    # are ignored as only variables which occur as keys in theta_radii_dict are used for building theta.

    
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
            
        theta_form = smlp.true
        #print('radii_dict', radii_dict)
        radii_dict_local = radii_dict.copy() 
        knobs = radii_dict_local.keys(); #print('knobs', knobs); print('cex', cex); print('delta', delta_dict)
        
        # use inputs in theta computation, by setting radii to 0, and use delta if specified (not None)
        if not universal and delta_rel is not None:
            for cex_var in cex.keys():
                if cex_var not in knobs:
                    radii_dict_local[cex_var] = {'rad-abs':0, 'rad-rel': None} # delta
        
        for var,radii in radii_dict_local.items():
            var_term = smlp.Var(var)
            # either rad-abs or rad-rel must be None -- for each var wr declare only one of these
            if radii['rad-abs'] is not None:
                rad = radii['rad-abs']; #print('rad', rad); 
                if delta_rel is not None: # we are generating a lemma
                    rad = rad * (1 + delta_rel) + delta_abs 
                rad_term = smlp.Cnst(rad)
            elif radii['rad-rel'] is not None:
                rad = radii['rad-rel']; #print('rad', rad)
                if delta_rel is not None: # we are generating a lemma
                    rad = rad * (1 + delta_rel) + delta_abs
                rad_term = smlp.Cnst(rad)
                
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
        eta_grid_form = smlp.true
        eta_grids_dict = self._specInst.get_spec_eta_grids_dict; #print('eta_grids_dict', eta_grids_dict)
        for var,grid in eta_grids_dict.items():
            eta_grid_disj = smlp.false
            var_term = smlp.Var(var)
            for gv in grid: # iterate over grid values
                if eta_grid_disj == smlp.false:
                    eta_grid_disj = var_term == smlp.Cnst(gv)
                else:
                    eta_grid_disj = self.smlp_or(eta_grid_disj, var_term == smlp.Cnst(gv))
            if eta_grid_form == smlp.true:
                eta_grid_form = eta_grid_disj
            else:
                eta_grid_form = self.smlp_and(eta_grid_form, eta_grid_disj)
        #print('eta_grid_form', eta_grid_form); 
        return eta_grid_form
                

    # Compute formulae alpha, beta, eta from respective expression string.
    def compute_input_ranges_formula_alpha(self, model_inputs):
        alpha_form = smlp.true
        alpha_dict = self._specInst.get_spec_alpha_bounds_dict; #print('alpha_dict', alpha_dict)
        for v,b in alpha_dict.items():
            if v not in model_inputs:
                continue
            mn = b['min']
            mx = b['max']
            #print('mn', mn, 'mx', mx)
            if mn is not None and mx is not None:
                if self._declare_domain_interface_only:
                    rng = self.smlp_and(smlp.Var(v) >= smlp.Cnst(mn), smlp.Var(v) <= smlp.Cnst(mx))
                    alpha_form = self.smlp_and(alpha_form, rng)
            elif mn is not None:
                rng = smlp.Var(v) >= smlp.Cnst(mn)
                alpha_form = self.smlp_and(alpha_form, rng)
            elif mx is not None:
                rng = smlp.Var(v) <= smlp.Cnst(mx)
                alpha_form = self.smlp_and(alpha_form, rng)
            else:
                assert False
        return alpha_form
    
    def compute_input_ranges_formula_alpha_eta(self, alpha_vs_eta, model_inputs):
        alpha_or_eta_form = smlp.true
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
                    rng = self.smlp_and(smlp.Var(v) >= smlp.Cnst(mn), smlp.Var(v) <= smlp.Cnst(mx))
                    alpha_or_eta_form = self.smlp_and(alpha_or_eta_form, rng)
            elif mn is not None:
                rng = smlp.Var(v) >= smlp.Cnst(mn)
                alpha_or_eta_form = self.smlp_and(alpha_or_eta_form, rng)
            elif mx is not None:
                rng = smlp.Var(v) <= smlp.Cnst(mx)
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
        #alph_form = smlp.true
        if alph_expr is None:
            alpha_expr = self._specInst.get_spec_alpha_global_expr
        if alph_expr is None:
            return smlp.true
        else:
            alph_expr_vars = get_expression_variables(alph_expr)
            dont_care_vars = list_subtraction_set(alph_expr_vars, model_inputs)
            if len(dont_care_vars) > 0:
                raise Exception('Variables ' + str(dont_care_vars) + 
                    ' in input constraints (alpha) are not part of the model')
            alph_glob = self._smlpTermsInst.ast_expr_to_term(alph_expr)
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
            return smlp.true
        else:
            beta_expr_vars = get_expression_variables(beta_expr)
            dont_care_vars = list_subtraction_set(beta_expr_vars, model_inps_outps)
            if len(dont_care_vars) > 0:
                raise Exception('Variables ' + str(dont_care_vars) + 
                    ' in optimization constraints (beta) are not part of the model')
            return self._smlpTermsInst.ast_expr_to_term(beta_expr)
    
    def compute_eta_formula(self, eta_expr, model_inputs):
        if eta_expr is None:
            return smlp.true
        else:
            # eta_expr can only contain knobs (control inputs), not free inputs or outputs (responses)
            eta_expr_vars = get_expression_variables(eta_expr)
            dont_care_vars = list_subtraction_set(eta_expr_vars, model_inputs)
            if len(dont_care_vars) > 0:
                raise Exception('Variables ' + str(dont_care_vars) + 
                    ' in knob constraints (eta) are not part of the model')
            return self._smlpTermsInst.ast_expr_to_term(eta_expr)

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
                var_component = smlp.component(smlp.Real, grid=list(range(interval[0], interval[1]+1)))
                assert False
            if self._declare_domain_interface_only or interval_has_none:
                var_component = smlp.component(smlp.Integer)
            else:
                var_component = smlp.component(smlp.Integer, 
                    interval=spec_domain_dict[var][self._SPEC_DOMAIN_INTERVAL_TAG]) #self._specInst.get_spec_interval_tag
        elif spec_domain_dict[var][self._SPEC_DOMAIN_RANGE_TAG] == self._specInst.get_spec_real_tag: #self._specInst.get_spec_range_tag
            if self._declare_domain_interface_only or interval_has_none:
                var_component = smlp.component(smlp.Real)
            else:
                var_component = smlp.component(smlp.Real, 
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
        alph_ranges = self.compute_input_ranges_formula_alpha_eta('alpha', feat_names) 
        alph_global = self.compute_global_alpha_formula(alph_expr, feat_names)
        alpha = self._smlpTermsInst.smlp_and(alph_ranges, alph_global)
        beta = self.compute_beta_formula(beta_expr, feat_names+resp_names)
        eta_ranges = self.compute_input_ranges_formula_alpha_eta('eta', feat_names)
        eta_grids = self.compute_grid_range_formulae_eta()
        eta_global = self.compute_eta_formula(eta_expr, feat_names); #print('eta_global', eta_global)
        eta = self._smlpTermsInst.smlp_and_multi([eta_ranges, eta_grids, eta_global]); #print('eta', eta)
        
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
        interface_consistent = True # TMP !!!!!!!!!!!!!!!!!!!!
        #interface_consistent = self.check_alpha_eta_consistency(domain_features, None, alpha, eta, 'ALL')
        if not interface_consistent:
            return None, None, None, eta, alpha, beta, False, False
        
        # now define solver donain that includes input, knob and output dec;arations from spec file.
        for var in resp_names:
            domain_dict[var] = self.var_domain(var, spec_domain_dict)
        
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
            system_term_dict = dict([(resp_name, self._smlpTermsInst.ast_expr_to_term(resp_expr)) \
                for resp_name, resp_expr in syst_expr_dict.items()]); #print('system_term_dict', system_term_dict); 
            self._smlp_terms_logger.info('System terms dictionary: ' + str(system_term_dict))
            self._smlp_terms_logger.info('Building system terms: End')
        else:
            system_term_dict = None
        
        self._smlp_terms_logger.info('Building model terms: Start')
        # model terms with terms for scaling inputs and / or unscaling responses are all composed to 
        # build model terms with inputs and outputs in the original scale. 
        if algo == 'system':
            if syst_expr_dict is None:
                raise Exception('System must be specified when model training algorithm is "system"')
            model_full_term_dict = system_term_dict
        else:
            model_full_term_dict = self.compute_models_terms_dict(algo, model, 
                model_features_dict, feat_names, resp_names, data_bounds, data_scaler, scale_feat, scale_resp)
        self._smlp_terms_logger.info('Building model terms: End')
        
        model_consistent = self.check_alpha_eta_consistency(domain, model_full_term_dict, alpha, eta, 'ALL')
        if not model_consistent:
            return domain, system_term_dict, model_full_term_dict, eta, alpha, beta, True, False
        
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
        
        # let solver know definition of responses (the model's function)
        if model_full_term_dict is not None:
            for resp_name, resp_term in model_full_term_dict.items():
                #base_solver.add(smlp.Var(resp_name) == resp_term)
                base_solver.add(op.eq(smlp.Var(resp_name), resp_term))
        return base_solver
    
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
        solver = self.create_model_exploration_instance_from_smlp_components(
            domain, model_full_term_dict, False, solver_logic)
        solver.add(alpha); #print('alpha', alpha)
        solver.add(eta); #print('eta', eta)
        res = solver.check(); #print('res', res)
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

