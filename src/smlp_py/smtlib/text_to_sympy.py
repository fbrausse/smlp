import re

import gmpy2
import z3
from pysmt import *
from sympy.logic.boolalg import And, Or, Not
from pysmt.shortcuts import Symbol, And, Or, Not, Implies, Iff, Ite, Equals, Plus, Minus, Times, Div, Pow, Bool, TRUE, \
    FALSE, Int, Real, simplify, LT, LE, GT, GE, ToReal
from pysmt.shortcuts import Or,  Equals
from pysmt.fnode import FNode
from pysmt.typing import BOOL, REAL, INT
from pysmt.rewritings import CNFizer
from pysmt.walkers import IdentityDagWalker, DagWalker
import ast
import smlp
from pysmt.smtlib.script import smtlibscript_from_formula
from io import StringIO

from typing import List, Dict, Optional, Tuple

from z3 import Tactic, Goal

pysmt_types = {
    "int": INT,
    "real": REAL,
    "bool": BOOL
}


class Equation:

    def __init__(self, variable, operator: str, scalar: float):
        self.variable = variable
        self.operator = operator
        self.scalar = scalar
        self._eq = [str(variable),operator,str(scalar)]

    def __str__(self):
        return "{0} {1} {2}".format(self.variable,self.operator,self.scalar)

    def __eq__(self, o: object) -> bool:
        return str(self) == str(o)

    def lhs(self):
        return self.variable

    def rhs(self):
        return self.scalar

    def op(self):
        return self.operator

    def __hash__(self) -> int:
        return hash(self.variable) * hash(self.operator) * int(self.scalar)

class InequalityChecker(DagWalker):
    def __init__(self, env=None):
        DagWalker.__init__(self, env=env)
        self.is_inequality = False
        self.contains_and_or = False

    def walk_and(self, formula, args, **kwargs):
        self.contains_and_or = True
        return formula

    def walk_or(self, formula, args, **kwargs):
        self.contains_and_or = True
        return formula

    def walk_le(self, formula, args, **kwargs):
        self.is_inequality = True
        return formula

    def walk_lt(self, formula, args, **kwargs):
        self.is_inequality = True
        return formula

    def walk_ge(self, formula, args, **kwargs):
        self.is_inequality = True
        return formula

    def walk_gt(self, formula, args, **kwargs):
        self.is_inequality = True
        return formula

def check_inequality(formula):
    checker = InequalityChecker()
    checker.walk(formula)
    return checker.is_inequality and not checker.contains_and_or

class TextToPysmtParser(object):
    SAT = "SAT"
    UNSAT = "UNSAT"
    types = pysmt_types
    real = Real
    true = TRUE
    false = FALSE
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TextToPysmtParser, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        self.symbols = {}
        self.inputs = []
        self.outputs = []
        self._ast_operators_map = {
            ast.Add: Plus,  # Addition
            ast.Sub: Minus,  # Subtraction
            ast.Mult: Times,  # Multiplication
            ast.Div: self._div_op,  # Division
            ast.Pow: Pow,  # Exponentiation
            ast.BitXor: Iff,  # Bitwise XOR (interpreted as logical Iff)

            ast.USub: lambda l: -l,  # Unary subtraction (negation)

            ast.Eq: Equals,  # Equality
            ast.NotEq: Not,  # Not equal
            ast.Lt: lambda l, r: l < r,  # Less than
            ast.LtE: lambda l, r: l <= r,  # Less than or equal to
            ast.Gt: lambda l, r: l > r,  # Greater than
            ast.GtE: lambda l, r: l >= r,  # Greater than or equal to

            ast.And: And,  # Logical AND
            ast.Or: Or,  # Logical OR
            ast.Not: Not,  # Logical NOT

            ast.IfExp: Ite,  # If expression
            ast.Call: And,
            'If': Ite,
            'And': And,
            'Not': Not,
            'Or': Or
        }

    def _div_op(self, left, right):
        # Ensure both operands are real numbers for division
        left = ToReal(left) if not left.is_real_constant() else left
        right = ToReal(right) if not right.is_real_constant() else right
        return Div(left, right)


    @staticmethod
    def and_(*expressions):
        return And(*expressions)

    @staticmethod
    def or_(*expressions):
        return Or(*expressions)

    @staticmethod
    def eq_(*expressions):
        return Equals(*expressions)

    @staticmethod
    def ite_(*expressions):
        return Ite(*expressions)

    @staticmethod
    def to_cnf(formula):
        cnfizer = CNFizer()
        cnf_formula = cnfizer.convert(formula)
        return cnf_formula

    @staticmethod
    def conjunction_to_disjunction(formula):
        if formula.is_and():
            negated_terms = [Not(arg) for arg in formula.args()]
            disjunction = Or(negated_terms)
            return simplify(Not(disjunction))
        else:
            raise ValueError("Input formula is not a conjunction")

    def is_comparison(self, node: FNode) -> bool:
        return node.is_le() or node.is_lt() or node.is_ge() or node.is_gt() or node.is_equals()

    def create_integer_disjunction(self, variable, values):
        variable = self.get_symbol(variable)
        if not variable:
            return None

        lower, upper = values
        value_range = range(lower, upper + 1)

        return Or(*(Equals(variable, Real(val)) for val in value_range))


    def split_disjunctions(self, formula: FNode) -> list:
        if formula.is_or():
            comparisons = [arg for arg in formula.args() if self.is_comparison(arg)]
            if len(comparisons) == len(formula.args()):
                return comparisons
        elif self.is_comparison(formula):
            return [formula]
        else:
            raise ValueError("Input formula is not a valid disjunction of comparisons")
        return []

    def opposite_comparator(self, comparator):
        # sympy only uses LE and LT
        # GE and GT are described using LE and LT and reversing the order of the symbol and number
        if comparator == "<=":
            return ">="
        elif comparator == "<":
            return ">"
        else:
            return comparator

    def decide_comparator(self, formula):
        node_type = formula.node_type()
        if node_type == 16:
            return "<="
        elif node_type == 17:
            return "<"
        elif node_type == 18:
            return "="
        else:
            return None

    def extract_coefficient(self, symbol):
        coeff = []
        # possible formats
        # 1) x-5
        # 2) a*x - 5
        for arg in symbol.args():
            if arg.is_constant():
                coeff.insert(0, arg)
            elif arg.is_symbol():
                coeff.append(arg)
            else:
                pass

        return coeff

    def extract_components(self, comparison: FNode, need_simplification=False):
        if need_simplification:
            smtlib = self.extract_smtlib(comparison)
            comparison = self.handle_ite_formula(smtlib, handle_ite=False)

        left = comparison.arg(0)
        right = comparison.arg(1)

        if not right.is_constant() and not left.is_constant():
            raise ValueError("The right-hand side of the formula must be a constant")

        comparator = self.decide_comparator(comparison)

        terms_subformula = left if right.is_constant() else right

        terms = []

        def traverse(node):
            if node.is_times():
                coeff, var = node.args()
                if coeff.is_constant() and var.is_symbol():
                    terms.append((float(coeff.constant_value()), var))
                elif var.is_constant() and coeff.is_symbol():
                    terms.append((float(var.constant_value()), coeff))
                else:
                    raise ValueError("Invalid term structure in linear inequality")
            elif node.is_plus():
                for arg in node.args():
                    traverse(arg)
            elif node.is_minus():
                left, right = node.args()
                traverse(left)
                if right.is_times():
                    coeff, var = right.args()
                    if coeff.is_constant() and var.is_symbol():
                        terms.append((-float(coeff.constant_value()), var))
                    elif var.is_constant() and coeff.is_symbol():
                        terms.append((-float(var.constant_value()), coeff))
                    else:
                        raise ValueError("Invalid term structure in linear inequality")
                else:
                    raise ValueError("Invalid term structure in linear inequality")
            elif node.is_symbol():
                terms.append((1.0, node))
            elif node.is_constant():
                terms.append((node.constant_value(), Real(0)))
            else:
                raise ValueError("Unsupported node type in linear inequality")

        traverse(terms_subformula)

        if right.is_constant():
            scalar = float(right.constant_value())
            return terms, comparator, scalar
        else:
            scalar = float(left.constant_value())
            return terms, self.opposite_comparator(comparator), scalar

    def process_formula(self, formula: FNode):
        components = []
        if formula.is_and():
            for arg in formula.args():
                components.extend(self.process_formula(arg))
        elif formula.is_or():
            print("Disjunction found, storing components:")
            for arg in formula.args():
                if arg.is_and():
                    components.extend(self.process_formula(arg))
                else:
                    components.append(arg)
        elif self.is_comparison(formula):
            components.append(formula)
        else:
            print("Other formula type encountered.")

        return components


    def propagate_negation(self, formula):
        """
        Apply negation to a formula and propagate the negation inside without leaving any negations in the formula.
        """
        formula = self.simplify(formula)
        if formula.is_not():
            return self.propagate_negation(formula.arg(0))  # Remove double negation if exists

        elif formula.is_and():
            # Apply De Morgan's law: not (A and B) -> (not A) or (not B)
            return Or([self.propagate_negation(Not(arg)) for arg in formula.args()])

        elif formula.is_or():
            # Apply De Morgan's law: not (A or B) -> (not A) and (not B)
            return And([self.propagate_negation(Not(arg)) for arg in formula.args()])

        elif formula.is_equals():
            # not (A = B) -> A != B
            A, B = formula.args()
            return And(LT(A, B), LT(B, A))

        elif formula.is_lt():
            # not (A < B) -> A >= B
            A, B = formula.args()
            return LE(B,A)

        elif formula.is_le():
            # not (A <= B) -> A > B
            A, B = formula.args()
            return LT(B, A)

        elif formula.is_plus() or formula.is_times():
            # Propagate negation inside arithmetic operations
            return formula

        elif formula.is_symbol() or formula.is_constant():
            # Apply negation directly to literals
            return Not(formula)

        else:
            raise NotImplementedError(f"Negation propagation not implemented for formula type: {formula}")

    def simplify(self, expression):
        return simplify(expression)

    def cast_number(self, symbol_type, number):
        if symbol_type == REAL:
            return Real(number)
        elif symbol_type == INT:
            return Int(number)

    def init_variables(self, symbols: List[Tuple[str, str, bool]]) -> None:
        for input_var in symbols:
            name, type, is_input = input_var
            unscaled_name = f"{name}_unscaled"
            # TODO: i replaced the type variable with real, make sure that's ok
            self.add_symbol(name, 'real', is_input=is_input, nn_type=type)
            self.add_symbol(unscaled_name, 'real', is_input=is_input, nn_type=type)

    def add_symbol(self, name, symbol_type, is_input=True, nn_type='real'):
        assert symbol_type.lower() in pysmt_types.keys()
        self.symbols[name] = Symbol(name, pysmt_types[symbol_type])

        if name.find("_unscaled") == -1:
            store = self.inputs if is_input else self.outputs
            store.append((name, nn_type))

    def get_symbol(self, name):
        assert name in self.symbols.keys()
        return self.symbols[name]

    def remove_first_and_last_line(self, text):
        # Split the text into a list of lines
        lines = text.split('\n')

        # Remove the first and last lines
        if len(lines) > 1:
            lines = lines[1:-2]
        else:
            # If there's only one line or no line, return an empty string
            lines = []

        # Join the remaining lines back into a single string
        return '\n'.join(lines)

    def extract_smtlib(self, formula):
        script = smtlibscript_from_formula(formula)
        outstream = StringIO()
        script.serialize(outstream)
        output = outstream.getvalue()
        return self.remove_first_and_last_line(output)

    def handle_ite_formula(self, formula, is_form2=False, handle_ite=True):
        # smlp_str = self.extract_smtlib(formula) if not isinstance(formula, str) else formula
        # smlp_str = f"""
        #                         (declare-fun y1 () Real)
        #                         (declare-fun y2 () Real)
        #                         (assert {formula})
        #                         """ if not isinstance(formula, str) else formula
        flag=False
        if is_form2:
            smlp_str = f"""
                        (declare-fun y1 () Real)
                        (declare-fun y2 () Real)
                        (assert {formula})
                        """
        elif isinstance(formula, str):
            smlp_str = formula
        else:
            smlp_str = self.extract_smtlib(formula)
            flag=False

        smlp_parsed = z3.parse_smt2_string(smlp_str)
        if flag:
            # Apply the tactic to the formula
            goal = Goal()
            goal.add(smlp_parsed)
            t = Tactic('tseitin-cnf')
            smlp_parsed = t(goal)[0]

        smlp_simplified = z3.simplify(smlp_parsed[0])
        ex = self.parse(str(smlp_simplified).replace('\n',''))
        if ex.is_not():
            ex = self.propagate_negation(ex)
        # ex = parser.replace_constants_with_floats_and_evaluate(ex)
        if handle_ite:
            marabou_formula = self.convert_ite_to_conjunctions_disjunctions(ex)
        else:
            marabou_formula = ex
        return marabou_formula

    def replace_constants_with_floats_and_evaluate(self, formula: FNode) -> FNode:
        def traverse(node: FNode) -> FNode:
            if node.is_plus():
                left, right = node.args()
                new_left = traverse(left)
                new_right = traverse(right)
                if new_left.is_constant() and new_right.is_constant():
                    return Real(new_left.constant_value() + new_right.constant_value())
                return Plus(new_left, new_right)
            elif node.is_minus():
                left, right = node.args()
                new_left = traverse(left)
                new_right = traverse(right)
                if new_left.is_constant() and new_right.is_constant():
                    return Real(new_left.constant_value() - new_right.constant_value())
                return Minus(new_left, new_right)
            elif node.is_times():
                left, right = node.args()
                new_left = traverse(left)
                new_right = traverse(right)
                if new_left.is_constant() and new_right.is_constant():
                    return Real(new_left.constant_value() * new_right.constant_value())
                return Times(new_left, new_right)
            elif node.is_div():
                left, right = node.args()
                new_left = traverse(left)
                new_right = traverse(right)
                if new_left.is_constant() and new_right.is_constant():
                    return Real(new_left.constant_value() / new_right.constant_value())
                return Div(new_left, new_right)
            elif node.is_ite():
                condition, true_branch, false_branch = node.args()
                new_condition = traverse(condition)
                new_true_branch = traverse(true_branch)
                new_false_branch = traverse(false_branch)
                return Ite(new_condition, new_true_branch, new_false_branch)
            elif node.is_le():
                left, right = node.args()
                new_left = traverse(left)
                new_right = traverse(right)
                return LE(new_left, new_right)
            elif node.is_lt():
                left, right = node.args()
                new_left = traverse(left)
                new_right = traverse(right)
                return LT(new_left, new_right)
            elif node.is_and():
                new_args = [traverse(arg) for arg in node.args()]
                return And(new_args)
            elif node.is_or():
                new_args = [traverse(arg) for arg in node.args()]
                return Or(new_args)
            elif node.is_constant():
                if isinstance(node.constant_value(), gmpy2.mpq):
                    return Real(float(node.constant_value()))
                elif node.is_int_constant():
                    return Real(float(node.constant_value()))
                elif node.is_real_constant():
                    return Real(node.constant_value())
            else:
                return node

        return traverse(formula)

    def parse(self, expr):
        assert isinstance(expr, str)
        symbol_list = self.symbols

        def eval_(node):
            if isinstance(node, ast.Num):
                # return Real(node.n) if isinstance(node.n, float) else Int(node.n)
                return Real(float(node.n))
            elif isinstance(node, ast.BinOp):
                left = eval_(node.left)
                right = eval_(node.right)
                if left.is_constant() and right.is_constant():
                    if isinstance(node.op, ast.Mult):
                        return Real(float(left.constant_value() * right.constant_value()))
                    elif isinstance(node.op, ast.Div):
                        return Real(float(left.constant_value() / right.constant_value()))
                return self._ast_operators_map[type(node.op)](left, right)
            elif isinstance(node, ast.UnaryOp):
                operand = eval_(node.operand)
                if operand.is_constant() and isinstance(node.op, ast.USub):
                    return Real(-operand.constant_value())
                return self._ast_operators_map[type(node.op)](operand)
            elif isinstance(node, ast.Name):
                return symbol_list[node.id]
            elif isinstance(node, ast.BoolOp):
                res_boolop = self._ast_operators_map[type(node.op)](eval_(node.values[0]), eval_(node.values[1]))
                for value in node.values[2:]:
                    res_boolop = self._ast_operators_map[type(node.op)](res_boolop, eval_(value))
                return res_boolop
            elif isinstance(node, ast.Compare):
                left = eval_(node.left)
                first_comparator = eval_(node.comparators[0])
                result = self._ast_operators_map[type(node.ops[0])](left, first_comparator)
                for op, comparator in zip(node.ops[1:], node.comparators[1:]):
                    left = eval_(comparator)
                    result = And(result, self._ast_operators_map[type(op)](left, eval_(comparator)))
                return result
            elif isinstance(node, ast.Call):
                func = node.func.id
                args = [eval_(arg) for arg in node.args]
                if func in self._ast_operators_map:
                    return self._ast_operators_map[func](*args)
                else:
                    raise ValueError(f"Unsupported function call: {func}")

            elif isinstance(node, ast.IfExp):
                return self._ast_operators_map[ast.IfExp](eval_(node.test), eval_(node.body), eval_(node.orelse))
            elif isinstance(node, ast.Constant):
                if node.value is True:
                    return TRUE()
                elif node.value is False:
                    return FALSE()
                elif isinstance(node.value, int):
                    return Int(node.value)
                elif isinstance(node.value, float):
                    return Real(node.value)
                else:
                    return node.value
            else:
                raise TypeError(f"Unexpected node type {type(node)}")

        return eval_(ast.parse(expr, mode='eval').body)

    def convert_ite_to_conjunctions_disjunctions(self, formula):
        def traverse(node):
            if node.is_ite():
                condition, true_branch, false_branch = node.args()
                return Or(
                    And(traverse(condition), traverse(true_branch)),
                    And(traverse(Not(condition)), traverse(false_branch))
                )
            elif node.is_and():
                new_args = [traverse(arg) for arg in node.args()]
                return And(new_args)
            elif node.is_or():
                new_args = [traverse(arg) for arg in node.args()]
                return Or(new_args)
            elif node.is_not():
                return self.propagate_negation(node)
            else:
                return node

        return traverse(formula)

if __name__ == "__main__":

    parser = TextToPysmtParser()
    parser.add_symbol('x1', 'int')
    parser.add_symbol('x2', 'real')
    parser.add_symbol('p2', 'real')
    parser.add_symbol('y1', 'real')
    parser.add_symbol('y2', 'real')

    # Example usage
    y1 = parser.get_symbol('y1')
    y2 = parser.get_symbol('y1')
    # Original formula: not(y1 >= 4.0 and y2 >= 8.0)
    original_formula = Not(And(LE(Real(4.0), y1), LE(Real(8.0), y2)))

    negated_formula = parser.propagate_negation(original_formula)

    print(f"Original formula: {original_formula}")
    print(f"Negated formula with propagated negation: {negated_formula}")

    def separate_conjunctions_and_disjunctions(formula):
        conjunctions = []
        disjunctions = []

        def traverse(node, source=None):
            if node.is_and():
                # conjunctions.extend(node.args())
                for arg in node.args():
                    traverse(arg, conjunctions)
            elif node.is_or():
                disjunctions.extend(node.args())
                # Or(*[recursively_convert_ite(arg) for arg in term.args()])
                # for arg in node.args():
                #     traverse(arg, disjunctions)
            elif node.is_le() or node.is_lt() or node.is_equals():
                source.append(node)
                source.append(node)
            else:
                # Leaf nodes (symbols, literals, etc.) are not conjunctions or disjunctions
                pass

        traverse(formula)
        return conjunctions, disjunctions


    x = Symbol('x', REAL)
    y = Symbol('y', REAL)

    # formula = And(LT(x, Real(10.0)), Or(LT(y, Real(10.0)), LT(y, Real(10.0))))
    # formula = And(LT(x, Real(10.0)), LT(y, Real(10.0)))
    formula = ((-1 <= y) & (0.0 <= x) & (y <= 1) & (x <= 10.0))
    conjunctions, disjunctions = separate_conjunctions_and_disjunctions(formula)

    # Define symbols

    # Create a formula
    formula = And(LT(x, Real(10.0)), Or(LT(y, Real(10.0)), LT(y, Real(10.0))))

    # Apply external negation
    negated_formula = Not(formula)

    # Simplify the formula
    simplified_formula = simplify(negated_formula)

    print(simplified_formula)


    formula = parser.parse('(y1+y2)/2')
    # formula = parser.parse('p2<5.0 and x1==10 and x2<12.0')
    x = Symbol('x', REAL)
    y = Symbol('y', REAL)
    a = Symbol('a', INT)
    b = Symbol('b', INT)

    form=parser.and_((x <= Real(5.0)),  # x < 5.0
                Equals(y, Real(10.0)),
                Equals(a, Int(3)),  # a < 3
                Equals(b, Int(4)))  # b = 4)
    print(formula)
    print(form)
    ts = parser.conjunction_to_disjunction(form)
    print(ts)

    x2 = Symbol('x2', REAL)
    x3 = Symbol('x3', REAL)

    disjunction = Or(LT(x2, Real(12.0)), GE(x3, Real(5.0)))

    comparisons_array = parser.split_disjunctions(disjunction)

    for comparison in comparisons_array:
        print(comparison)

    x = Symbol('x', REAL)
    y = Symbol('y', REAL)
    z = Symbol('z', REAL)

    # Example formula with conjunctions and disjunctions
    # formula = And(Or(LT(x, Real(10.0)), GT(y, Real(5.0))), LE(z, Real(3.0)))
    formula = And(LT(x, Real(10.0)), GE(y, Real(4.0)))
    t = parser.to_cnf(formula)


    def frozenset_to_formula(cnf):
        clauses = []
        for clause in cnf:
            literals = []
            for literal in clause:
                if literal.is_not():
                    literals.append(Not(literal.arg(0)))
                else:
                    literals.append(literal)
            clauses.append(Or(literals))
        return And(clauses)


    # Reconstruct the formula
    t = frozenset_to_formula(t)
    t = parser.simplify(t)
    # Process the formula
    res = parser.process_formula(formula)
    print(res)