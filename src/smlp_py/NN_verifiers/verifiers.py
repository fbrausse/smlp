import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Optional, Tuple

from maraboupy import Marabou
from maraboupy import MarabouCore
from maraboupy import MarabouUtils
import tensorflow as tf
from pysmt.shortcuts import Symbol, And, Not, Or, Implies, simplify, LT, Real, Times, Minus, Plus, Equals, Int, ToReal
from pysmt.typing import BOOL, REAL, INT
import numpy as np
from maraboupy.MarabouPythonic import *
from pysmt.walkers import IdentityDagWalker
from fractions import Fraction
import smlp

from src.smlp_py.smtlib.smt_to_pysmt import smtlib_to_pysmt
from src.smlp_py.smtlib.text_to_sympy import TextToPysmtParser
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import json

_operators_ = [">=", "<=", "<", ">"]

convert_comparison_operators = {
                "=": MarabouCore.Equation.EQ,
                "<=": MarabouCore.Equation.LE,
                ">=": MarabouCore.Equation.GE
            }

class Verifier(ABC):
    @abstractmethod
    def add_disjunction(self):
        pass


class Variable:
    _input_index = 0
    _output_index = 0

    class Type(Enum):
        Real = 0
        Int = 1

    class Bounds:
        def __init__(self, lower=-np.inf, upper=np.inf):
            self.lower = lower
            self.upper = upper

    def __init__(self, form: Type, index=None, name="", is_input=True):
        self.index = index
        self.form = form
        self.name = name
        self.is_input = is_input
        self.bounds = Variable.Bounds()

    @staticmethod
    def get_index(direction="output"):
        return Variable._input_index if direction == "input" else Variable._output_index

    def set_lower_bound(self, lower):
        self.bounds.lower = lower

    def set_upper_bound(self, upper):
        self.bounds.upper = upper

class MarabouVerifier(Verifier):
    def __init__(self, parser=None, variable_ranges=None, is_temp=False):
        # MarabouNetwork containing network instance
        self.network = None

        # Dictionary containing variables
        self.bounds = {}

        # List of MarabouCommon.Equation currently applied to network query
        self.equations = []

        # List of variables
        self.variables = []

        self.variable_ranges = variable_ranges

        self.unscaled_variables = []

        self.model_file_path = "./"
        self.log_path = "marabou.log"
        self.data_bounds_file = self.find_file_path("../../../result/abc_smlp_toy_basic_data_bounds.json")
        self.data_bounds = None
        # Adds conjunction of equations between bounds in form:
        # e.g. Int(var), var >= 0, var <= 3 -> Or(var == 0, var == 1, var == 2, var == 3)

        self.input_index = 0
        self.output_index = 0

        self.parser = parser
        self.network_num_vars = None
        self.init_variables(is_temp=is_temp)

        if self.variable_ranges:
            self.initialize()


    def initialize(self, variable_ranges=None):
        if variable_ranges:
            self.variable_ranges = variable_ranges

        self.model_file_path = self.find_file_path('../../../result/abc_smlp_toy_basic_nn_keras_model_complete.h5')
        self.convert_to_pb()
        self.load_json()
        self.network_num_vars = self.network.numVars
        self.add_unscaled_variables()
        self.create_integer_range()

    def reset(self):
        self.network.clear()
        self.network = Marabou.read_tf('model.pb')
        self.unscaled_variables = []
        self.add_unscaled_variables()
        # Default bounds for network
        for equation in self.equations:
            self.apply_restrictions(equation)


    def load_json(self):
        with open(self.data_bounds_file, 'r') as file:
            self.data_bounds = json.load(file)

    def epsilon(self, e, direction):
        if direction == 'down':
            return np.nextafter(e, -np.inf)
        elif direction == 'up':
            return np.nextafter(e, np.inf)
        else:
            raise ValueError("Direction must be 'up' or 'down'")

    def find_file_path(self, relative_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        relative_h5_path = os.path.join(script_dir, relative_path)
        absolute_h5_path = os.path.normpath(relative_h5_path)
        return absolute_h5_path

    def convert_to_pb(self, output_model_file_path="."):
        model = tf.keras.models.load_model(self.model_file_path)
        tf.saved_model.save(model, output_model_file_path)
        # Load the SavedModel
        model = tf.saved_model.load(output_model_file_path)
        concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        print("converted h5 to pb...")

        # Convert to ConcreteFunction
        frozen_func = convert_variables_to_constants_v2(concrete_func)
        graph_def = frozen_func.graph.as_graph_def()

        # Save the frozen graph
        with tf.io.gfile.GFile('model.pb', 'wb') as f:
            f.write(graph_def.SerializeToString())

        self.network = Marabou.read_tf('model.pb')



    def init_variables(self, is_temp=False) -> None:
        self.create_variables(is_input=True, is_temp=is_temp)
        self.create_variables(is_input=False, is_temp=is_temp)

    def create_variables(self, is_input=True, is_temp=False):
        store = self.parser.inputs if is_input else self.parser.outputs
        for var in store:
            name, type = var
            var_type = Variable.Type.Real if type.lower() == "real" else Variable.Type.Int
            if name.startswith(('x', 'p', 'y')):
                index = self.input_index if is_input else self.output_index
                self.variables.append(Variable(var_type, name=name, index=index, is_input=is_input))

                if is_input:
                    self.input_index += 1
                else:
                    self.output_index += 1

    def create_integer_range(self):
        integer_variables = [variable for variable in self.variables if variable.form == Variable.Type.Int]
        for variable in integer_variables:
            int_range = self.variable_ranges.get(variable.name)
            if not int_range:
                raise Exception(f"Need integer rangers for variable {variable.name}")
            ranges = int_range['interval']
            lower, upper = ranges[0], ranges[-1]
            variable.bounds = Variable.Bounds(lower=lower, upper=upper)
            integer_formula = self.parser.create_integer_disjunction(f'{variable.name}_unscaled', (lower, upper))
            self.add_permanent_constraint(integer_formula)


    def add_unscaled_variables(self):
        for variable in self.variables:
            unscaled_variable = self.network.getNewVariable()
            self.unscaled_variables.append(Variable(Variable.Type.Real, index=unscaled_variable, name=f"{variable.name}_unscaled", is_input=True))

        self.convert_scaled_unscaled()


    def convert_scaled_unscaled(self):
        for scaled_var, unscaled_var in zip(self.variables, self.unscaled_variables):
            bounds = self.data_bounds[scaled_var.name]
            min_value, max_value = bounds["min"], bounds["max"]

            scaling_factor = max_value - min_value

            _, scaled_var_index = self.get_variable_by_name(scaled_var.name)
            _, unscaled_var_index = self.get_variable_by_name(unscaled_var.name)

            # Create an equation representing (x_max - x_min) * x_scaled - x_unscaled =  - x_min
            eq = MarabouUtils.Equation(MarabouCore.Equation.EQ)
            eq.addAddend(scaling_factor, scaled_var_index)
            eq.addAddend(-1, unscaled_var_index)
            eq.setScalar(-min_value)

            # Add the equation to the network
            # self.add_permanent_constraint(eq)
            self.network.addEquation(eq)


    def get_variable_by_name(self, name: str) -> Optional[Tuple[Variable, int]]:
        is_output = name.startswith("y")
        is_unscaled = name.find("_unscaled") != -1
        repository = self.unscaled_variables if is_unscaled else self.variables

        for index, variable in enumerate(repository):
            if variable.name == name:
                if is_unscaled:
                    return variable, variable.index
                elif is_output:
                    index -= self.input_index
                index = self.network.outputVars[0][0][index] if is_output else self.network.inputVars[0][0][index]
                return variable, index
        return None


    def add_permanent_constraint(self, formula):
        self.equations.append(formula)
        self.apply_restrictions(formula)

    def add_bound(self, variable:str, value, direction="upper", strict=True):
        var, var_index = self.get_variable_by_name(f"{variable}_unscaled")
        if var is None:
            return None

        epsilon_direction = "down" if direction == "upper" else "up"
        value = self.epsilon(value, epsilon_direction) if strict else value

        if direction == "upper":
            self.network.setUpperBound(var_index, value)
            var.set_upper_bound(value)
        elif direction == "lower":
            self.network.setLowerBound(var_index, value)
            var.set_lower_bound(value)

    def add_equality(self, variable, value):
        var, var_index = self.get_variable_by_name(f"{variable}_unscaled")

        eq = MarabouUtils.Equation(MarabouCore.Equation.EQ)
        eq.addAddend(1, var_index)
        eq.setScalar(value)
        self.network.addEquation(eq)


    def apply_restrictions(self, formula, need_simplification=False):
        formula = self.parser.simplify(formula)
        conjunctions, disjunctions = self.process_formula(formula)

        for conjunction in conjunctions:
            self.process_comparison(conjunction, need_simplification)

        self.process_disjunctions(disjunctions, need_simplification)

    def transform_pysmt_to_marabou_equation(self, formula):
        symbols, comparator, scalar = formula
        equation_type = None

        if comparator in convert_comparison_operators:
            equation_type = convert_comparison_operators[comparator]
        else:
            if comparator == '<':
                equation_type = MarabouCore.Equation.LE
                scalar = self.epsilon(scalar, "down")
            elif comparator == '>':
                equation_type = MarabouCore.Equation.GE
                scalar = self.epsilon(scalar, "up")

        equation = MarabouUtils.Equation(equation_type)
        equation.setScalar(scalar)

        for parameter in symbols:
            coefficient, symbol = parameter
            # TODO: do not enforce the unscaled variables
            name = str(symbol)
            if name.find("_unscaled") == -1:
                name += "_unscaled"
            symbol, index = self.get_variable_by_name(name)
            equation.addAddend(coefficient, index)

        return equation

    def is_negation_of_ite(self, formula):
        if formula.is_and():
            if len(formula.args()) == 2:
                # this is a custom logical block for handling the negation of objectives which yield a formula that looks like
                # Or(A,B,C) where C = And(Or(D,E),Or(F,G)) (1) , which needs to be translated into:
                # let K = Or(D,E), then C=And(K, Or(F,G)), which is equivalent to: Or(And(K,F),And(K,G)) (2).
                # Then, using (2): And(K,F) = And(F, Or(D,E)), which is equivalent to: Or(And(F,D), And(F,E)) (3)
                # Same applied to And(G, K) = Or(And(G,D), And(G,E)) (4)
                # Finally: Or(And(K,F),And(K,G)) = Or(Or(And(F,D), And(F,E)), Or(And(G,D), And(G,E))),
                # Which can be simplified to Or(And(F,D), And(F,E), And(G,D), And(G,E))
                left = formula.args()[0]
                right = formula.args()[1]
                if left.is_or() and len(left.args()) == 2 and right.is_or() and len(right.args()) == 2:
                    eq_1, eq_2 = left.args()[0], left.args()[1]
                    eq_3, eq_4 = right.args()[0], right.args()[1]
                    return True, [And(eq_1, eq_3), And(eq_1, eq_4), And(eq_2, eq_3), And(eq_2, eq_4)]
        return False, []

    def create_equation(self, formula, from_and=False, need_simplification=False):
        equations = []
        formula = self.parser.simplify(formula)

        if formula.is_and():
            equation = [self.create_equation(eq, from_and=True) for eq in formula.args()]
            return equation
        elif formula.is_le() or formula.is_lt() or formula.is_equals():
            res = self.parser.extract_components(formula, need_simplification)
            equations.append(self.transform_pysmt_to_marabou_equation(res))

        return equations[0] if from_and else equations

    def process_disjunctions(self, disjunctions, need_simplification=False):
        marabou_disjunction = []
        for disjunction in disjunctions:
            # split the disjunction into separate formulas
            for formula in disjunction.args():
                res, formulas = self.is_negation_of_ite(formula)
                if res:
                    for form in formulas:
                        equation = self.create_equation(form, from_and=False, need_simplification=need_simplification)
                        marabou_disjunction.append(equation)
                else:
                    equation = self.create_equation(formula, from_and=False, need_simplification=need_simplification)
                    marabou_disjunction.append(equation)

        if len(marabou_disjunction) > 0:
            self.network.addDisjunctionConstraint(marabou_disjunction)

    def process_formula(self, formula):
        conjunctions = []
        disjunctions = []

        def traverse(node, source=[]):
            if node.is_and():
                # conjunctions.extend(node.args())
                for arg in node.args():
                    traverse(arg, conjunctions)
            elif node.is_or():
                disjunctions.append(node)
            elif node.is_le() or node.is_lt() or node.is_equals():
                source.append(node)
            else:
                # Leaf nodes (symbols, literals, etc.) are not conjunctions or disjunctions
                pass

        traverse(formula)
        return conjunctions, disjunctions

    def process_comparison(self, formula, need_simplification=False):
        if formula.is_le() or formula.is_lt() or formula.is_equals():
            symbols, comparison, constant = self.parser.extract_components(formula, need_simplification)

            if len(symbols) > 1:
                equation = self.transform_pysmt_to_marabou_equation((symbols, comparison, constant))
                self.network.addEquation(equation)
            else:
                _, symbol = symbols[0]
                symbol = str(symbol)

                if comparison == "<=":
                    self.add_bound(symbol, constant, direction="upper", strict=False)
                elif comparison == "<":
                    self.add_bound(symbol, constant, direction="upper", strict=True)
                if comparison == ">=":
                    self.add_bound(symbol, constant, direction="lower", strict=False)
                elif comparison == ">":
                    self.add_bound(symbol, constant, direction="lower", strict=True)
                elif comparison == "=":
                    # TODO: add a marabou equation instead
                    self.add_equality(symbol, constant)
                    # self.add_bound(symbol, constant, direction="lower", strict=False)
                    # self.add_bound(symbol, constant, direction="upper", strict=False)
        else:
            return


    def find_witness(self, witness):
        answers = {"result":"SAT", "witness":{}, 'witness_var':{}}
        for variable in self.unscaled_variables:
            _, unscaled_index = self.get_variable_by_name(variable.name)
            name = variable.name.replace("_unscaled", "")
            scaled_var, _ = self.get_variable_by_name(name)
            answers['witness_var'][scaled_var] = witness[unscaled_index]
            answers['witness'][scaled_var.name] = witness[unscaled_index]
        print(answers['witness'])
        return answers

    def solve(self):
        try:
            results = self.network.solve()
            if results and results[0] == 'unsat':
                return "UNSAT", {"result":"UNSAT", "witness": {}}
            else:  # sat
                return "SAT", self.find_witness(results[1])
        except Exception as e:
            print(e)
            return None

    def add_disjunction(self,):
        pass




if __name__ == "__main__":
    parser = TextToPysmtParser()
    # p2 is an int not a real
    parser.init_variables(symbols=[("x1", "real"), ('x2', 'int'), ('p1', 'real'), ('p2', 'real'),
                                       ('y1', 'real'), ('y2', 'real')])

    mb = MarabouVerifier(parser=parser)
    mb.init_variables(inputs=[("x1", "Real"),('x2', 'Integer'), ('p1', 'Integer'), ('p2', 'Integer')], outputs=[('y1', 'Real'), ('y2', 'Real')])


    def linearize(expr):
        """
        Linearize the given expression, ensuring it is in a linear format.
        """
        if expr.is_real_constant():
            return expr, 0
        elif expr.is_symbol():
            return expr, 0
        elif expr.is_plus():
            lhs, lhs_const = linearize(expr.arg(0))
            rhs, rhs_const = linearize(expr.arg(1))
            return Plus(lhs, rhs), lhs_const + rhs_const
        elif expr.is_minus():
            lhs, lhs_const = linearize(expr.arg(0))
            rhs, rhs_const = linearize(expr.arg(1))
            return Minus(lhs, rhs), lhs_const - rhs_const
        elif expr.is_times():
            const_part = 1
            var_part = None
            for arg in expr.args():
                if arg.is_real_constant():
                    const_part *= arg.constant_value()
                else:
                    var_expr, var_const = linearize(arg)
                    if var_const != 0:
                        raise ValueError(f"Non-linear term detected: {expr}")
                    if var_part is None:
                        var_part = var_expr
                    else:
                        raise ValueError(f"Non-linear term detected: {expr}")
            return Times(Real(const_part), var_part), 0
        else:
            raise ValueError(f"Unsupported operation: {expr}")


    def simplify_to_linear(formula):
        """
        Simplify a given formula to a linear format if possible.
        """
        if formula.is_lt() or formula.is_le():
            lhs, lhs_const = linearize(formula.arg(0))
            rhs, rhs_const = linearize(formula.arg(1))
            return LT(Plus(lhs, Real(lhs_const - rhs_const)), rhs)
        elif formula.is_gt() or formula.is_ge():
            lhs, lhs_const = linearize(formula.arg(0))
            rhs, rhs_const = linearize(formula.arg(1))
            return LT(rhs, Plus(lhs, Real(lhs_const - rhs_const)))
        elif formula.is_equals():
            lhs, lhs_const = linearize(formula.arg(0))
            rhs, rhs_const = linearize(formula.arg(1))
            return Equals(Plus(lhs, Real(lhs_const - rhs_const)), rhs)
        else:
            raise ValueError(f"Unsupported formula type: {formula}")

    y1 = parser.get_symbol("y1")
    y2 = parser.get_symbol("y2")
    p1 = parser.get_symbol("p1")
    p2 = parser.get_symbol("p2")

    # formula = ( (-1 <= 5*x2) | ( (0.0 == x1) & (x2 > 1) ) )
    # Construct the left-hand side: 0.1 * (x1 - 0.2)
    lhs = Times(Real(0.1), Minus(y1, Real(0.2)))

    # Construct the right-hand side: 0.3 * (0.4 * (x2 - x1) - 0.5)
    inner_term = Minus(y2, y1)
    scaled_inner_term = Times(Real(0.4), inner_term)
    rhs_inner = Minus(scaled_inner_term, Real(0.5))
    rhs = Times(Real(0.3), rhs_inner)

    # Construct the inequality: lhs < rhs
    inequality = LT(lhs, rhs)
    # f = simplify_to_linear(inequality)
    # formula = parser.parse("p1==4.0 or (p1==8.0 and p2 > 3)")
    # formula = parser.parse("((3 <= p2) & (p2 <= 4) & (7656119366529843/1125899906842624 <= p1) & (p1 <= 8106479329266893/1125899906842624))")
    # formula = "(let ((|:0| (- p1 7))) (let ((|:1| (- p2 4))) (and (and true (<= (ite (< |:0| 0) (- |:0|) |:0|) (/ 1 5))) (<= (ite (< |:1| 0) (- |:1|) |:1|) (/ 1 5)))))"

    # formula = ((3 <= p2) & (p2 <= 4) & (7656119366529843/1125899906842624 <= p1) & (p1 <= 8106479329266893/1125899906842624))
    formula = And(p1.Equals(Real(4)), Or(p1.Equals(Real(8)), And(LT(Real(3), p2), p1.Equals(Real(5)))))
    var_types = {
        'y1': 'REAL',
        'y2': 'REAL',
        'p1': 'REAL',
        'p2': 'INT',
        'x1': 'REAL',
        'x2': 'INT'
    }
    # formula = smtlib_to_pysmt(formula, var_types)
    # mb.apply_restrictions(formula)


    # mb.add_bounds("x1", (0,10))
    # mb.add_bounds("x2", (-1, 1), num="int")
    # mb.add_bounds("p1", (0, 10), num="grid", grid=[2, 4, 7])
    # mb.add_bounds("p2", (3, 7), num="int")
    # mb.alpha()
    #
    # for var in mb.network.outputVars[0][0]:
    #     print(var)
    #
    exitCode1, vals1, stats1 = mb.solve()
    print(exitCode1)


# TODO: CHECK IF MARABOU NATIVELY SUPPORTS INTEGERS: it does not
    # def add_bounds(self, variable, bounds=None, num="real", grid=None):
    #     var, is_output = self.get_variable_by_name(variable)
    #     if var is None:
    #         return None
    #
    #     # TODO: handle case when one of the two is None
    #     if bounds:
    #         lower, upper = bounds
    #         self.network.setLowerBound(var.index, lower)
    #         self.network.setUpperBound(var.index, upper)
    #
    #         if num == "int":
    #             # add all distinct integer values
    #             grid = range(lower, upper+1)
    #
    #     if num in ["int", "grid"] and grid is not None:
    #         disjunction = []
    #         for i in grid:
    #             eq1 = MarabouUtils.Equation(MarabouCore.Equation.EQ)
    #             eq1.addAddend(1, var.index)
    #             eq1.setScalar(i)
    #             disjunction.append([eq1])
    #
    #         self.network.addDisjunctionConstraint(disjunction)


# def alpha(self):
#     # (((-1 <= x2) & (0.0 <= x1) & (x2 <= 1) & (x1 <= 10.0)) & (((p2 < 5) & (x1 = 10.0)) & (x2 < 12)))
#     #     p2<5 and x1==10 and x2<12
#     # (p2≥5)∨(x1#10)∨(x2≥12)
#
#     p1, is_output = self.get_variable_by_name("p1")
#     p2, is_output = self.get_variable_by_name("p2")
#     x1, is_output = self.get_variable_by_name("x1")
#     x2, is_output = self.get_variable_by_name("x2")
#     y1, is_output = self.get_variable_by_name("y1")
#     y2, is_output = self.get_variable_by_name("y2")
#
#     #
#     # self.network.setUpperBound(p2.index, 5-epsilon)
#     v = Var(p2.index)
#
#     # self.network.addConstraint(v <= self.epsilon(5, "down"))
#     #
#     # self.network.setUpperBound(x1.index, self.epsilon(10,'up'))
#     # self.network.setLowerBound(x1.index, self.epsilon(10, "down"))
#     #
#     # self.network.setUpperBound(x2.index, self.epsilon(12, "down"))
#     #
#     # self.network.setLowerBound(y1.index, 4)
#     # self.network.setUpperBound(y2.index, 8)
#
#     #     p1==4.0 or (p1==8.0 and p2 > 3)
#     eq1 = MarabouUtils.Equation(MarabouCore.Equation.EQ)
#     eq1.addAddend(1, p1.index)
#     eq1.setScalar(4)
#
#     eq2 = MarabouUtils.Equation(MarabouCore.Equation.EQ)
#     eq2.addAddend(1, p1.index)
#     eq2.setScalar(8)
#
#     eq3 = MarabouUtils.Equation(MarabouCore.Equation.GE)
#     eq3.addAddend(1, p2.index)
#     eq3.setScalar(self.epsilon(3, "up"))
#
#     self.network.addDisjunctionConstraint([[eq1], [eq2, eq3]])
#
#     # b1 = self.network.getNewVariable()
#     #
#     # # Define the epsilon value
#     # epsilon = 1e-5
#     #
#     # # Constraint for (y1 + y2) / 2 > 1 when b1 = 1
#     # # This is equivalent to y1 + y2 > 2
#     # self.network.addInequality([y1, y2, b1], [1, 1, -2], -epsilon)  # y1 + y2 - 2*b1 > 0 -> y1 + y2 > 2 when b1 = 1
#     #
#     # # Ensure b1 is binary
#     # self.network.setLowerBound(b1, 0)
#     # self.network.setUpperBound(b1, 1)