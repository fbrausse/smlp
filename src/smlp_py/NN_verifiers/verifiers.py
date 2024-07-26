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
        lower = -np.inf
        upper = np.inf

    def __init__(self, form: Type, index=None, name="", is_input=True):
        if is_input and not index:
            self.index = Variable._input_index
            Variable._input_index += 1
        elif is_input and index:
            # auxiliary scaled variable
            self.index = index
        else:
            self.index = Variable._output_index
            Variable._output_index += 1

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
    def __init__(self, model_path=None, parser=None):
        # MarabouNetwork containing network instance
        self.network = None

        # Dictionary containing variables
        self.bounds = {}

        # Dictionary containing MarabouCommon.Disjunction for each variable
        self.disjunctions = dict()

        # List containing yet to be added equations and statements
        self.unprocessed_eq = []

        # List of MarabouCommon.Equation currently applied to network query
        self.equations = set()

        # Error variable bounding around excluded values,
        # e.g. var != val -> And(var >= val + epsilon, var <= val - epsilon)

        # List of variables
        self.variables = []

        self.unscaled_variables = []

        self.model_file_path = "./"
        self.log_path = "marabou.log"
        self.data_bounds_file = "/home/kkon/Desktop/smlp/result/abc_smlp_toy_basic_data_bounds.json"
        self.data_bounds = None
        # Adds conjunction of equations between bounds in form:
        # e.g. Int(var), var >= 0, var <= 3 -> Or(var == 0, var == 1, var == 2, var == 3)
        self.int_enable = False

        # Stack for keeping ipq
        self.ipq_stack = []

        self.parser = parser


    def initialize(self):
        self.model_file_path = "/home/kkon/Desktop/smlp/result/abc_smlp_toy_basic_nn_keras_model_complete.h5"
        self.convert_to_pb()
        self.load_json()
        self.add_unscaled_variables()


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


    def convert_to_pb(self, output_model_file_path="."):
        model = tf.keras.models.load_model(self.model_file_path)
        tf.saved_model.save(model, output_model_file_path)
        # Load the SavedModel
        model = tf.saved_model.load(output_model_file_path)
        concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

        # Convert to ConcreteFunction
        frozen_func = convert_variables_to_constants_v2(concrete_func)
        graph_def = frozen_func.graph.as_graph_def()

        # Save the frozen graph
        with tf.io.gfile.GFile('model.pb', 'wb') as f:
            f.write(graph_def.SerializeToString())

        self.network = Marabou.read_tf('model.pb')
        ipq = self.network.getInputQuery()
        self.ipq_stack.append(ipq)
        print("converted h5 to pb...")


    def init_variables(self, inputs: List[Tuple[str, str]], outputs: List[Tuple[str, str]]) -> None:
        for input_var in inputs:
            name, type = input_var
            var_type = Variable.Type.Real if type.lower() == "real" else Variable.Type.Int
            self.variables.append(Variable(var_type, name=name, is_input=True))

        for output_var in outputs:
            name, type = output_var
            var_type = Variable.Type.Real if type.lower() == "real" else Variable.Type.Int
            self.variables.append(Variable(var_type, name=name, is_input=False))



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

            # Create an equation representing (x_max - x_min) * x_scaled - x_unscaled =  - x_min
            eq = MarabouUtils.Equation(MarabouCore.Equation.EQ)
            eq.addAddend(scaling_factor, scaled_var.index)
            eq.addAddend(-1, unscaled_var.index)
            eq.setScalar(-min_value)

            # Add the equation to the network
            self.network.addEquation(eq)


    def get_variable_by_name(self, name: str) -> Optional[Tuple[Variable, int]]:
        is_output = name.startswith("y")
        is_unscaled = name.find("_unscaled")
        repository = self.unscaled_variables if is_unscaled else self.variables

        for index, variable in enumerate(repository):
            if variable.name == name:
                if is_unscaled:
                    return variable, variable.index
                elif is_output:
                    index -= Variable.get_index("input")
                index = self.network.outputVars[0][0][index] if is_output else self.network.inputVars[0][0][index]
                return variable, index
        return None

    def reset(self):
        self.network.clear()
        self.network = Marabou.read_tf(self.model_file_path, modelType="savedModel_v2")

        # Default bounds for network


        if self.int_enable:
            for variable in self.variables:
                if variable.type == Variable.Type.Int:
                    possible_values = list(range(int(variable.bounds.min), int(variable.bounds.max)))
                    possible_values = map(lambda p: "{var} == {val}".format(var=variable.name, val=p), possible_values)
                    self.add_disjunctions(possible_values)
                    print("{var} is int type adding values in range {min} to {max}".format(var=variable,
                                                                                           min=variable.bounds.min,
                                                                                           max=variable.bounds.max))

    def add_bound(self, variable:str, value, direction="upper", strict=True):
        var, var_index = self.get_variable_by_name(variable)
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


    # TODO: CHECK IF MARABOU NATIVELY SUPPORTS INTEGERS: it does not
    def add_bounds(self, variable, bounds=None, num="real", grid=None):
        var, is_output = self.get_variable_by_name(variable)
        if var is None:
            return None

        # TODO: handle case when one of the two is None
        if bounds:
            lower, upper = bounds
            self.network.setLowerBound(var.index, lower)
            self.network.setUpperBound(var.index, upper)

            if num == "int":
                # add all distinct integer values
                grid = range(lower, upper+1)

        if num in ["int", "grid"] and grid is not None:
            disjunction = []
            for i in grid:
                eq1 = MarabouUtils.Equation(MarabouCore.Equation.EQ)
                eq1.addAddend(1, var.index)
                eq1.setScalar(i)
                disjunction.append([eq1])

            self.network.addDisjunctionConstraint(disjunction)

    def apply_restrictions(self, formula):
        conjunctions, disjunctions = self.process_formula(formula)

        for conjunction in conjunctions:
            self.process_comparison(conjunction)

        self.process_disjunctions(disjunctions)

    def transform_pysmt_to_marabou_equation(self, formula):
        symbol, comparator, scalar = formula
        symbol, is_output = self.get_variable_by_name(str(symbol))
        equation_type = None
        scalar = float(scalar.constant_value())

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
        equation.addAddend(1, symbol.index)
        equation.setScalar(scalar)
        return equation

    def create_equation(self, formula):
        equations = []
        if formula.is_and():
            equation = [self.create_equation(eq) for eq in formula.args()]
            return equation
        elif formula.is_le() or formula.is_lt() or formula.is_equals():
            res = self.parser.extract_components(formula)
            equations.append(self.transform_pysmt_to_marabou_equation(res))

        return equations

    def process_disjunctions(self, disjunctions):
        marabou_disjunction = []
        for disjunction in disjunctions:
            # split the disjunction into separate formulas
            for formula in disjunction.args():
                equation = self.create_equation(formula)
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

    def process_comparison(self, formula):
        if formula.is_le() or formula.is_lt() or formula.is_equals():
            symbol, comparison, constant = self.parser.extract_components(formula)
            symbol = str(symbol)
            constant = float(constant.constant_value())

            if comparison == "<=":
                self.add_bound(symbol, constant, direction="upper", strict=False)
            elif comparison == "<":
                self.add_bound(symbol, constant, direction="upper", strict=True)
            if comparison == ">=":
                self.add_bound(symbol, constant, direction="lower", strict=False)
            elif comparison == ">":
                self.add_bound(symbol, constant, direction="lower", strict=True)
            elif comparison == "=":
                self.add_bound(symbol, constant, direction="lower", strict=False)
                self.add_bound(symbol, constant, direction="upper", strict=False)
        else:
            return

    def alpha(self):
        # (((-1 <= x2) & (0.0 <= x1) & (x2 <= 1) & (x1 <= 10.0)) & (((p2 < 5) & (x1 = 10.0)) & (x2 < 12)))
        #     p2<5 and x1==10 and x2<12
        # (p2≥5)∨(x1#10)∨(x2≥12)

        p1, is_output = self.get_variable_by_name("p1")
        p2, is_output = self.get_variable_by_name("p2")
        x1, is_output = self.get_variable_by_name("x1")
        x2, is_output = self.get_variable_by_name("x2")
        y1, is_output = self.get_variable_by_name("y1")
        y2, is_output = self.get_variable_by_name("y2")

        #
        # self.network.setUpperBound(p2.index, 5-epsilon)
        v = Var(p2.index)

        # self.network.addConstraint(v <= self.epsilon(5, "down"))
        #
        # self.network.setUpperBound(x1.index, self.epsilon(10,'up'))
        # self.network.setLowerBound(x1.index, self.epsilon(10, "down"))
        #
        # self.network.setUpperBound(x2.index, self.epsilon(12, "down"))
        #
        # self.network.setLowerBound(y1.index, 4)
        # self.network.setUpperBound(y2.index, 8)

        #     p1==4.0 or (p1==8.0 and p2 > 3)
        eq1 = MarabouUtils.Equation(MarabouCore.Equation.EQ)
        eq1.addAddend(1, p1.index)
        eq1.setScalar(4)

        eq2 = MarabouUtils.Equation(MarabouCore.Equation.EQ)
        eq2.addAddend(1, p1.index)
        eq2.setScalar(8)

        eq3 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        eq3.addAddend(1, p2.index)
        eq3.setScalar(self.epsilon(3, "up"))

        self.network.addDisjunctionConstraint([[eq1], [eq2, eq3]])

        # b1 = self.network.getNewVariable()
        #
        # # Define the epsilon value
        # epsilon = 1e-5
        #
        # # Constraint for (y1 + y2) / 2 > 1 when b1 = 1
        # # This is equivalent to y1 + y2 > 2
        # self.network.addInequality([y1, y2, b1], [1, 1, -2], -epsilon)  # y1 + y2 - 2*b1 > 0 -> y1 + y2 > 2 when b1 = 1
        #
        # # Ensure b1 is binary
        # self.network.setLowerBound(b1, 0)
        # self.network.setUpperBound(b1, 1)
    def find_witness(self, witness):
        answers = {}
        for variable in self.variables:
            _, index = self.get_variable_by_name(variable.name)
            answers[variable.name] = witness[index]
        return answers

    def solve(self):
        try:
            results = self.network.solve()
            if results and results[0] == 'unsat':
                return {}
            else:  # sat
                return self.find_witness(results[1])
        except Exception as e:
            print(e)
            return None

    def add_disjunction(self,):
        pass




if __name__ == "__main__":
    parser = TextToPysmtParser()
    # p2 is an int not a real
    parser.init_variables(inputs=[("x1", "real"), ('x2', 'int'), ('p1', 'real'), ('p2', 'real'),
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
