import re

from pysmt.smtlib.parser import SmtLibParser
from pysmt.shortcuts import Symbol, simplify, get_env
from pysmt.typing import REAL, INT, BOOL
from io import StringIO
from pysmt.rewritings import CNFizer

from pysmt.shortcuts import Symbol, And, LE, Real, Ite, Or, Not, LT, Equals, Plus, Minus, Times, Div
from pysmt.typing import REAL
from sympy import sympify


def smtlib_to_pysmt(smt_query, var_types):
    """
    Converts an SMT-LIB query string to a PySMT formula.

    Parameters:
    smt_query (str): The SMT-LIB query string.
    var_types (dict): A dictionary mapping variable names to their types (REAL, INT, BOOL).

    Returns:
    pysmt.shortcuts.FNode: The PySMT formula.
    """
    # Initialize the SMT-LIB parser
    parser = SmtLibParser()

    # Build the declarations for the variables
    declarations = []
    for var, vtype in var_types.items():
        if vtype == 'REAL':
            declarations.append(f"(declare-fun {var} () Real)")
            Symbol(var, REAL)
        elif vtype == 'INT':
            declarations.append(f"(declare-fun {var} () Int)")
            Symbol(var, INT)
        elif vtype == 'BOOL':
            declarations.append(f"(declare-fun {var} () Bool)")
            Symbol(var, BOOL)
        else:
            raise ValueError(f"Unsupported variable type: {vtype}")

    # Join the declarations with the original SMT-LIB query
    smt_query_with_declarations = "\n".join(declarations) + f"\n(assert {smt_query})"

    # Parse the SMT-LIB query
    script = parser.get_script(StringIO(smt_query_with_declarations))

    # Extract the formula from the script
    formula = script.get_last_formula()

    # Simplify the parsed formula
    simplified_formula = simplify(formula)

    return simplified_formula


def convert_fractions_to_floats(formula: str) -> str:
    # Regular expression to find fractions in the format 'numerator/denominator'
    fraction_pattern = re.compile(r'(\d+/\d+)')

    def fraction_to_float(match):
        fraction = match.group()
        numerator, denominator = map(int, fraction.split('/'))
        return str(numerator / denominator)

    # Substitute all fractions in the formula with their float equivalents
    formula_with_floats = fraction_pattern.sub(fraction_to_float, formula)

    return formula_with_floats


def convert_ternary_to_logic(formula: str) -> str:
    # Regular expression to find ternary statements in the format 'condition ? true_expr : false_expr'
    ternary_pattern = re.compile(r'\(([^()]+)\?\(([^()]+)\):\(([^()]+)\)\)')

    while ternary_pattern.search(formula):
        formula = ternary_pattern.sub(
            lambda match: f'(({match.group(1)}) & ({match.group(2)}) | (~({match.group(1)}) & ({match.group(3)})))',
            formula)

    return formula

def pysmt_convert_fractions_to_floats(term):
    if term.is_constant() and term.is_real_constant():
        value = term.constant_value()
        return Real(float(value))
    elif term.is_symbol():
        return term
    elif term.is_plus() or term.is_minus() or term.is_times() or term.is_div():
        if term.node_type() == 12:
            return Plus(*[pysmt_convert_fractions_to_floats(arg) for arg in term.args()])
        elif term.node_type() == 13:
            return Minus(*[pysmt_convert_fractions_to_floats(arg) for arg in term.args()])
        elif term.node_type() == 14:
            return Times(*[pysmt_convert_fractions_to_floats(arg) for arg in term.args()])
        elif term.node_type() == 15:
            return Div(*[pysmt_convert_fractions_to_floats(arg) for arg in term.args()])
    return term

def recursively_convert_ite(term):
    if term.is_ite():
        condition = term.arg(0)
        true_branch = recursively_convert_ite(term.arg(1))
        false_branch = recursively_convert_ite(term.arg(2))
        return Or(And(condition, true_branch), And(Not(condition), false_branch))
    elif term.is_and():
        return And(*[recursively_convert_ite(arg) for arg in term.args()])
    elif term.is_or():
        return Or(*[recursively_convert_ite(arg) for arg in term.args()])
    elif term.is_not():
        return Not(recursively_convert_ite(term.arg(0)))
    elif term.is_le() or term.is_lt() or term.is_equals():
        left = recursively_convert_ite(term.arg(0))
        right = recursively_convert_ite(term.arg(1))
        if term.node_type() == 16:
            return LE(left, right)
        elif term.node_type() == 17:
            return LT(left, right)
        elif term.node_type() == 18:
            return Equals(left, right)
    else:
        return pysmt_convert_fractions_to_floats(term)

# node types:
# 12: +
# 13: -
# 14: /
# 15: *
# 16: <=
# 17: <
# 18: ==
# 19: ITE


# Example usage
if __name__ == "__main__":



    # Define the SMT-LIB query as a string
    # smt_query = "(let ((|:0| (* (/ 281474976710656 2944425288877159) (- y1 (/ 1080863910568919 4503599627370496))))) (let ((|:1| (* (/ 281474976710656 2559564553220679) (- (* (/ 1 2) (+ y1 y2)) (/ 1170935903116329 1125899906842624))))) (and (>= (ite (< |:0| |:1|) |:0| |:1|) 1) (and (>= y1 4) (>= y2 8)))))"
    smt_query = "(and (and true (and (>= x1 0) (<= x1 10))) (and (>= x2 (- 1)) (<= x2 1)))"
    # Define variable types
    var_types = {
        'y1': 'REAL',
        'y2': 'REAL',
        'p1': 'REAL',
        'p2': 'REAL',
        'x1': 'REAL',
        'x2': 'REAL'
    }

    # Convert SMT-LIB to PySMT
    pysmt_formula = smtlib_to_pysmt(smt_query, var_types)
    #
    # pysmt_formula = recursively_convert_ite(pysmt_formula)
    #
    # print(pysmt_formula.serialize())

    # pysmt_formula = pysmt_formula.serialize()
    # # Print the PySMT formula
    # print("Converted PySMT Formula:")
    # print(pysmt_formula)
    #
    # pysmt_formula = convert_fractions_to_floats(pysmt_formula)
    # print("Removed fractions:")
    # print(pysmt_formula)
    #
    # pysmt_formula = recursively_convert_ite(sympify(pysmt_formula))
    # # pysmt_formula = convert_ternary_to_logic(pysmt_formula)
    # print("Removed ITE:")
    # print(pysmt_formula)


    cnfizer = CNFizer()
    cnf_formula = cnfizer.convert(pysmt_formula)
    print("CNF PySMT Formula:")
    print(cnf_formula)

    # Example: Add an additional constraint to the formula
    # p1 = Symbol('p1', )
    # additional_constraint = p1 != 3
    # combined_formula = simplify(pysmt_formula & additional_constraint)

    # print("Combined Formula with Additional Constraint:")
    # print


#####################################################################################
