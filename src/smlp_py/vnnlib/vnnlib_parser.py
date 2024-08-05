
import re

import sympy
from sympy import *
from sympy.logic.boolalg import And, Or, Not
from sympy.core.relational import Relational
from sympy.core.numbers import *
import ast
import smlp
import pysmt

def variable_init_string(variable):
    return f"(declare-const {variable} Real)\n"


def replace_let_vars(expression):
    pattern = r"\(let \(\((\|:\d+\|) \((.+?)\)\)\) (.*)\)"
    while 'let' in expression:
        match = re.search(pattern, expression)
        if not match:
            break

        var_name, var_expr, rest_expression = match.groups()

        rest_expression = rest_expression.replace(var_name, f"({var_expr})")

        expression = rest_expression

    return expression


class VnnLibParser(object):

    def __init__(self, inputs=[], outputs=[]):
        from maraboupy import Marabou, MarabouCore

        self.variables = {}
        self.file_name = "query.vnnlib"
        self.inputs = inputs
        self.outputs = outputs
        self.final_output = ""
        self.sympy_expr = ""
        self.symbols = None
        self.symbol_dict = None
        self.global_variable_constraints = {}
        self.formula_list = []
        self.initialize()

        self.text_parser = TextToSympyParser(self.symbol_dict)

    def initialize(self):
        self.initialize_variables(self.inputs, "X")
        self.initialize_variables(self.outputs, "Y")

        symb = ""
        for symbol in self.inputs:
            symb += f"{symbol} "

        for symbol in self.outputs:
            symb += f"{symbol} "

        symb = symb[:-1]
        self.symbols = symbols(symb)
        self.symbol_dict = {str(sym): sym for sym in self.symbols}
        self.symbol_dict.update({
            'And': And,
            'Or': Or,
            'Not': Not
        })


    def replace_names(self):
        for key, variable in self.variables.items():
            self.final_output = self.final_output.replace(key, variable)

    def initialize_variables(self, data, variable_name):
        for i, arg in enumerate(data):
            variable = f"{variable_name}_{i}"
            self.variables[arg] = variable
            self.final_output += variable_init_string(variable)

    def finalize(self):
        self.replace_names()
        with open(self.file_name, 'w') as file:
            file.write(self.final_output)
        self.call_marabou()


    def add(self, expression):
        expression = replace_let_vars(expression)
        self.final_output += f"(assert \n ({expression}) \n )\n"

    def and_(self, exp1, exp2):
        return sympy.logic.boolalg.And(exp1, exp2)

    def and_multi_(self, forms):
        res = True
        for form in forms:
            res = form if res is True else self.and_(res, form)
        return res
    def or_(self, exp1, exp2):
        return sympy.logic.boolalg.Or(exp1, exp2)

    def equal(self, var, num):
        return sympy.And(var <= num, var >= num)

    def get_symbol(self, symbol):
        return self.symbol_dict.get(symbol)

    def save(self, formula):
        self.formula_list.append(formula)

    def not_(self, expression):
        expression = replace_let_vars(expression)

    def sympy_and(self, exp1, exp2):
        return f"And({exp1}, {exp2})"

    def add_global_constaints(self, variable, expression):
        # check if the constraint exists in our list
        assert variable in self.symbol_dict

        self.global_variable_constraints[variable] = expression



    def call_marabou(self):
        onnx_file = "/home/ntinouldinho/Desktop/smlp/src/test.onnx"
        property_filename = f"/home/ntinouldinho/Desktop/smlp/src/{self.file_name}"

        network = Marabou.read_onnx(onnx_file)
        network.saveQuery("./query.txt")
        ipq = Marabou.load_query("./query.txt")
        MarabouCore.loadProperty(ipq, property_filename)
        exitCode_ipq, vals_ipq, _ = Marabou.solve_query(ipq, propertyFilename=property_filename, filename="res.log")




class SymbolicExpressionHandler:
    def __init__(self, variables):
        self.symbols = symbols(variables)
        self.symbol_dict = {str(sym): sym for sym in self.symbols}
        self.symbol_dict.update({
            'And': And,
            'Or': Or,
            'Not': Not
        })

    def parse_expression(self, expression):
        return parse_expr(expression, local_dict=self.symbol_dict)

    def sympy_to_vnnlib(self, expr):

        def parse_expr(e):
            # Handle logical operators
            if isinstance(e, And):
                return "(and {})".format(' '.join(parse_expr(arg) for arg in e.args))
            elif isinstance(e, Or):
                return "(or {})".format(' '.join(parse_expr(arg) for arg in e.args))
            elif isinstance(e, Not):
                return "(not {})".format(parse_expr(e.args[0]))

            elif isinstance(e, Relational):
                left = parse_expr(e.lhs)
                right = parse_expr(e.rhs)
                op = e.rel_op
                return "({} {} {})".format(op, left, right)

            elif isinstance(e, Abs):
                return "(abs {})".format(parse_expr(e.args[0]))

            elif isinstance(e, (Symbol, Integer, Float, Zero, One)):
                return str(e)

            else:
                raise TypeError("Unsupported type: {}".format(type(e)))

        # Start the parsing
        return parse_expr(expr)


class TextToSympyParser(object):
    def __init__(self, map):
        self.symbols = map
        self._ast_operators_map = {
            ast.Add: sympy.Add,  # Addition
            ast.Sub: sympy.Mul,  # Subtraction (handles through Mul with -1)
            ast.Mult: sympy.Mul,  # Multiplication
            ast.Div: sympy.div,  # Division (true division, use sympy.Mul with sympy.Pow for reciprocal)
            ast.Pow: sympy.Pow,  # Exponentiation
            ast.BitXor: sympy.Xor,  # Bitwise XOR

            ast.USub: sympy.Mul,  # Unary subtraction (negation, effectively multiplying by -1)

            ast.Eq: sympy.Eq,  # Equality
            ast.NotEq: sympy.Ne,  # Not equal
            ast.Lt: sympy.Lt,  # Less than
            ast.LtE: sympy.Le,  # Less than or equal to
            ast.Gt: sympy.Gt,  # Greater than
            ast.GtE: sympy.Ge,  # Greater than or equal to

            ast.And: sympy.And,  # Logical AND
            ast.Or: sympy.Or,  # Logical OR
            ast.Not: sympy.Not,  # Logical NOT

            ast.IfExp: sympy.ITE  # If expression
        }

    def parse(self, expr):
        # print('evaluating AST expression ====', expr)
        assert isinstance(expr, str)
        symbol_list = self.symbols

        # recursion
        def eval_(node):
            if isinstance(node, ast.Num):  # <number>
                # print('node Num', node.n, type(node.n))
                return sympy.Float(node.n)
            elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
                # print('node BinOp', node.op, type(node.op))
                if type(node.op) not in [ast.Div, ast.Pow]:
                    return self._ast_operators_map[type(node.op)](eval_(node.left), eval_(node.right))
                elif type(node.op) == ast.Div:
                    if type(node.right) == ast.Constant:
                        if node.right.n == 0:
                            raise Exception('Division by 0 in parsed expression ' + expr)
                        elif not isinstance(node.right.n, int):
                            raise Exception(
                                'Division in parsed expression is only supported for integer constants; got ' + expr)
                        else:
                            # print('node.right.n', node.right.n, type(node.right.n))
                            return self._ast_operators_map[ast.Mult](smlp.Cnst(smlp.Q(1) / smlp.Q(node.right.n)),
                                                                     eval_(node.left))
                    else:
                        raise Exception('Opreator ' + str(self._ast_operators_map[type(node.op)]) +
                                        ' with non-constant demominator within ' + str(
                            expr) + ' is not supported in ast_expr_to_term')
                elif type(node.op) == ast.Pow:
                    if type(node.right) == ast.Constant:
                        if type(node.right.n) == int:
                            # print('node.right.n', node.right.n)
                            if node.right.n == 0:
                                return sympy.Float(1)
                            elif node.right.n > 0:
                                left_term = res_pow = eval_(node.left)
                                for i in range(1, node.right.n):
                                    res_pow = sympy.Mul(res_pow, left_term)
                                # print('res_pow', res_pow)
                                return res_pow
                    raise Exception('Opreator ' + str(self._ast_operators_map[type(node.op)]) +
                                    ' with non-constant or negative exponent within ' +
                                    str(expr) + 'is not supported in ast_expr_to_term')
                else:
                    raise Exception('Implementation error in function ast_expr_to_term')
            elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
                # print('unary op', node.op, type(node.op));
                return self._ast_operators_map[type(node.op)](eval_(node.operand))
            elif isinstance(node, ast.Name):  # variable
                # print('node Var', node.id, type(node.id))
                return symbol_list[node.id]
            elif isinstance(node, ast.BoolOp):
                res_boolop = self._ast_operators_map[type(node.op)](eval_(node.values[0]), eval_(node.values[1]))
                if len(node.values) > 2:
                    for i in range(2, len(node.values)):
                        res_boolop = self._ast_operators_map[type(node.op)](res_boolop, eval_(node.values[i]))
                # print('res_boolop', res_boolop)
                return res_boolop
            elif isinstance(node, ast.Compare):
                # print('node Compare', node.ops, type(node.ops), 'left', node.left, 'comp', node.comparators);
                # print('len ops', len(node.ops), 'len comparators', len(node.comparators))
                assert len(node.ops) == len(node.comparators)
                left_term_0 = eval_(node.left)
                right_term_0 = eval_(node.comparators[0])
                if type(node.ops[0]) == ast.Eq:
                    # if x==10 then x<=10 and x>=10
                    if type(left_term_0) == sympy.Symbol and type(right_term_0) == sympy.Float:
                        res_comp =  sympy.And(left_term_0 <= right_term_0, left_term_0 >= right_term_0)
                else:
                    res_comp = self._ast_operators_map[type(node.ops[0])](left_term_0,
                                                                      right_term_0);  # print('res_comp_0', res_comp)
                if len(node.ops) > 1:
                    # print('enum', list(range(1, len(node.ops))))
                    left_term_i = right_term_0
                    for i in range(1, len(node.ops)):
                        right_term_i = eval_(node.comparators[i])
                        # print('i', i, 'left', left_term_i, 'right', right_term_i)
                        res_comp_i = self._ast_operators_map[type(node.ops[i])](left_term_i, right_term_i)
                        res_comp = sympy.And(res_comp, res_comp_i)  # self._ast_operators_map[type(node.op.And)]
                        # for the next iteration (if any):
                        left_term_i = right_term_i
                # print('res_comp', res_comp)
                return res_comp
            elif isinstance(node, ast.List):
                # print('node List', 'elts', node.elts, type(node.elts), 'expr_context', node.expr_context);
                raise Exception('Parsing expressions with lists is not supported')
            elif isinstance(node, ast.Constant):
                if node.n == True:
                    return True
                if node.n == False:
                    return False
                raise Exception('Unsupported comstant ' + str(node.n) + ' in funtion ast_expr_to_term')
            elif isinstance(node, ast.IfExp):
                res_test = eval_(node.test)
                res_body = eval_(node.body)
                res_orelse = eval_(node.orelse)
                # res_ifexp = smlp.Ite(res_test, res_body, res_orelse)
                res_ifexp = self._ast_operators_map[ast.IfExp](res_test, res_body, res_orelse)
                # print('res_ifexp',res_ifexp)
                return res_ifexp
            else:
                print('Unexpected node type ' + str(type(node)))
                # print('node type', type(node))
                raise TypeError(node)

        return eval_(ast.parse(expr, mode='eval').body)


if __name__ == "__main__":
    # variable_names = 'x1 x2 p1 p2 y1 y2'
    # handler = SymbolicExpressionHandler(variable_names)
    # expr_string = "Or(And(x1<=0, y1>9), And(x2<7, y2>10))"
    # parsed_expr = handler.parse_expression(expr_string)
    # b = Not(parsed_expr)
    # b = simplify_logic(b)
    # print(b)
    #
    from sympy import symbols, parse_expr, And
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations

    # Define symbols
    p2, x1, x2 = symbols('p2 x1 x2')

    # Define the expression string
    expression_str = "(p2 < 5) & (x1 == 10) & (x2 < 12)"

    # Standard transformations
    transformations = standard_transformations

    # Parse the expression
    expr = parse_expr(expression_str, transformations=transformations, local_dict={'p2': p2, 'x1': x1, 'x2': x2})

    # Print the parsed expression
    print(expr)











