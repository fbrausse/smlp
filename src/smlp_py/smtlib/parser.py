import ast
from pysmt.shortcuts import Symbol, And, Or, Not, Implies, Iff, Ite, Equals, Plus, Minus, Times, Div, Pow, Bool, TRUE, FALSE, Int, Real
from pysmt.typing import BOOL, REAL, INT

pysmt_types = {
    "int": INT,
    "real": REAL,
    "bool": BOOL
}

class TextToPysmtParser(object):
    def __init__(self):
        self.symbols = {}
        self._ast_operators_map = {
            ast.Add: Plus,          # Addition
            ast.Sub: Minus,         # Subtraction
            ast.Mult: Times,        # Multiplication
            ast.Div: Div,           # Division
            ast.Pow: Pow,           # Exponentiation
            ast.BitXor: Iff,        # Bitwise XOR (interpreted as logical Iff)

            ast.USub: Minus,        # Unary subtraction (negation)

            ast.Eq: Equals,         # Equality
            ast.NotEq: Not,         # Not equal
            ast.Lt: lambda l, r: l < r,       # Less than
            ast.LtE: lambda l, r: l <= r,     # Less than or equal to
            ast.Gt: lambda l, r: l > r,       # Greater than
            ast.GtE: lambda l, r: l >= r,     # Greater than or equal to

            ast.And: And,           # Logical AND
            ast.Or: Or,             # Logical OR
            ast.Not: Not,           # Logical NOT

            ast.IfExp: Ite          # If expression
        }

    def add_symbol(self, name, symbol_type):
        assert symbol_type in pysmt_types.keys()
        self.symbols[name] = Symbol(name, pysmt_types[symbol_type])

    def parse(self, expr):
        assert isinstance(expr, str)
        symbol_list = self.symbols

        def eval_(node):
            if isinstance(node, ast.Num):
                return Real(node.n) if isinstance(node.n, float) else Int(node.n)
            elif isinstance(node, ast.BinOp):
                return self._ast_operators_map[type(node.op)](eval_(node.left), eval_(node.right))
            elif isinstance(node, ast.UnaryOp):
                return self._ast_operators_map[type(node.op)](eval_(node.operand))
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


if __name__ == "__main__":

    parser = TextToPysmtParser()
    parser.add_symbol('x1', 'int')
    parser.add_symbol('x2', 'real')
    parser.add_symbol('p2', 'real')

    formula = parser.parse('p2<5.0 and x1==10 and x2<12.0')
    print(formula)

