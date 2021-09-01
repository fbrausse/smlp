from re import S
from numpy import index_exp
from tensorflow.python.keras.backend import variable
from z3.z3 import Q
from maraboupy import Marabou
from maraboupy import MarabouCore
import tensorflow as tf
from typing import Dict, Tuple
from common import MinMax
import z3
from fractions import Fraction

from enum import Enum
import json
import sys

# order of variables is important for "find_operator" method
_operators_ = [">=", "<=","<", ">"]

# Determines whether string is a number 
def is_number(s: str) -> bool:
    return s.isnumeric() or (s[0] == '-' and s[1:].isnumeric())

# returns location of operator + type
def find_operator(s: str):
    for operator in _operators_:
        if s.find(operator) != -1:
            return s.find(operator), operator

# Class representing a disjunction of equations for a single variable
class Disjunction:
    
    def __init__(self, variable_name=""):

        # List of equations representing disjunction
        self.disjunction = []

        # variable name
        self.variable_name = variable_name

    def __add__(self, other):

        assert isinstance(other, list)
        for elem in other:
            assert isinstance(elem, Equation)

        self.disjunction.append(other)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self.disjunction):
            j = self.n
            self.n += 1
            return self.disjunction[j]
        else:
            raise StopIteration

    def clear(self):
        self.disjunction.clear()

    def append(self,other):
        self.disjunction.append(other)

    def remove(self, other):
        for j in range(len(self.disjunction)):
            if self.disjunction[j] == other:
                self.disjunction.remove(other)

    def __str__(self):
        return map(self.disjunction, lambda x: str(x))

# Class representing equation in the form. {variable} {operator} {scalar}
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

    def __lt__   (self, o: object):

        if type(o) == Equation:
            if self.variable == o.variable:
                if self.operator == o.operator:
                    return self.scalar < o.scalar
                else:
                    return self.operator < o.operator
            else:
                return self.variable < o.variable

# Class representing Variable
class Variable:

    class Type(Enum):
        Real = 0
        Int = 1

    def __init__(self, index: int, type, bounds: MinMax, name=""):
        self.index = index
        self.type = type
        self.bounds = bounds 
        self.name = name 

    def __str__(self) -> str:
        if self.name == "":
            return "Var {0}".format(self.index)
        else:
            return self.name

    def __hash__(self) -> int:
        if self.name == "":
            return self.index
        else:
            return hash(self.name)

    def __eq__(self, o: object) -> bool:
        return self.name == o

    def __lt__(self, o:object) -> bool:
        if self.name == "":
            return self.index < o.index
        else:
            return self.name < o.name