from re import S, VERBOSE
from numpy import add, index_exp
from tensorflow.python.keras.backend import variable
from tensorflow.python.keras.backend_config import epsilon
from z3.z3 import Q
from maraboupy import Marabou
from maraboupy import MarabouCore
import tensorflow as tf
from typing import Dict, Tuple
from common import MinMax
import z3
from fractions import Fraction
import numpy as np

from enum import Enum
import json

from MarabouCommon import *

class MarabouSolver():

    def __init__(self):
        
        # MarabouNetwork containing network instance
        self.network = None

        # Dictionary containing variables 
        self.bounds = {}

        # Dictionary containing MarabouCommon.Disjunction for each variable 
        self.disjunctions = dict()

        # List containing yet to be added equations and statements
        self.unprocessed_eq = []

        # List of MarabouCommon.Equation currently applied to network query
        self.equations = list()


        # List of variables 
        self.variables = []

        self.model_file_path = "./"
        self.log_path = "marabou.log"
        self.record_path = "marabou.check"

        # Adds conjunction of equations between bounds in form:
        # e.g. Int(var), var >= 0, var <= 3 -> Or(var == 0, var == 1, var == 2, var == 3)
        self.int_enable = True

        self.pre_ipq = None

    def log(self,v):
        with open(self.log_path,"a") as f:
            f.write(str(v) + "\n")
            

    def eval(self, point):
        #x = 0
        self.network.evaluateWithMarabou(point, filename="marabou.eval")

    def record(self,v):
        with open(self.record_path,"a") as f:
            f.write(str(v) + "\n")

    def dump(self):
        self.pre_ipq.dump()

    # Convert h5 to pb model which is necessary for the 
    def convert_to_pb(self, input_model_file_path: str, output_model_file_path: str):
        model = tf.keras.models.load_model(input_model_file_path)
        tf.saved_model.save(model,output_model_file_path)
        self.network = Marabou.read_tf(self.model_file_path,modelType="savedModel_v2")
        for inputB in self.network.inputVars[0][0]:
            self.network.setLowerBound(inputB, 0)
            self.network.setUpperBound(inputB, 1.0)
        print("converted h5 to pb...")

    # initiliaze variable bounds and network
    def initialize(self, spec_path):
        with open(spec_path, 'r') as f:
            self.all_spec = json.load(f, parse_float=Fraction)

    # Outputs z3 solver.sexpr() and current applied equations
    def check(self, solver):
        self.log("\n checking ... \n sexpr:")

        # Z3 Solver state 
        self.log(solver.sexpr())

        # Current equations (not including network)
        self.log("\n equations: \n")
        
        for equation in self.equations:
            self.log(equation)


    # Reset the network to intial bounds and initialize network
    def reset(self):
        self.network.clear()
        self.network = Marabou.read_tf(self.model_file_path,modelType="savedModel_v2")

        # Default bounds for netwokr
        for inputB in self.network.inputVars[0][0]:
            self.network.setLowerBound(inputB, 0)
            self.network.setUpperBound(inputB, 1.0)

        self.equations.clear()
        self.disjunctions.clear()

        if self.int_enable:
            for variable in self.variables:
                if variable.type == Variable.Type.Int:
                    possible_values = list(range(int(variable.bounds.min), int(variable.bounds.max)))
                    possible_values = map(lambda p: "{var} == {val}".format(var=variable.name, val=p), possible_values)
                    self.add_disjunctions(possible_values)
                    print("{var} is int type adding values in range {min} to {max}".format(var=variable,
                    min=variable.bounds.min, max=variable.bounds.max))
                
    # Adds bounds 0 <= x <= 1
    def add_bounds(self,bounds):
        i = 0

        for key in bounds.keys():
            if key in self.variables:
                continue

            type = Variable.Type.Real
            if self.all_spec[i]["range"] == "int":
                type = Variable.Type.Int
            

            self.variables.append(Variable(
                    i, type,MinMax(bounds[key]['min'],bounds[key]['max']), key)
                )

            self.log("Variable ({}) min: {}, max: {}, range: {}".format(key,
            self.variables[i].bounds.min,
            self.variables[i].bounds.max
            ,self.variables[i].bounds.range))

            self.bounds[key] = self.variables[i]

            i += 1

        self.bounds["Output"] = Variable(i, Variable.Type.Real,MinMax(0,1), "Output")


        del self.all_spec
        if self.int_enable:
            for variable in self.variables:
       
                if variable.type == Variable.Type.Int:
                    possible_values = list(range(int(variable.bounds.min), int(variable.bounds.max)))
                    possible_values = map(lambda p: "{var} == {val}".format(var=variable.name, val=p), possible_values)
                    self.add_disjunctions(possible_values)
                    print("{var} is int type adding values in range {min} to {max}".format(var=variable,
                    min=variable.bounds.min, max=variable.bounds.max))

    # Apply disjunctions that were added
    def apply_disjunctions(self):

        # Loop input variables 
        for var in self.disjunctions.keys():
            disjunctions = []
            flat = []

            # Loop through each Disjunction
            for disjunction in self.disjunctions[var]:
                # Loop through each conjunction of MarabouCommon.Equations
                for equation in disjunction:

                    # Create Equation
                    eq = None
                    if equation.op() == "==":
                        eq = MarabouCore.Equation(MarabouCore.Equation.EQ)
                    elif equation.op() == ">=":
                        eq = MarabouCore.Equation(MarabouCore.Equation.GE)
                    else:
                        eq = MarabouCore.Equation(MarabouCore.Equation.LE)

                    eq.addAddend(1, equation.lhs().index)
                    eq.setScalar(equation.lhs().bounds.norm(float(equation.rhs())))

                    disjunctions.append([eq])
                    flat.append(equation)
            self.equations.append("Or(\n" + ''.join(str(e) + ",\n" for e in flat) + ")")

        # Add disjunction constraint to network
            self.network.addDisjunctionConstraint(disjunctions)

        print("Disjunctions applied")

    # Add disjunctions to be processed 
    def add_disjunctions(self, disjunction):
        for equation in disjunction:

            # Process equation
            parts = str(equation).split(" ")
 
            # Swap lhs = var, rhs = val
            if is_number(parts[0]) and not is_number(parts[2]):
                parts[0], parts[2] = parts[2], parts[0]

            # Determine variable
            var_index = self.variables.index(parts[0])
            
            # Cannot find variable in input variables
            if var_index == -1:
                self.log("incorrect format or variable not formed: {1}".format(parts))
                exit(-1)

            var = self.variables[var_index]

            # If disjunction does not exist, add to dictionary
            if not parts[0] in self.disjunctions.keys():
                self.disjunctions[parts[0]] = Disjunction(str(parts[0]))

            # Check whether constraint is == or !=
            if parts[1] == '==':
                self.disjunctions[parts[0]].append([Equation(var, parts[1], float(parts[2]))])
            elif parts[1] == '!=':

                # Remove == constraint from disjunction if exists and replace with epsilon bounded region
                self.disjunctions[parts[0]].remove(["{0} {1} {2}".format(self.variables[self.variables.index(parts[0])].index,
                                                                         "==",str(float(parts[2])))])
                self.disjunctions[parts[0]].append([Equation(var, ">=", np.nextafter(float(parts[2]), np.inf)),
                                                    Equation(var, "<=", np.nextafter(float(parts[2]), -np.inf))])


        print("Disjunction added")


    # Not intended for use with "=="
    # Expects unnormalized scalars
    # Adds singular constraints in form {var} {op} {val}
    def add_constraints(self, constraints):
        self.log("adding constraints")
        
        # Loop through input strings
        for c in constraints:
            size = 2 # size of operator (number of characters)
            add_epsilon = False # whether bound is strict inequality

            c = str(c)
            location, op = find_operator(c) # find location and type of operator
            self.log(c)
            if len(op) == 1:
                add_epsilon = True
                size = 1
                op += "="

            # split equation around operator
            lhs = c[:location]
            rhs = c[location + size:]

            if rhs.replace(" ", "") in self.bounds.keys():
                lhs, rhs = rhs, lhs

                if op == ">=":
                    op = "<="
                elif op == "<=":
                    op = ">="

            # Check if lhs of equation is var or rhs is 
            if lhs.replace(" ","") in self.bounds.keys():

                # Evaluate numeric expression 
                scalar = eval(rhs)
                #self.log("{} {} {}".format(lhs,op,scalar))
                #self.log(self.bounds[str(lhs.replace(" ",""))].bounds.norm(float(scalar)))
                #self.log("min: {}, max: {}, range: {}\n".format(self.bounds[str(lhs.replace(" ",""))].bounds.min,
                                                        #self.bounds[str(lhs.replace(" ",""))].bounds.max, 
                                                        #self.bounds[str(lhs.replace(" ",""))].bounds.range))



                addend = lhs.replace(" ","")

                if op == "<=":

                    if add_epsilon:
                        scalar = np.nextafter(scalar,-np.inf)
                        

                    # Set upperbound on variable and add equation
                    self.network.setUpperBound(self.bounds[addend].index, self.bounds[str(addend)].bounds.norm(scalar))
                    
                    self.equations.append(Equation(
                            self.bounds[addend],op,scalar)
                        )
                    
                else:

                    if add_epsilon:
                        scalar = np.nextafter(scalar,np.inf)
                    # Set upperbound on variable and add equation
                    self.network.setLowerBound(self.bounds[addend].index, self.bounds[str(addend)].bounds.norm(scalar))
                    
                    self.equations.append(Equation(
                            self.bounds[addend],op,(scalar)
                        ))
                                    
            else:
                self.log("unknown constant {0}".format(c))
                exit(-1)

    # Solve query returns -> (Bool (sat or unsat), Model)
    def solve(self, solver):
        print("Solving ... ")

        # Process equations
        for ls in self.unprocessed_eq:
            
            # Sometimes there will be an empty list, just remove it
            if type(ls) == list:
                self.log(ls)
                continue

            if type(ls) == z3.z3.BoolRef:
                if str(ls.decl()) == "Or":
                    self.add_disjunctions(ls.children())
                if str(ls.decl()) == "And":
                    self.add_constraints(ls.children())
                if str(ls.decl()) == ">=" or str(ls.decl()) == "<=":
                    self.add_constraints(["{0} {1} {2}".format(ls.children()[0], str(ls.decl()), ls.children()[1])])
            
        # Clear equations
        self.unprocessed_eq.clear()

        # Apply disjunctions
        self.apply_disjunctions()

        for inputB in self.network.inputVars[0][0]:
            if self.network.upperBounds[inputB] >= 1.0:
                self.network.setUpperBound(inputB, 1.0)
            if self.network.lowerBounds[inputB] <= 0.0:
                self.network.setLowerBound(inputB, 0)

        ipq = self.network.getMarabouQuery()
        Marabou.solve_query(ipq)
        self.pre_ipq = ipq
        #ipq.dump()
        #x = input()
        # check z3 and marabou for consistency -> log file
        self.check(solver)

        print("Marabou is solving ... ")

        # options
        options = Marabou.createOptions(snc=True, numWorkers=4, verbosity=0)

        

        b, stats = self.network.solve(options=options)
        print("Marabou solved ")

        self.log("Lower bounds")
        
        for i in range(len(self.bounds.keys()) - 1):
            self.log("{0} -> {1}".format(self.variables[i].name, self.variables[i].bounds.denorm(self.network.lowerBounds[i])))
        self.log("-" * 20)

        self.log("Upper bounds")
        for i in range(len(self.bounds.keys()) - 1):
            self.log("{0} -> {1}".format(self.variables[i].name, self.variables[i].bounds.denorm(self.network.upperBounds[i])))

        self.log("-" * 20)
        # unsat
        if len(b) == 0:
            return False, None
        else: # sat
            for i in range(len(self.bounds.keys()) - 1):
                self.log("{0} -> {1}".format(self.variables[i].name, self.variables[i].bounds.denorm(b[i])))
            self.log("{0} -> {1}".format("Output", b[len(b)-1]))
            return True, b

    # rearrage all equations to be 
    # sum(c * x) = scalar
    def add(self, *p):
        if type(p) != z3.z3.BoolRef:
            for ps in p:
                self.unprocessed_eq.append(ps)
        else:
            self.unprocessed_eq.append(p)

    # Applies >= threshhold for safe point
    def add_safepoint(self, th):
        print("Adding safepoint")
        self.network.setLowerBound(self.network.outputVars[0][0], float(th))
        self.network.setUpperBound(self.network.outputVars[0][0], 1.)
        self.equations.append(Equation(self.bounds["Output"],">=",(float(th))))

    # Applies <= threhshold for counter examples
    def add_counterexample(self, th):
        print("Adding counterexample")
        self.network.setLowerBound(self.network.outputVars[0][0], 0)
        self.network.setUpperBound(self.network.outputVars[0][0], np.nextafter(float(th), -np.inf))
        self.equations.append(Equation(self.bounds["Output"],"<=",np.nextafter(float(th), -np.inf)))