import sys
sys.path.append('/home/Desktop/Marabou/maraboupy')
from maraboupy import Marabou, MarabouCore
from maraboupy.MarabouCore import *
import numpy as np

import os


class ONNXNetwork:
    def __init__(self):
        filename = "../../test.onnx"
        # filename = "test.onnx"
        self.network = Marabou.read_onnx(filename)

    def beta(self):
        # self.network.setLowerBound(4, 4)
        # self.network.setUpperBound(4, 10)
        #
        # self.network.setLowerBound(5, 8)
        # self.network.setUpperBound(5, 20)

        # BEST SOLUTION
        self.network.setLowerBound(4, 0.24)
        self.network.setUpperBound(4, 10.7007)

        self.network.setLowerBound(5, 1.12)
        self.network.setUpperBound(5, 12.02)

    def alpha(self):
        #     p2<5 and x1==10 and x2<12
        # (p2≥5)∨(x1#10)∨(x2≥12)

        epsilon = 1e-12

        eq1 = MarabouCore.Equation(MarabouCore.Equation.GE)
        eq1.addAddend(1, 3)
        eq1.setScalar(5)

        eq2 = MarabouCore.Equation(MarabouCore.Equation.GE)
        eq2.addAddend(1, 1)
        eq2.setScalar(12)

        eq3 = MarabouCore.Equation(MarabouCore.Equation.GE)
        eq3.addAddend(1, 0)
        eq3.setScalar(10+epsilon)

        eq4 = MarabouCore.Equation(MarabouCore.Equation.LE)
        eq4.addAddend(1, 0)
        eq4.setScalar(10 - 1e-12)

        self.network.addDisjunctionConstraint([[eq1], [eq2], [eq3], [eq4]])

    def add_bounds(self, var, bounds, num="real", grid=None):
        lower, upper = bounds
        self.network.setLowerBound(var, lower)
        self.network.setUpperBound(var, upper)

        if num == "in":
            disjunction = []

            for i in range(lower, upper+1):
                eq1 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                eq1.addAddend(1, var)
                eq1.setScalar(i)
                disjunction.append([eq1])

            self.network.addDisjunctionConstraint(disjunction)

            if grid is not None:
                disjunction = []

                for num in grid:
                    eq1 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                    eq1.addAddend(1, var)
                    eq1.setScalar(num)
                    disjunction.append([eq1])

                self.network.addDisjunctionConstraint(disjunction)


    def run_marabou(self):
        options = Marabou.createOptions(verbosity = 10)

        grid = [2, 4, 7]
        for var in self.network.inputVars[0][0]:
            # if var == 0:
            #     self.add_bounds(var, (0, 10))
            # elif var == 1:
            #     self.add_bounds(var, (-1, 1), num="int")
            # elif var == 2:
            #     self.add_bounds(var, (0, 10), num="int", grid=grid)
            # elif var == 3:
            #     self.add_bounds(var, (3, 7), num="int")

            # BEST SOLUTION
            if var == 0:
                self.add_bounds(var, (-0.8218, 9.546))
            elif var == 1:
                self.add_bounds(var, (-1, 1), num="int")
            elif var == 2:
                self.add_bounds(var, (0.1, 10), num="int", grid=grid)
            elif var == 3:
                self.add_bounds(var, (3, 7), num="int")

        # self.alpha()
        self.beta()

        exitCode, vals, stats = self.network.solve(options = options)

        # Test Marabou equations against onnxruntime at an example input point
        # inputPoint = np.ones(inputVars.shape)
        # marabouEval = network.evaluateWithMarabou([inputPoint], options = options)[0]
        # onnxEval = network.evaluateWithoutMarabou([inputPoint])[0]
        print(exitCode, vals, stats)
# ONNXNetwork().run_marabou()

# if __name__ == "__main__":
#         onnx_file = "/home/ntinouldinho/Desktop/Marabou/data/test.onnx"
#         # property_filename = "/home/ntinouldinho/Desktop/Marabou/data/model_constraints.vnnlib"
#         # onnx_file = "/home/ntinouldinho/Desktop/smlp/src/test.onnx"
#         property_filename = "/home/ntinouldinho/Desktop/smlp/src/query.vnnlib"
#
#         network = Marabou.read_onnx(onnx_file)
#         network.saveQuery("./query.txt")
#
#         try:
#             ipq = Marabou.load_query("./query.txt")
#             # MarabouCore.loadProperty(ipq, property_filename)
#             exitCode_ipq, vals_ipq, _ = Marabou.solve_query(ipq, propertyFilename=property_filename, filename="res.log")
#             print(exitCode_ipq, vals_ipq)
#
#         except Exception as e:
#             print(e)

if __name__ == "__main__":
    network = Marabou.read_tf("/home/ntinouldinho/Desktop/smlp/result/abc_smlp_toy_basic_model_checkpoint.h5")




