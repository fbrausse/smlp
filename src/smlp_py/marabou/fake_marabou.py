from maraboupy import Marabou
import numpy as np

# %%
# Set the Marabou option to restrict printing
options = Marabou.createOptions(verbosity = 0)

# %%
# Fully-connected network example
# -------------------------------
#
# This network has inputs x0, x1, and was trained to create outputs that approximate
# y0 = abs(x0) + abs(x1), y1 = x0^2 + x1^2
print("Fully Connected Network Example")
filename = "../../test.onnx"
network = Marabou.read_onnx(filename)


# %%
# Set input bounds
network.setLowerBound(0,-10.0)
network.setUpperBound(0, 10.0)
network.setLowerBound(1,-10.0)
network.setUpperBound(1, 10.0)
network.setLowerBound(2,-10.0)
network.setUpperBound(2, 10.0)
network.setLowerBound(3,-10.0)
network.setUpperBound(3, 10.0)
network.setLowerBound(4,-10.0)
network.setUpperBound(4, 10.0)
network.setLowerBound(5,-10.0)
network.setUpperBound(5, 10.0)

# %%
# Call to Marabou solver
exitCode, vals, stats = network.solve(options = options)