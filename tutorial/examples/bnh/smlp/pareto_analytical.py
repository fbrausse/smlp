#!/usr/bin/env python3.11
import numpy as np

# Generate analytical Pareto front in decision space
# Segment 1: x1 = x2 for x in [0, 3]
x1_seg1 = [5*(1-w)/(w*4+(1-w)) for w in [1,0.8,0.6,0.4,0.2]]
x2_seg1 = x1_seg1

# Segment 2: x1 in [3, 5], x2 = 3
x1_seg2 = [3,5]
x2_seg2 = np.full_like(x1_seg2, 3)

# Combine segments for decision space
x1_pareto = np.concatenate([x1_seg1, x1_seg2])
x2_pareto = np.concatenate([x2_seg1, x2_seg2])

# Calculate corresponding objective values
# f1 = 4*x1^2 + 4*x2^2
# f2 = (x1-5)^2 + (x2-5)^2
f1_pareto = 4 * x1_pareto**2 + 4 * x2_pareto**2
f2_pareto = (x1_pareto - 5)**2 + (x2_pareto - 5)**2

print("X1,X2,F1,F2")
[print(f"{x1_pareto[i]:.6f},{x2_pareto[i]:.6f},{f1_pareto[i]:.6f},{f2_pareto[i]:.6f}") for i in range(len(x1_pareto))]
