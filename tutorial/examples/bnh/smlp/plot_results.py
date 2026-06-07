#!/usr/bin/env python3.11
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sys import argv
from math import inf

# Read the data files
X1 = pd.read_csv('bnh_pareto_X1.txt', header=None, names=['X1'])
X2 = pd.read_csv('bnh_pareto_X2.txt', header=None, names=['X2'])
F1 = pd.read_csv('bnh_pareto_F1.txt', header=None, names=['F1'])
F2 = pd.read_csv('bnh_pareto_F2.txt', header=None, names=['F2'])

# Combine X1 and X2
df_X = pd.concat([X1, X2], axis=1)

# Combine F1 and F2
df_F = pd.concat([F1, F2], axis=1)

# Generate analytical Pareto front in decision space
# Segment 1: x1 = x2 for x in [0, 3]
x1_seg1 = np.linspace(0, 3, 100)
x2_seg1 = x1_seg1

# Segment 2: x1 in [3, 5], x2 = 3
x1_seg2 = np.linspace(3, 5, 100)
x2_seg2 = np.full_like(x1_seg2, 3)

# Combine segments for decision space
x1_pareto = np.concatenate([x1_seg1, x1_seg2])
x2_pareto = np.concatenate([x2_seg1, x2_seg2])

# Calculate corresponding objective values
# f1 = 4*x1^2 + 4*x2^2
# f2 = (x1-5)^2 + (x2-5)^2
f1_pareto = 4 * x1_pareto**2 + 4 * x2_pareto**2
f2_pareto = (x1_pareto - 5)**2 + (x2_pareto - 5)**2

# Create two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Plot X1 vs X2
sns.scatterplot(data=df_X, x='X1', y='X2', ax=ax1, label='Computed Solutions')
ax1.plot(x1_pareto, x2_pareto, 'r-', linewidth=2, label='Analytical Pareto Front')
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
ax1.set_title('Pareto Front in Decision Space')
ax1.grid(True)
ax1.legend()

# Plot F1 vs F2
sns.scatterplot(data=df_F, x='F1', y='F2', ax=ax2, label='Computed Solutions')
ax2.plot(f1_pareto, f2_pareto, 'r-', linewidth=2, label='Analytical Pareto Front')
ax2.set_xlabel('F1')
ax2.set_ylabel('F2')
ax2.set_title('Pareto Front in Objective Space')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
timeout = inf
if len(argv) > 2:
    if '-timeout' == argv[1]:
        timeout = int(argv[2]) 
if not inf == timeout:
    timer = fig.canvas.new_timer(interval=timeout*1000, callbacks=[(plt.close, [], {})])
    timer.start()
plt.show()
