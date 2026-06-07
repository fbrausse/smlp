#!/usr/bin/env python3.11
"""
Constrained Optimization Example using SMLP
Classic Lagrange Multiplier Problem (appears in many optimization textbooks)

Problem:
    Minimize: f(x1, x2) = (x1 - 2)^2 + (x2 - 1)^2
    Subject to: x1^2 + x2^2 - 1 <= 0  (inside unit circle)
    
    Geometrically: Find the point on the unit circle closest to (2, 1)

Analytical Solution (using Lagrange multipliers):
    Setting ∇f = λ∇g where g(x1,x2) = x1² + x2² - 1:
    - 2(x1-2) = 2λx1  →  x1 = 2/(1+λ)
    - 2(x2-1) = 2λx2  →  x2 = 1/(1+λ)
    - Substituting into constraint: 5 = (1+λ)²
    - Solving: λ = √5 - 1
    
Expected Result:
    Optimal point: x1 = 2/√5 ≈ 0.894427, x2 = 1/√5 ≈ 0.447214
    Minimum value: f = (√5 - 1)² ≈ 1.527864
    
Reference: Wolfram Alpha
https://www.wolframalpha.com/input?i=Minimize%3A+f%28x1%2C+x2%29+%3D+%28x1+-+2%29%5E2+%2B+%28x2+-+1%29%5E2+subject+to+x1%5E2+%2B+x2%5E2+-+1+%3C%3D+0
"""

from sys import argv
from numpy import linspace, meshgrid, cos, sin, pi, sqrt, inf
from pandas import read_csv, concat
from gzip import open as gzopen
from matplotlib import pyplot as plt

def main():
# Initial guess
    rng = range(0, 1000)
    x1_start, x1_stop = (-1.5, 2.5)
    x2_start, x2_stop = (-1.5, 2.0)
    x1 = linspace(x1_start, x1_stop, rng.stop)
    x2 = linspace(x2_start, x2_stop, rng.stop)
    X1, X2 = meshgrid(x1, x2)
    Z = (X1 - 2)**2 + (X2 - 1)**2
    with gzopen('Constraint_dora.csv.gz',"wt") as ds:
        ds.write("X1,X2,Y1\n")
        [[ds.write(f"{X1[i][j]},{X2[i][j]},{Z[i][j]}\n") for j in rng] for i in rng]

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot contours of objective function
    contours = ax.contour(X1, X2, Z, levels=20, cmap='viridis', alpha=0.6)
    ax.clabel(contours, inline=True, fontsize=8)

# Plot constraint boundary (unit circle)
    theta = linspace(0, 2*pi, 100)
    circle_x = cos(theta)
    circle_y = sin(theta)
    ax.plot(circle_x, circle_y, 'r-', linewidth=2, label='Constraint boundary')
    ax.fill(circle_x, circle_y, alpha=0.1, color='red', label='Feasible region')
    
    # Plot unconstrained optimum
    ax.plot(2, 1, 'bs', markersize=12, label='Unconstrained optimum (2, 1)')

    ax.set_xlabel('x1', fontsize=12)
    ax.set_ylabel('x2', fontsize=12)
    ax.set_title('Constrained Optimization: Minimize f(x1,x2) subject to x1² + x2² ≤ 1', 
                 fontsize=14)
    # Plot constrained optimum
    ax.plot(2/sqrt(5), 1/sqrt(5), 'go', markersize=12, label=f'Constrained optimum ({2/sqrt(5):.3f}, {1/sqrt(5):.3f})')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.set_ylim(-1.5, 2.0)

    plt.tight_layout()
    timeout = inf
    if len(argv) > 2:
        if '-timeout' == argv[1]:
            timeout = int(argv[2]) 
    if not inf == timeout:
        timer = fig.canvas.new_timer(interval=timeout*1000, callbacks=[(plt.close, [], {})])
        timer.start()
    plt.show()   

if __name__ == "__main__":
    main()
