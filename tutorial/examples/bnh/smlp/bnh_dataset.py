#!/usr/bin/env python3.11
"""
Binh and Korn (BNH) Multi-Objective Optimization Problem
Reference: https://web.archive.org/web/20190801183649/https://pdfs.semanticscholar.org/cf68/41a6848ca2023342519b0e0e536b88bdea1d.pdf, Test Case 2 on page 6
Problem Definition:
-------------------
Minimize:
    f₁(x) = 4x₁² + 4x₂²
    f₂(x) = (x₁ - 5)² + (x₂ - 5)²

Subject to:
    C₁(x) = (x₁ - 5)² + x₂² ≤ 25
    C₂(x) = (x₁ - 8)² + (x₂ + 3)² ≥ 7.7
    0 ≤ x₁ ≤ 5
    0 ≤ x₂ ≤ 3

Characteristics:
----------------
- 2 objectives (minimize both)
- 2 constraints
- 2 decision variables
- Constraints do not make any solution in the unconstrained Pareto-optimal 
  front infeasible, but introduce additional difficulty in solving
"""

from numpy import linspace
from gzip import open as gzopen

def main():
    n_samples = 101  # Reduced from 1000
    x1_range = linspace(0, 5, n_samples)
    x2_range = linspace(0, 3, n_samples)

    # Create gzipped dataset    
    with gzopen("bnh.csv.gz", "wt") as f_data:
        f_data.write("X1,X2,F1,F2\n")
        for x1 in x1_range:
            for x2 in x2_range:
                F1 = 4*x1**2 + 4*x2**2
                F2 = (x1-5)**2 + (x2-5)**2
                f_data.write(f"{x1},{x2},{F1},{F2}\n")

if __name__ == "__main__":
    main()
