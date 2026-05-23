#!/usr/bin/env python3.11
"""
Witness Certification Visualizations
Produces  witness_geometry.png  – matplotlib geometry plot
"""

import math
import textwrap
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from math import inf, sqrt
from sys import argv

# ── shared palette ───────────────────────────────────────────────────────────
C_BLUE       = "#378ADD"
C_GREEN      = "#1D9E75"
C_LIGHTGREEN = "#90EE90"
C_RED        = "#E24B4A"
C_NAVYBLUE   = "#000080"
C_GRAY       = "#888780"
C_MAGENTA    = "#FF00FF"
C_DARKGREEN  = "#006400"

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ════════════════════════════════════════════════════════════════════════════
# Figure 1 – Geometry (matplotlib → PNG)
# ════════════════════════════════════════════════════════════════════════════
def plot_geometry(ax):
    ax.set_aspect("equal")
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("x₁", fontsize=13)
    ax.set_ylabel("x₂", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.set_title("SMLP Witness certification\n"
                 r"$f_1(x)=4x_1^2+4x_2^2$" + "\n" + r"query: y = $(f_1-0.692042)^2<4$ and 0 ≤ $x_1 ≤ 5$ and 0 ≤ $x_2 ≤ 3$",
                 fontsize=15, pad=10)

    # ── constraint boundary |x| = 0.580091 (quarter circle) ──────────────────
    R_constraint = 0.580091

    wedge = mpatches.Wedge((0,0), R_constraint, 0, 90, facecolor=C_LIGHTGREEN, edgecolor=C_BLUE, linestyle='--')
    ax.add_patch(wedge)
    ax.text(0.05, R_constraint - 0.05, "y=4", fontsize=12, color=C_BLUE, va="center")

    # ── Arc for y = 9 ────────────────────────────────────────────────────
    x0 = 0.427793
    px = py = x0
    arc0 = mpatches.Arc((0,0), px*2*sqrt(2), py*2*sqrt(2), angle=0, theta1=0, theta2=90, color=C_RED, linestyle='--')
    ax.add_patch(arc0)
    ax.text(0.05, py*sqrt(2) + 0.04, "y=5", fontsize=12, color=C_RED, va="center")

    # ── Arc for y = 0 ────────────────────────────────────────────────────
    x0 = 0.294118
    px = py = x0
    arc0 = mpatches.Arc((0,0), px*2*sqrt(2), py*2*sqrt(2), angle=0, theta1=0, theta2=90, color=C_DARKGREEN, linestyle='--')
    ax.add_patch(arc0)
    ax.text(0.05, py*sqrt(2) - 0.05, "y=0", fontsize=12, color=C_NAVYBLUE, va="center")

    # ── Witness point ────────────────────────────────────────────────────
    px = py = x0
    ax.plot(px, py, "o", color=C_NAVYBLUE, ms=9, zorder=5)

    # ── PASS square: half-side = 0.285, full side = 0.570 ───────────────────
    hs_pass = 0.285
    ax.add_patch(mpatches.Rectangle(
        (px - hs_pass, py - hs_pass), 2 * hs_pass, 2 * hs_pass,
        facecolor=C_MAGENTA, alpha=0.14, linewidth=0, zorder=2))
    ax.add_patch(mpatches.Rectangle(
        (px - hs_pass, py - hs_pass), 2 * hs_pass, 2 * hs_pass,
        facecolor="none", edgecolor=C_GREEN, lw=1.8, zorder=3))
    ax.text(px + hs_pass - 0.175, py + 0.19,
            f"side={2*hs_pass:.3f}\n  PASS ✓ →", fontsize=12,
            color=C_GREEN, va="center")

    # ── FAIL square: half-side = 0.286, full side = 0.572 ───────────────────
    hs_fail = 0.286
    ax.add_patch(mpatches.Rectangle(
        (px - hs_fail, py - hs_fail), 2 * hs_fail, 2 * hs_fail,
        facecolor="none", edgecolor=C_RED, lw=1.8, ls="--", zorder=3))
    ax.text(px + hs_fail + 0.01, py - 0.08,
            f"side={2*hs_fail:.3f}\n← FAIL ✗", fontsize=12,
            color=C_RED, va="center")

    # ── half-side annotation ──────────────────────────────────────────────────
    bd = R_constraint - px
    ax.annotate("", xy=(px + hs_fail + 0.005, py), xytext=(px, py),
                arrowprops=dict(arrowstyle="<->", color=C_RED, lw=1.5))
    ax.text(px + hs_fail / 2 - 0.06, py - 0.05, f"rad-abs = {hs_fail}",
            fontsize=14, color=C_RED, ha="center")

    handles = [
        Line2D([0], [0], color=C_BLUE, lw=1.5, ls="--",
               label=r"$|x|= √(x_1^2+x_2^2)$" + f"={R_constraint}, y = 4 - constraint boundary"),
        mpatches.Patch(facecolor=C_GREEN, alpha=0.65,
                       label=f"SMLP query is TRUE"),
        mpatches.Patch(facecolor=C_MAGENTA, alpha=0.35,
                       label=f"certify PASS square side=2*{hs_pass}={2*hs_pass:.3f}"),
        mpatches.Patch(facecolor="none", edgecolor=C_RED, lw=1.5,
                       linestyle="--", label=f"certify FAIL  square side=2*{hs_fail}={2*hs_fail:.3f}"),
        Line2D([0], [0], marker="o", color=C_NAVYBLUE, lw=0,
               markersize=8, label=f"Witness point ({x0}, {x0}), y=0"),
    ]
    legend  = ax.legend(handles=handles, fontsize=11, loc="upper right", framealpha=0.85)
    ax.set_facecolor("#FAFAF8")
    legend.get_texts()[0].set_color(C_BLUE)
    legend.get_texts()[1].set_color(C_GREEN)
    legend.get_texts()[2].set_color(C_MAGENTA)
    legend.get_texts()[3].set_color(C_RED)
    legend.get_texts()[4].set_color(C_NAVYBLUE)


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def main(timeout: float = inf):
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.subplots(1, 1)

    plot_geometry(ax1)
    #plot_pipeline(ax2)
    fig.tight_layout()

    fig.savefig("witness_certify.png",
                dpi=150, bbox_inches="tight")
    print("Saved witness_certify.png")
    
    if not inf == timeout:
        timer = fig.canvas.new_timer(interval=int(timeout)*1000, callbacks=[(plt.close, [], {})])
        timer.start()
    plt.show()

if __name__ == "__main__":
    timeout = inf
    if len(argv) > 2:
        if '-timeout' == argv[1]:
            timeout = float(argv[2])
    main(timeout)
