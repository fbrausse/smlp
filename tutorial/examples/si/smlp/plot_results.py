#!/usr/bin/env python3
import os
import platform
import shutil
import subprocess
import tempfile
import os.path

cache_root = os.path.join(tempfile.gettempdir(), "smlp-cache")
os.environ.setdefault("XDG_CACHE_HOME", cache_root)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(cache_root, "matplotlib"))

import matplotlib

IS_MACOS = platform.system() == "Darwin"

# define IS_AARCH64

try:
    with open('/proc/sys/kernel/arch') as f:        
        if 'aarch64' in f.read():
            IS_AARCH64= True
        else:
            IS_AARCH64= False
except OSError:
    IS_AARCH64= False
        
try:
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_TK = True
    TK_IMPORT_ERROR = None
except ImportError as exc:
    tk = None
    FigureCanvasTkAgg = None
    HAS_TK = False
    TK_IMPORT_ERROR = exc
    
# switch off Tk if aarch64 docker (used in MacOS)

if IS_AARCH64 and os.path.isfile("/.dockerenv"):
    tk = None
    FigureCanvasTkAgg = None
    HAS_TK = False
    
# Configure the backend before importing pyplot or seaborn. Use TkAgg when Tk
# is present because this script embeds the figure in a Tk window; otherwise
# fall back to Agg so pyenv/macOS Python builds without _tkinter can still save
# and open the PNG.
if HAS_TK:
    matplotlib.use("TkAgg")
else:
    matplotlib.use("Agg")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
from math import inf
from sys import argv
from os.path import basename, realpath
from pathlib import Path

# ── 1. Load data ──────────────────────────────────────────────────────────────
rows = []
if len(argv) < 2:
    print(f"\nUsage {basename(realpath(argv[0]))} <relative_optimized_margin_sorted> [-timeout <seconds>]\n")  
    exit(0)
file_path = argv[1]
with open(file_path) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            label, value = parts[0], float(parts[1])
            m = re.match(r"o(\d+)_CH(\d+)RANK(\d+)Byte(\d+)", label)
            if m:
                o, ch, rank, byte = m.groups()
                rows.append({
                    "label": label,
                    "value": value,
                    "Octet": f"O{o}",
                    "Channel": f"CH{ch}",
                    "Rank": f"RANK{rank}",
                    "Byte": int(byte),
                    "group": f"O{o} CH{ch} RANK{rank}",
                    "ch_rank_byte_group": f"CH{ch} RANK{rank} Byte{byte}",
                    "_ch": int(ch),
                    "_rank": int(rank),
                    "_byte": int(byte),
                })

# ── df_value: sorted by value (for right chart) ───────────────────────────────
df_value = pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)

# ── df_grouped: sorted by O/CH/RANK group then Byte (left chart) ──────────────
df_grouped = pd.DataFrame(rows).sort_values(
    ["group", "_byte"], ascending=True
).reset_index(drop=True)
df_grouped["index"] = df_grouped.index

# ── 2. Colour palette — shared by both charts, keyed on group (O CH RANK) ────
groups = sorted(df_value["group"].unique())
palette = sns.color_palette("tab10", len(groups))
group_color = dict(zip(groups, palette))

# ── 3. Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    1, 2,
    figsize=(18, 9),
    gridspec_kw={"width_ratios": [2, 1]},
)
fig.patch.set_facecolor("#f8f9fb")

# ── LEFT: horizontal bar chart grouped by O/CH/RANK ───────────────────────────
ax = axes[0]
ax.set_facecolor("#f0f2f5")

colors = [group_color[g] for g in df_grouped["group"]]
bars = ax.barh(
    df_grouped["index"], df_grouped["value"],
    color=colors, edgecolor="white", linewidth=0.4, height=0.85
)

# Add value labels on bars
for bar, val in zip(bars, df_grouped["value"]):
    x = bar.get_width()
    ax.text(
        x + 0.003, bar.get_y() + bar.get_height() / 2,
        f"{val:.3f}", va="center", ha="left",
        fontsize=6.5, color="#333333"
    )

# Draw separator lines between O/CH/RANK groups
prev_grp = None
for idx, row in df_grouped.iterrows():
    curr_grp = row["group"]
    if prev_grp is not None and curr_grp != prev_grp:
        ax.axhline(idx - 0.5, color="#888888", linewidth=0.7, linestyle="-", alpha=0.5)
    prev_grp = curr_grp

# Threshold reference lines
ax.axvline(0.9, color="#e74c3c", linestyle="--", linewidth=1.2, alpha=0.8, label="Threshold = 0.90")
ax.axvline(1.0, color="#2ecc71", linestyle="--", linewidth=1.2, alpha=0.8, label="Max = 1.00")

ax.set_yticks(df_grouped["index"])
ax.set_yticklabels(df_grouped["label"], fontsize=7)
ax.invert_yaxis()
ax.set_xlim(0, 1.08)
ax.set_xlabel("Relative Optimized Margin", fontsize=11, labelpad=8)
plot_name = Path(file_path).stem.replace('_relative_optimized_margin_sorted', '')
ax.set_title(f"{plot_name} Optimization Results — Grouped by O / CH / RANK", fontsize=13, fontweight="bold", pad=12)
ax.tick_params(axis="x", labelsize=9)
ax.legend(fontsize=9, loc="lower right")
ax.axvspan(0.9, 1.08, alpha=0.06, color="#2ecc71")
ax.xaxis.grid(True, linestyle="--", alpha=0.5, color="white")
ax.set_axisbelow(True)

# ── RIGHT: dot-strip plot by Group (unchanged) ────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor("#f0f2f5")

group_order = sorted(df_value["group"].unique(), reverse=True)
for i, grp in enumerate(group_order):
    sub = df_value[df_value["group"] == grp]
    jitter = np.random.default_rng(42).uniform(-0.18, 0.18, len(sub))
    ax2.scatter(
        sub["value"], [i + j for j in jitter],
        color=group_color[grp], s=55, alpha=0.85,
        edgecolors="white", linewidths=0.5, zorder=3
    )
    mean_val = sub["value"].mean()
    ax2.hlines(i, mean_val - 0.001, mean_val + 0.001, colors=group_color[grp],
               linewidths=0, zorder=2)
    ax2.scatter([mean_val], [i], color=group_color[grp], s=120,
                marker="D", edgecolors="#333", linewidths=0.8, zorder=4)

ax2.set_yticks(range(len(group_order)))
ax2.set_yticklabels(group_order, fontsize=9)
ax2.set_xlabel("Relative Optimized Margin", fontsize=11, labelpad=8)
ax2.set_title("Distribution by Group\n(◆ = mean)", fontsize=12, fontweight="bold", pad=12)
ax2.axvline(0.9, color="#e74c3c", linestyle="--", linewidth=1.2, alpha=0.8)
ax2.axvspan(0.9, 1.08, alpha=0.06, color="#2ecc71")
ax2.set_xlim(0.4, 1.08)
ax2.xaxis.grid(True, linestyle="--", alpha=0.5, color="white")
ax2.set_axisbelow(True)
ax2.tick_params(axis="x", labelsize=9)

# ── Summary stats box ─────────────────────────────────────────────────────────
n_total = len(df_grouped)
n_above = (df_grouped["value"] >= 0.9).sum()
stats_text = (
    f"Total signals: {n_total}\n"
    f"Signals ≥ 0.90: {n_above} ({100*n_above/n_total:.0f}%)\n"
    f"Signals < 0.90: {n_total - n_above} ({100*(n_total-n_above)/n_total:.0f}%)\n"
    f"Min: {df_grouped['value'].min():.4f}   Max: {df_grouped['value'].max():.4f}\n"
    f"Mean: {df_grouped['value'].mean():.4f}   Median: {df_grouped['value'].median():.4f}"
)
fig.text(
    0.5, 0.01, stats_text,
    ha="center", va="bottom", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#dce3ec", edgecolor="#aab4c2", alpha=0.9)
)
plt.tight_layout(rect=[0, 0.08, 1, 1])
png_path = Path(f"{plot_name}.png").resolve()
plt.savefig(png_path, dpi=150, bbox_inches="tight")

# ── Scrollable Tk window ──────────────────────────────────────────────────────
timeout = inf
if len(argv) > 3:
    if '-timeout' == argv[2]:
        timeout = int(argv[3])

if not HAS_TK:
    print(f"Saved plot to {png_path}")
    if IS_MACOS and shutil.which("open") and timeout == inf:
        subprocess.Popen(["open", str(png_path)])
    elif IS_MACOS and timeout != inf:
        print("Skipping macOS Preview auto-open because -timeout was requested.")
    else:
        print(f"Skipping preview: Tk viewer unavailable: {TK_IMPORT_ERROR}")
    exit(0)

root = tk.Tk()
root.title(f"{plot_name} — Optimization Results")

# Size the window within the available screen. This avoids oversized windows
# on smaller MacBook displays while preserving the original Linux dimensions.
screen_w = root.winfo_screenwidth()
screen_h = root.winfo_screenheight()
window_w = min(1280, screen_w)
window_h = min(1024, screen_h)
root.geometry(f"{window_w}x{window_h}")

# Outer frame holds canvas + scrollbars
outer = tk.Frame(root)
outer.pack(fill=tk.BOTH, expand=True)

# Horizontal + vertical scrollbars
hbar = tk.Scrollbar(outer, orient=tk.HORIZONTAL)
hbar.pack(side=tk.BOTTOM, fill=tk.X)
vbar = tk.Scrollbar(outer, orient=tk.VERTICAL)
vbar.pack(side=tk.RIGHT, fill=tk.Y)

# Tk canvas (scroll container)
tk_canvas = tk.Canvas(outer, xscrollcommand=hbar.set, yscrollcommand=vbar.set)
tk_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
hbar.config(command=tk_canvas.xview)
vbar.config(command=tk_canvas.yview)

# Embed matplotlib figure into the Tk canvas
mpl_canvas = FigureCanvasTkAgg(fig, master=tk_canvas)
mpl_widget = mpl_canvas.get_tk_widget()
# Place the widget inside the canvas
fig_w_px = int(fig.get_figwidth() * fig.dpi)
fig_h_px = int(fig.get_figheight() * fig.dpi)
tk_canvas.create_window(0, 0, anchor="nw", window=mpl_widget, width=fig_w_px, height=fig_h_px)
tk_canvas.config(scrollregion=(0, 0, fig_w_px, fig_h_px))

mpl_canvas.draw()

# Mouse-wheel horizontal scroll (Shift+wheel). macOS trackpads usually emit
# small deltas, while X11/Windows wheels typically emit multiples of 120.
def _wheel_units(delta):
    if delta == 0:
        return 0
    if IS_MACOS and abs(delta) < 120:
        return -1 if delta > 0 else 1
    return int(-1 * (delta / 120))

def _on_mousewheel(event):
    tk_canvas.yview_scroll(_wheel_units(event.delta), "units")

def _on_shift_mousewheel(event):
    tk_canvas.xview_scroll(_wheel_units(event.delta), "units")

root.bind_all("<MouseWheel>", _on_mousewheel)
root.bind_all("<Shift-MouseWheel>", _on_shift_mousewheel)
# Linux scroll events
root.bind_all("<Button-4>", lambda e: tk_canvas.yview_scroll(-1, "units"))
root.bind_all("<Button-5>", lambda e: tk_canvas.yview_scroll(1, "units"))
root.bind_all("<Shift-Button-4>", lambda e: tk_canvas.xview_scroll(-1, "units"))
root.bind_all("<Shift-Button-5>", lambda e: tk_canvas.xview_scroll(1, "units"))

def _quit(e=None):
    root.destroy()
    import os; os._exit(0)

root.bind_all("<Escape>", _quit)
root.bind_all("<q>", _quit)
if IS_MACOS:
    root.bind_all("<Command-q>", _quit)
    root.createcommand("tk::mac::Quit", _quit)

if not inf == timeout:
    root.after(int(timeout * 1000), _quit)

root.mainloop()
