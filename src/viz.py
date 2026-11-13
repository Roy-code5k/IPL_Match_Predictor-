"""Shared matplotlib style for notebooks and reports.

Palette: validated categorical slots (fixed order, never cycled), a
blue<->red diverging map with a neutral gray midpoint, recessive grid and
axes, and text in ink colours rather than series colours.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
NEUTRAL_MARK = "#c3c2b7"  # de-emphasised marks (e.g. defunct teams)

# Categorical slots in fixed order. Charts with 4+ series that overlap
# (scatter, etc.) must fold extras into "Other" (see dataviz guidance).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, AQUA = SERIES[:3]

# Diverging: below-midpoint red, midpoint gray, above-midpoint blue.
DIVERGING = LinearSegmentedColormap.from_list(
    "red_gray_blue", ["#b8302f", "#e34948", "#f0efec", "#2a78d6", "#1c5cab"]
)


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlecolor": INK,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "legend.frameon": False,
        "legend.labelcolor": INK_SECONDARY,
    })


def percent_axis(ax, axis: str = "y") -> None:
    fmt = mpl.ticker.PercentFormatter(xmax=1, decimals=0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def reference_line(ax, value: float = 0.5, axis: str = "y", label: str | None = "50%") -> None:
    """Dashed neutral reference (e.g. a coin-flip 50%)."""
    draw = ax.axhline if axis == "y" else ax.axvline
    draw(value, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    if label:
        if axis == "y":
            ax.annotate(label, xy=(1, value), xycoords=("axes fraction", "data"),
                        xytext=(4, 0), textcoords="offset points", va="center",
                        color=INK_MUTED, fontsize=9)
        else:
            ax.annotate(label, xy=(value, 1), xycoords=("data", "axes fraction"),
                        xytext=(0, 4), textcoords="offset points", ha="center",
                        color=INK_MUTED, fontsize=9)


def save(fig, name: str, folder) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png")
    plt.show()
