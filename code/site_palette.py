"""
Figure styling that matches the website.

Every figure on the site used to be generated on a white background in a
palette that appeared nowhere else on the page, which meant each chart punched
a bright rectangle into a dark page in colours the page never used. This module
is the fix: one palette, defined once in `site-palette.json`, read by both the
Python scripts and the MATLAB ones, so a figure sits on the same surface as the
page it lands in.

Usage
-----
    from site_palette import P, use_site_style, accent

    use_site_style()                 # call once, before creating figures
    fig, ax = plt.subplots()
    ax.plot(x, measured, color=P["reference"], label="Measured")
    ax.plot(x, model,    color=accent("f1"), label="Model")
    finish(ax)                       # tidy spines, grid and legend

Design rules this encodes
-------------------------
- Measured or ground-truth data takes the neutral; the thing being argued for
  takes the page accent. One series is the point, the other is context.
- Categorical series are assigned from `series` in fixed order, never cycled.
- Grid and axes recede. They are navigation, not data.
- Text wears text colours, never the series colour.
"""

from pathlib import Path
import json

import matplotlib as mpl
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
P = json.loads((_HERE / "site-palette.json").read_text(encoding="utf-8"))


def accent(name="site"):
    """Accent for a given page, e.g. accent('f1')."""
    return P["accents"].get(name, P["accents"]["site"])


def series(i):
    """Categorical slot i (0-based), in fixed order. Never cycles silently."""
    s = P["series"]
    if i >= len(s):
        raise IndexError(
            f"categorical slot {i} requested but only {len(s)} are defined. "
            "Fold the tail into 'Other' or use small multiples rather than "
            "inventing a new hue, which would be indistinguishable under CVD."
        )
    return s[i]


def use_site_style():
    """Apply the site's figure style globally."""
    mpl.rcParams.update({
        "figure.facecolor":  P["surface"],
        "savefig.facecolor": P["surface"],
        "axes.facecolor":    P["surface"],
        "savefig.edgecolor": "none",

        "text.color":       P["body"],
        "axes.labelcolor":  P["muted"],
        "axes.titlecolor":  P["text"],
        "xtick.color":      P["muted"],
        "ytick.color":      P["muted"],

        "axes.edgecolor":   P["border"],
        "axes.linewidth":   1.0,
        "axes.grid":        True,
        "grid.color":       P["border"],
        "grid.alpha":       0.9,
        "grid.linewidth":   0.8,
        "axes.axisbelow":   True,

        "axes.spines.top":   False,
        "axes.spines.right": False,

        "legend.facecolor":  P["sunken"],
        "legend.edgecolor":  P["border"],
        "legend.framealpha": 1.0,
        "legend.labelcolor": P["body"],

        "axes.prop_cycle": mpl.cycler(color=P["series"]),

        "font.family":     "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size":       10,
        "axes.titlesize":  11.5,
        "axes.labelsize":  10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,

        "figure.dpi": 150,
    })


def finish(ax, legend=True, loc="best"):
    """Tidy one axes: recede the spines, keep the legend quiet."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(P["border"])
    if legend and ax.get_legend_handles_labels()[0]:
        lg = ax.legend(loc=loc)
        lg.get_frame().set_linewidth(1.0)
    return ax


def cmap_accent(name="site", n=256):
    """
    Sequential ramp from a dim step to the page accent.

    One hue for magnitude. Never a rainbow: a rainbow implies category
    boundaries the data does not have and reverses under CVD.

    The low end anchors on the border step rather than on the sunken surface.
    Anchoring below the page surface makes the smallest values read as holes
    punched in the page instead of as data, which is wrong whenever low means
    "slow" rather than "absent". This way low values recede without vanishing.
    """
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        f"site_{name}", [P["border"], accent(name)], N=n)


def cmap_diverging(n=256):
    """
    Two hues with a neutral midpoint, for signed quantities like model error.

    Matplotlib's `coolwarm` is the usual reach here and is wrong on a dark
    surface: its midpoint is near-white, so zero error reads as the brightest
    value on the plot and the eye lands on exactly the places where the model
    is correct. This ramp puts the page surface at the midpoint instead, so
    zero recedes and both kinds of error advance.
    """
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "site_div",
        [P["accents"]["site"], P["surface"], P["accents"]["f1"]],
        N=n)


def shades(name="site", k=4):
    """
    k steps of one hue, for an ordered sweep (e.g. four downforce levels).

    Ordered data gets one hue stepped by lightness, not k unrelated hues.
    """
    import numpy as np
    cm = cmap_accent(name)
    return [mpl.colors.to_hex(cm(v)) for v in np.linspace(0.42, 1.0, k)]
