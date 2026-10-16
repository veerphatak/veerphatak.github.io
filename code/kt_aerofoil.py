"""
Karman-Trefftz aerofoil geometry, and how it responds to its parameters
=======================================================================

Veer Phatak, Advanced Aeronautics (SESA3025).

WHAT THIS IS
------------
A Karman-Trefftz aerofoil is produced by conformally mapping a circle in the
complex plane onto an aerofoil section. Three parameters control it: a
thickness factor, a camber factor, and the trailing-edge angle. This script
generates the geometry, sweeps each parameter independently, and extracts the
resulting maximum thickness and camber together with their chordwise positions.

My assigned case was thickness factor eps = 0.07, camber beta = 0.07, and
trailing-edge angle tau = 0.05.

THE TRANSFORM
-------------
Take a circle through the point zeta = 1, with its centre offset to
mu = -eps + i*beta. Offsetting along the real axis produces thickness;
offsetting along the imaginary axis produces camber. Then apply

    n = 2 - tau/pi

    z = n * [ (zeta+1)^n + (zeta-1)^n ] / [ (zeta+1)^n - (zeta-1)^n ]

With n = 2 exactly (tau = 0) this reduces to the Joukowski transform, which
gives a cusped trailing edge of zero angle. The Karman-Trefftz generalisation
exists precisely to avoid that: a real aerofoil has a finite wedge angle at the
trailing edge, and a cusp is both unbuildable and badly behaved in a panel
method.

WHY THE FIGURES LOOK LIKE THIS
------------------------------
Each figure pairs the scalar response (how the extracted quantity moves with
the parameter) against the geometry itself (what that actually looks like).
The scalar panel alone invites you to trust a straight line without checking it
is describing a sensible shape; the shapes alone make it hard to see that the
relationship is close to linear. Together they answer "how much" and "so what".

Sections are drawn with one hue stepped by lightness rather than in separate
colours, because the sweep is ordered. Separate hues would imply the cases are
unrelated categories.

USAGE
-----
    python kt_aerofoil.py
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from site_palette import P, accent, use_site_style, shades, finish

use_site_style()

HERE = Path(__file__).resolve().parent
FIGS = HERE.parent / "media" / "figures"

EPS, BETA, TAU = 0.07, 0.07, 0.05       # the assigned case
N_PTS = 600


def karman_trefftz(eps=EPS, beta=BETA, tau=TAU, n_pts=N_PTS):
    """
    Return (x, y) for the aerofoil, normalised to unit chord with the leading
    edge at x = 0.
    """
    n = 2.0 - tau / np.pi
    mu = complex(-eps, beta)
    R = abs(1.0 - mu)                    # circle passes through zeta = 1

    theta = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=True)
    zeta = mu + R * np.exp(1j * theta)

    a = (zeta + 1.0) ** n
    b = (zeta - 1.0) ** n
    z = n * (a + b) / (a - b)

    x, y = z.real, z.imag
    chord = x.max() - x.min()
    return (x - x.min()) / chord, y / chord


def camber_and_thickness(x, y):
    """
    Split the section into upper and lower surfaces on a common chordwise grid,
    then return the camber line, the thickness distribution, and the grid.

    Done by interpolating each surface separately rather than by pairing raw
    points, because the mapping does not return points at matching x on the two
    surfaces, and pairing them by index would silently shear the result.
    """
    i_le = int(np.argmin(x))
    xs = np.linspace(0.0, 1.0, 400)

    def surf(xa, ya):
        o = np.argsort(xa)
        return np.interp(xs, xa[o], ya[o])

    s1 = surf(x[:i_le + 1], y[:i_le + 1])
    s2 = surf(x[i_le:], y[i_le:])
    upper, lower = np.maximum(s1, s2), np.minimum(s1, s2)
    return xs, 0.5 * (upper + lower), upper - lower


def sweep(param, values):
    """Extract the peak quantity and its chordwise position across a sweep."""
    out = []
    for v in values:
        kw = {param: v}
        x, y = karman_trefftz(**kw)
        xs, camber, thick = camber_and_thickness(x, y)
        out.append({
            "value": v,
            "max_thickness": thick.max(),
            "x_max_thickness": xs[int(np.argmax(thick))],
            "max_camber": camber.max(),
            "x_max_camber": xs[int(np.argmax(camber))],
            "xy": (x, y),
        })
    return out


def figure(data, sym, param_label, scalar_key, pos_key, scalar_label, pos_label,
           titles, fname, show_camber=False):
    """
    Three panels: the geometry across the top at true aspect, and the two
    extracted quantities beneath it.

    The geometry gets the full width because it is drawn at equal aspect. An
    aerofoil plotted with a stretched vertical axis looks like a different and
    much thicker section than it is, which is the one thing a geometry figure
    must not do, so the panel is shaped to fit the aerofoil rather than the
    aerofoil squeezed to fit the panel.
    """
    cols = shades("coursework", len(data))
    fig = plt.figure(figsize=(12.2, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], hspace=0.55, wspace=0.26)

    # ---- top: the sections themselves ----
    ax = fig.add_subplot(gs[0, :])
    for r, c in zip(data, cols):
        x, y = r["xy"]
        ax.plot(x, y, color=c, lw=1.4)
        if show_camber:
            xs, cam, _ = camber_and_thickness(x, y)
            ax.plot(xs, cam, color=c, lw=0.9, ls="--")
    ax.set_aspect("equal")
    ax.set_xlabel("x / c")
    ax.set_ylabel("y / c")
    ax.set_title(titles[0])
    ax.set_xlim(-0.03, 1.03)
    finish(ax, legend=False)

    # Direct labels on the extremes beat a six-entry legend for an ordered
    # sweep: the reader only needs to know which end is which, and position
    # already says which curve is which. Labels wear the text colour rather
    # than the series colour, so they stay legible at the dim end of the ramp.
    x0, y0 = data[0]["xy"]
    x1, y1 = data[-1]["xy"]
    ax.margins(y=0.42)
    ax.annotate(f"{sym} = {data[-1]['value']:.2f}",
                (x1[np.argmax(y1)], y1.max()),
                textcoords="offset points", xytext=(0, 9),
                color=P["body"], fontsize=8.5, ha="center")
    ax.annotate(f"{sym} = {data[0]['value']:.2f}",
                (x0[np.argmin(y0)], y0.min()),
                textcoords="offset points", xytext=(0, -14),
                color=P["muted"], fontsize=8.5, ha="center")
    if show_camber:
        i = int(np.argmin(np.abs(x1 - 0.78)))
        ax.annotate("camber lines dashed", (0.78, y1[i]),
                    textcoords="offset points", xytext=(24, -20),
                    color=P["muted"], fontsize=8, ha="left")

    xs_ = np.array([r["value"] for r in data])

    # ---- bottom left: the peak value ----
    ax = fig.add_subplot(gs[1, 0])
    ys_ = np.array([r[scalar_key] for r in data])
    ax.plot(xs_, ys_, "-o", color=accent("coursework"), lw=1.8, ms=5)
    ax.set_xlabel(param_label); ax.set_ylabel(scalar_label)
    ax.set_title(titles[1])
    finish(ax, legend=False)

    # Quantify the departure from a straight line instead of drawing a dotted
    # line underneath that the data hides anyway.
    fit = np.interp(xs_, [xs_[0], xs_[-1]], [ys_[0], ys_[-1]])
    span = ys_.max() - ys_.min()
    dev = 100.0 * np.abs(ys_ - fit).max() / span if span else 0.0
    ax.text(0.03, 0.95, f"within {dev:.1f}% of linear", transform=ax.transAxes,
            color=P["muted"], fontsize=8.5, va="top")

    # ---- bottom right: where the peak sits along the chord ----
    ax = fig.add_subplot(gs[1, 1])
    ps_ = np.array([r[pos_key] for r in data])
    ax.plot(xs_, ps_, "-o", color=accent("coursework"), lw=1.8, ms=5)
    ax.set_xlabel(param_label); ax.set_ylabel(pos_label)
    ax.set_title(titles[2])
    ax.set_ylim(0, 1)
    finish(ax, legend=False)

    fig.savefig(FIGS / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {fname}")


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    print("Karman-Trefftz geometry")

    x, y = karman_trefftz()
    xs, cam, thick = camber_and_thickness(x, y)
    print(f"  assigned case eps={EPS}, beta={BETA}, tau={TAU}")
    print(f"    max thickness {100*thick.max():.2f}% c at x/c = {xs[np.argmax(thick)]:.3f}")
    print(f"    max camber    {100*cam.max():.2f}% c at x/c = {xs[np.argmax(cam)]:.3f}")
    print(f"    closest standard section: NACA "
          f"{round(100*cam.max()):.0f}{round(10*xs[int(np.argmax(cam))]):.0f}"
          f"{round(100*thick.max()):02.0f}")

    vals = np.linspace(0.0, 0.10, 6)

    figure(sweep("eps", vals),
           sym="$\\epsilon$", param_label="Thickness parameter, $\\epsilon$",
           scalar_key="max_thickness", pos_key="x_max_thickness",
           scalar_label="Maximum thickness, $t/c$",
           pos_label="Position of max thickness, $x/c$",
           titles=("Section shape as $\\epsilon$ increases, 0 to 0.10 in six steps",
                   "Maximum thickness against $\\epsilon$",
                   "Where the maximum sits along the chord"),
           fname="kt-thickness-sweep.png")

    figure(sweep("beta", vals),
           sym="$\\beta$", param_label="Camber parameter, $\\beta$",
           scalar_key="max_camber", pos_key="x_max_camber",
           scalar_label="Maximum camber, $y_c/c$",
           pos_label="Position of max camber, $x/c$",
           titles=("Section shape as $\\beta$ increases, 0 to 0.10 in six steps",
                   "Maximum camber against $\\beta$",
                   "Where the maximum sits along the chord"),
           fname="kt-camber-sweep.png", show_camber=True)


if __name__ == "__main__":
    main()
