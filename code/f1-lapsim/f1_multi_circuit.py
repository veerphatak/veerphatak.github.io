"""
Validating one lap time model across five circuits
==================================================

Veer Phatak, 2026.

WHY THIS EXISTS
---------------
`f1_lap_model.py` predicts the 2025 Bahrain pole lap to within 0.72%. One
circuit cannot distinguish "the physics is right" from "three parameters were
fitted to one curve". A single good number is not validation, it is a fit.

So this script runs the *same solver*, unchanged, around five circuits chosen
to span the aerodynamic and power axis, and asks a harder question than "is
the lap time close".

    Monza        lowest downforce        tests drag and power in isolation
    Barcelona    balanced reference      the neutral case
    Bahrain       traction and braking   the existing baseline
    Suzuka       high-speed aero corners tests the downforce model
    Monaco       max downforce, min speed deliberately the hard case

THE RESULT THAT MATTERS
-----------------------
Not five good error numbers. Two things:

1. A validation table: circuit, predicted lap, actual lap, percentage error,
   speed-trace RMSE.

2. `clA` and `cdA` fitted INDEPENDENTLY per circuit, and how they then vary.
   If the fits come out low at Monza and high at Monaco, tracking real
   aerodynamic configurations the model was never told about, that is
   independent evidence it captures physics rather than curve-fitting one lap.
   That is the actual headline, and it is a much stronger claim than any single
   percentage error.

Monaco is expected to be the worst result and is published anyway. Quasi-steady
breaks down there: low-speed traction limits, kerb riding, elevation change, no
yaw dynamics. A circuit where the model degrades, with the reason stated, is
more credible than five suspiciously good numbers.

A NOTE ON THE DATA SOURCE
-------------------------
The original script used FastF1, which reads Formula 1's live timing API. That
endpoint began returning HTTP 401 to ordinary residential connections during
2026, so sessions that are not already in the local FastF1 cache can no longer
be downloaded through it.

This script therefore reads the same underlying telemetry from OpenF1
(https://openf1.org), a free, unauthenticated mirror of the same feed. Two
streams are fetched per lap, position (x, y at about 3.8 Hz) and car data
(speed, throttle, brake, DRS at about 3.8 Hz), and merged onto their union
timebase, which is exactly what FastF1's `get_telemetry()` does internally and
gives a comparable effective resolution of about 7.5 Hz.

Nothing is redistributed. Raw pulls are cached locally into a directory that is
not committed, and the cache is reused on every later run.

PHYSICS
-------
The car model, the corner-limit calculation, the three-pass solver, the
curvature extraction and the parameter fit are all IMPORTED from
`f1_lap_model.py` rather than reimplemented here. That is deliberate: the claim
is that one model works across five circuits, and that claim is only honest if
it is literally the same code in every case.

USAGE
-----
    python f1_multi_circuit.py                 # all five circuits
    python f1_multi_circuit.py monza monaco    # a subset
    python f1_multi_circuit.py --refresh       # ignore the local cache
"""

from pathlib import Path
import datetime as dt
import json
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.optimize import minimize
from scipy.signal import savgol_filter

# The physics comes from the single-circuit model, unchanged. One source of
# truth: if the solver changes, every circuit here changes with it.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from site_palette import (P, accent, series, use_site_style, cmap_accent,
                          cmap_diverging, shades, finish, legend_above)
use_site_style()

from f1_lap_model import (
    F1Car,
    fit_car,
    solve_lap,
    find_corners,
)

# Tightest corner radius treated as physically real. Monaco's Grand Hotel
# hairpin, the tightest corner in Formula 1, is around 12 m. Anything below
# this is an artifact of differentiating sampled position, not a corner.
R_FLOOR = 12.0

HERE = Path(__file__).parent
RAW = HERE / "cache_openf1"           # gitignored, see .gitignore
FIGS = HERE.parent.parent / "media" / "figures"
DATA = HERE.parent.parent / "data" / "multi"

YEAR = 2025
SESSION = "Qualifying"
DS = 2.0                               # metres between solver points
API = "https://api.openf1.org/v1"

# slug, OpenF1 circuit_short_name, display name, what it tests
CIRCUITS = [
    ("monza",     "Monza",       "Monza",     "Lowest downforce"),
    ("barcelona", "Catalunya",   "Barcelona", "Balanced reference"),
    ("bahrain",   "Sakhir",      "Bahrain",   "Traction and braking"),
    ("suzuka",    "Suzuka",      "Suzuka",    "High-speed aero corners"),
    ("monaco",    "Monte Carlo", "Monaco",    "Max downforce, lowest speed"),
]


# ----------------------------------------------------------------------
# 1. Telemetry, from OpenF1, cached
# ----------------------------------------------------------------------
def _get(url, tries=4):
    """GET with a small backoff. The API is free, so do not hammer it."""
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "f1-lapsim/1.0"})
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode())
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            if attempt == tries - 1:
                raise
            wait = 2 ** attempt
            print(f"      request failed ({e}), retrying in {wait}s")
            time.sleep(wait)


def _iso(t):
    return t.isoformat().replace("+00:00", "Z")


def fetch_pole_lap(slug, short_name, refresh=False, quiet=False):
    """
    Pull the fastest qualifying lap for one circuit and return it as a dict of
    plain arrays. Cached to disk, so this only touches the network once.
    """
    RAW.mkdir(exist_ok=True)
    cached = RAW / f"{YEAR}_{slug}_{SESSION.lower()}.json"
    if cached.exists() and not refresh:
        d = json.loads(cached.read_text())
        if not quiet:
            print(f"      cached ({len(d['t'])} merged samples)")
        return d

    sessions = _get(f"{API}/sessions?year={YEAR}&session_name={SESSION}")
    match = [s for s in sessions if s.get("circuit_short_name") == short_name]
    if not match:
        raise RuntimeError(f"no {YEAR} {SESSION} session found for {short_name!r}")
    sk = match[0]["session_key"]

    laps = [l for l in _get(f"{API}/laps?session_key={sk}") if l.get("lap_duration")]
    if not laps:
        raise RuntimeError(f"no timed laps for session_key={sk}")
    pole = min(laps, key=lambda l: l["lap_duration"])
    dn, real = pole["driver_number"], float(pole["lap_duration"])

    t0 = dt.datetime.fromisoformat(pole["date_start"])
    t1 = t0 + dt.timedelta(seconds=real + 0.5)
    window = f"session_key={sk}&driver_number={dn}&date>{_iso(t0)}&date<{_iso(t1)}"
    loc = _get(f"{API}/location?{window}")
    car = _get(f"{API}/car_data?{window}")
    if len(loc) < 50 or len(car) < 50:
        raise RuntimeError(f"{slug}: too few samples (loc={len(loc)}, car={len(car)})")

    drivers = _get(f"{API}/drivers?session_key={sk}&driver_number={dn}")
    who = drivers[0] if drivers else {}

    # Merge the two streams onto their union timebase, which is what FastF1
    # does internally. Each stream is interpolated onto the union, so neither
    # is thrown away and the effective rate is roughly the sum of the two.
    stamp = lambda rows: np.array(
        [dt.datetime.fromisoformat(r["date"]).timestamp() for r in rows])
    tl, tc = stamp(loc), stamp(car)
    grid = np.unique(np.concatenate([tl, tc]))
    grid = grid[(grid >= max(tl[0], tc[0])) & (grid <= min(tl[-1], tc[-1]))]

    col = lambda rows, k: np.array([float(r[k]) for r in rows])
    out = {
        "slug": slug,
        "circuit": match[0].get("circuit_short_name", short_name),
        "location": match[0].get("location", ""),
        "session_key": sk,
        "driver_number": dn,
        "driver": who.get("name_acronym") or str(dn),
        "team": who.get("team_name", ""),
        "realLapTime": real,
        "t": (grid - grid[0]).tolist(),
        # position telemetry is in 1/10 m, same convention as FastF1
        "x": np.interp(grid, tl, col(loc, "x")).tolist(),
        "y": np.interp(grid, tl, col(loc, "y")).tolist(),
        "speed": np.interp(grid, tc, col(car, "speed")).tolist(),      # km/h
        "throttle": np.interp(grid, tc, col(car, "throttle")).tolist(),
        "brake": np.interp(grid, tc, col(car, "brake")).tolist(),
        "drs": np.interp(grid, tc, col(car, "drs")).tolist(),
    }
    cached.write_text(json.dumps(out))
    if not quiet:
        print(f"      fetched {len(loc)} position + {len(car)} car samples "
              f"-> {len(grid)} merged ({cached.stat().st_size/1024:.0f} KB cached)")
    return out


def curvature_native(x_raw, y_raw, d_raw, d_grid, smooth_m=70.0):
    """
    Corner radius, computed at the telemetry's OWN resolution and only then
    interpolated onto the solver grid.

    This is the fix for a real and destructive artifact. The position channel
    arrives roughly every 9 to 10 metres of track. Interpolating that onto a
    2 m solver grid produces a piecewise-linear path, whose curvature is zero
    along each straight segment and spikes at every join. Smoothing over the
    fine grid cannot repair it, because a 31-point window spans only about
    three real samples. The spikes then invert to corner radii near zero,
    which the solver reads as corners no car could take, and one such point
    propagates through the forward and backward passes and wrecks the lap.
    At Barcelona and Monaco that was worth several seconds.

    Differentiating at native spacing removes the cause rather than filtering
    the symptom. A median filter then rejects isolated survivors, and anything
    still tighter than the tightest real corner in Formula 1 is clipped and
    counted, so the honesty cost of the clip is visible rather than hidden.

    The resolution limit is real and does not go away: at 9 m sampling,
    Monaco's tightest corners are described by about three points. That is a
    property of the data, and it belongs in the limitations, not in a filter.
    """
    sp = float(np.median(np.diff(d_raw)))
    w = int(max(5, round(smooth_m / sp)))
    w = w if w % 2 else w + 1
    w = min(w, (len(x_raw) // 2) * 2 - 1)

    xs = savgol_filter(x_raw, w, 3)
    ys = savgol_filter(y_raw, w, 3)
    dx = np.gradient(xs, d_raw);   dy = np.gradient(ys, d_raw)
    ddx = np.gradient(dx, d_raw);  ddy = np.gradient(dy, d_raw)

    num = np.abs(dx * ddy - dy * ddx)
    den = (dx ** 2 + dy ** 2) ** 1.5
    kappa = np.where(den > 1e-12, num / den, 0.0)
    kappa = median_filter(kappa, size=5, mode="nearest")
    kappa = np.clip(kappa, 0.0, None)

    with np.errstate(divide="ignore"):
        R_native = np.where(kappa > 1e-6, 1.0 / kappa, np.inf)

    R = np.interp(d_grid, d_raw, R_native)
    clipped = int(np.sum(R < R_FLOOR))
    R = np.where(R < R_FLOOR, R_FLOOR, R)
    return R, w, sp, clipped


def fit_cl_cd(R, v_real, drs, mu, ds=DS):
    """
    Fit downforce and drag only, with grip held at a value shared by every
    circuit.

    Fitting mu, clA and cdA together per circuit is not identifiable from one
    speed trace: grip and downforce both enter the cornering limit and trade
    off against each other almost freely. The free-parameter version of this
    study returned mu from 1.34 to 2.35 across five circuits, which is not a
    physical range, and the fitted downforce then absorbed whatever was left.

    Holding mu fixed across all circuits removes that freedom. What is left,
    the per-circuit clA and cdA, is then comparable between circuits, which is
    the entire point of the exercise.
    """
    def cost(p):
        cl, cd = p
        if not (2.0 < cl < 14.0 and 0.6 < cd < 4.5):
            return 1e6
        _, v = solve_lap(F1Car(mu=mu, cl_a=cl, cd_a=cd), R, drs, ds)
        return float(np.sqrt(np.mean((v - v_real) ** 2)))

    best = minimize(cost, x0=[5.0, 1.30], method="Nelder-Mead",
                    options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 400})
    cl, cd = best.x
    return F1Car(mu=mu, cl_a=cl, cd_a=cd), float(best.fun)


def to_distance_grid(tel, ds=DS):
    """
    Put the lap on an evenly spaced distance grid.

    Distance is obtained by integrating speed over time rather than by summing
    the x/y path, because the speed channel is the better conditioned of the
    two and the position trace is what will be differentiated for curvature.
    """
    t = np.asarray(tel["t"], float)
    v = np.asarray(tel["speed"], float) / 3.6                 # m/s
    d = np.concatenate([[0.0], np.cumsum(np.diff(t) * 0.5 * (v[1:] + v[:-1]))])

    grid = np.arange(0.0, d[-1], ds)
    pick = lambda k: np.interp(grid, d, np.asarray(tel[k], float))
    return {
        "distance": grid,
        "speed": pick("speed") / 3.6,                          # m/s
        "x": pick("x") * 0.1,                                  # metres
        "y": pick("y") * 0.1,
        "throttle": pick("throttle"),
        "brake": pick("brake"),
        "drs": (pick("drs") >= 10).astype(float),
        "lap_length": float(d[-1]),
        # Native-resolution arrays, kept so curvature can be computed at the
        # resolution the data actually has rather than at the solver's.
        "d_raw": d,
        "x_raw": np.asarray(tel["x"], float) * 0.1,
        "y_raw": np.asarray(tel["y"], float) * 0.1,
    }


# ----------------------------------------------------------------------
# 2. Run one circuit
# ----------------------------------------------------------------------
def prepare(slug, short_name, refresh=False, quiet=False):
    """Fetch, grid and extract geometry once. Reused by every fit mode."""
    if not quiet:
        print(f"  {slug:10s} ", end="", flush=True)
    tel = fetch_pole_lap(slug, short_name, refresh=refresh, quiet=quiet)
    g = to_distance_grid(tel)
    R, win, spacing, clipped = curvature_native(
        g["x_raw"], g["y_raw"], g["d_raw"], g["distance"])
    g["R"] = R
    g["geom"] = {"window": win, "nativeSpacingM": round(spacing, 2),
                 "clippedAtFloor": clipped,
                 "clippedFraction": round(clipped / len(R), 5)}
    return tel, g


def run_circuit(slug, short_name, label, tests, refresh=False, mu=None, prepared=None):
    tel, g = prepared if prepared else prepare(slug, short_name, refresh=refresh)
    R, v_real, drs = g["R"], g["speed"], g["drs"]

    if mu is None:
        car, rmse = fit_car(R, v_real, drs, ds=DS)
    else:
        car, rmse = fit_cl_cd(R, v_real, drs, mu, ds=DS)
    model_time, v_model = solve_lap(car, R, drs, ds=DS)
    real_time = float(tel["realLapTime"])

    finite = R[np.isfinite(R)]
    res = {
        "slug": slug,
        "circuit": label,
        "tests": tests,
        "driver": tel["driver"],
        "team": tel["team"],
        "realLapTime": round(real_time, 3),
        "modelLapTime": round(model_time, 3),
        "errorSeconds": round(model_time - real_time, 3),
        "errorPercent": round(100.0 * (model_time - real_time) / real_time, 3),
        "speedRmseKph": round(rmse * 3.6, 2),
        "mu": round(car.mu, 3),
        "clA": round(car.cl_a, 3),
        "cdA": round(car.cd_a, 3),
        "clA_over_cdA": round(car.cl_a / car.cd_a, 3),
        "lapLengthM": round(g["lap_length"], 1),
        "medianRadiusM": round(float(np.median(finite)), 1),
        "minRadiusM": round(float(finite.min()), 1),
        "fractionUnder250m": round(float(np.mean(R < 250)), 4),
        "samples": len(v_real),
        "cornersDetected": len(find_corners(v_real, ds=DS)),
        "muFixed": mu is not None,
        "geometry": g["geom"],
    }
    print(f"  {label:10s}{real_time:8.3f}s actual |{model_time:8.3f}s model | "
          f"{res['errorPercent']:+6.2f}% | RMSE {rmse*3.6:4.1f} km/h | "
          f"mu {car.mu:4.2f} clA {car.cl_a:5.2f} cdA {car.cd_a:4.2f}")

    # Export in the same shape the browser simulator already consumes, so a
    # circuit selector on the page is a loader change rather than a rewrite.
    DATA.mkdir(parents=True, exist_ok=True)
    step = 2
    payload = {
        "circuit": f"{label} {YEAR}",
        "session": "Qualifying, pole lap",
        "driver": tel["driver"],
        "team": tel["team"],
        "realLapTime": round(real_time, 3),
        "ds": DS * step,
        "fitted": {"mu": round(car.mu, 3), "clA": round(car.cl_a, 3),
                   "cdA": round(car.cd_a, 3), "mass": car.mass,
                   "power": car.power, "brakeG": car.brake_g, "vMax": car.v_max},
        "source": "OpenF1 (openf1.org), position and car data, merged",
        "distance": [round(float(x), 1) for x in g["distance"][::step]],
        "radius": [None if not np.isfinite(r) else round(float(r), 1)
                   for r in R[::step]],
        "speedReal": [round(float(v) * 3.6, 1) for v in v_real[::step]],
        "speedModel": [round(float(v) * 3.6, 1) for v in v_model[::step]],
        "drs": [int(v) for v in drs[::step]],
        "x": [round(float(v) * 10, 1) for v in g["x"][::step]],
        "y": [round(float(v) * 10, 1) for v in g["y"][::step]],
    }
    (DATA / f"{slug}-lap.json").write_text(json.dumps(payload), encoding="utf-8")
    return res


# ----------------------------------------------------------------------
# 3. The figure: how the fits vary, which is the actual headline
# ----------------------------------------------------------------------
INK, MUTED, GRID = P["text"], P["muted"], P["border"]
EMPH = accent("f1")          # reserved for the circuit where the model degrades


def identifiability(rows, prepared, mu, factor=0.4):
    """
    Ask whether the fitted parameters mean anything.

    A fit returns a number whatever you give it. The question that decides
    whether that number is a measurement or an artifact is: how much worse does
    the fit get if the parameter is badly wrong? If a 40% error in clA costs
    almost nothing, then the data does not constrain clA, and the per-circuit
    value is not evidence of anything about that circuit.

    The mechanism to look for is physical. Downforce scales with dynamic
    pressure, so on a slow circuit there is very little aerodynamic load for
    clA to act on, and clA becomes close to unobservable. That predicts the
    penalty should rise with mean dynamic pressure over the lap.
    """
    print("\n" + "=" * 92)
    print("IS ANY OF THIS IDENTIFIABLE? SENSITIVITY OF THE FIT TO clA")
    print("=" * 92)
    print(f"{'Circuit':<11}{'clA':>7}{'RMSE':>8}{'mean q':>9}"
          f"{'penalty for '+str(int(factor*100))+'% clA error':>30}")
    print("-" * 92)
    out = []
    for r in rows:
        _, g = prepared[r["slug"]]
        R, v, drs = g["R"], g["speed"], g["drs"]
        base = r["speedRmseKph"]

        def err(cl):
            _, vm = solve_lap(F1Car(mu=r["mu"], cl_a=cl, cd_a=r["cdA"]), R, drs, DS)
            return float(np.sqrt(np.mean((vm - v) ** 2))) * 3.6

        pen = min(err(r["clA"] * (1 - factor)), err(r["clA"] * (1 + factor))) - base
        q = float(np.mean(0.5 * 1.2 * v ** 2))
        r["meanDynamicPressurePa"] = round(q, 1)
        r["clAPenaltyKph"] = round(pen, 2)
        out.append((q, pen))
        verdict = ("not identifiable" if pen < 2.0
                   else "weakly identifiable" if pen < 6.0 else "identifiable")
        print(f"{r['circuit']:<11}{r['clA']:>7.2f}{base:>8.1f}{q:>8.0f}Pa"
              f"{pen:>+16.1f} km/h   {verdict}")
    print("-" * 92)
    if len(out) > 2:
        cc = float(np.corrcoef([q for q, _ in out], [p for _, p in out])[0, 1])
        print(f"\n  correlation(mean dynamic pressure, clA penalty) = {cc:+.3f}")
        print(f"  Baseline speed-trace error is {min(r['speedRmseKph'] for r in rows):.0f}"
              f" to {max(r['speedRmseKph'] for r in rows):.0f} km/h, so any penalty")
        print("  below about 2 km/h is well inside the model's own error and the")
        print("  fitted value should not be read as a property of the circuit.")
    return rows


def validation_figure(rows, path):
    """
    Three panels, one measure each. Never two y-scales on one axis: clA and cdA
    differ in magnitude by roughly a factor of four, so they get their own
    panels rather than a dual axis that invites a false comparison.
    """
    order = sorted(rows, key=lambda r: r["clA"])
    names = [r["circuit"] for r in order]
    ypos = np.arange(len(order))
    # Emphasis goes on the circuits where clA is NOT identifiable, because
    # that, not the lap time error, is what this study actually found.
    unident = {r["circuit"] for r in rows if r.get("clAPenaltyKph", 99) < 2.0}
    colour = [EMPH if n in unident else INK for n in names]

    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.6))

    def barh(ax, vals, title, fmt, xlabel):
        ax.barh(ypos, vals, 0.62, color=colour)
        ax.set_yticks(ypos)
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10.5, color=INK, pad=9)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.grid(axis="x", alpha=0.25, color=GRID)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        span = max(abs(v) for v in vals) or 1.0
        for i, v in enumerate(vals):
            off = span * 0.02
            ax.text(v + (off if v >= 0 else -off), i, fmt.format(v),
                    va="center", ha="left" if v >= 0 else "right",
                    fontsize=9, color=INK)
        pad = span * 0.40
        lo = min(0, min(vals) - pad)
        ax.set_xlim(lo, max(vals) + pad)

    barh(axes[0], [r["clA"] for r in order],
         "Fitted downforce, $C_LA$", "{:.2f}", "$C_LA$ (m$^2$)")
    barh(axes[1], [r["cdA"] for r in order],
         "Fitted drag, $C_DA$", "{:.2f}", "$C_DA$ (m$^2$)")
    barh(axes[2], [r["errorPercent"] for r in order],
         "Lap time error against the real lap", "{:+.2f}%", "Model minus measured (%)")
    axes[2].axvline(0, color=MUTED, lw=1)
    barh(axes[3], [r.get("clAPenaltyKph", 0.0) for r in order],
         "Does the data constrain $C_LA$?", "{:+.1f}",
         "Extra speed error from a 40% $C_LA$ error (km/h)")
    axes[3].axvline(2.0, color=EMPH, lw=1, ls="--")
    axes[3].text(2.0, len(order) - 0.35, " below this, not identifiable",
                 fontsize=8, color=EMPH, va="center", ha="left")

    fig.suptitle(
        "One model, five circuits: the fitted aerodynamics do not recover real circuit configurations",
        fontsize=13, color=INK, y=1.03)
    fig.text(0.5, -0.085,
             "Expected downforce order was Monza lowest and Monaco highest. The fit returns Monaco lowest and Suzuka highest, so the ordering is not recovered.\n"
             "The right-hand panel shows why: on three of five circuits a 40% error in $C_LA$ costs under 2 km/h of speed-trace error, which is inside the model's own\n"
             "27 to 39 km/h baseline. Those fitted values are artifacts of an unidentifiable parameter, not properties of the circuit. Red marks the unidentifiable cases.\n"
             "Fitted on 2025 qualifying pole laps, grip held common across circuits at mu = 1.30. Telemetry: OpenF1.",
             ha="center", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  figure -> {path.name}")


# ----------------------------------------------------------------------
# 4. Main
# ----------------------------------------------------------------------
def main(argv):
    refresh = "--refresh" in argv
    wanted = [a.lower() for a in argv if not a.startswith("-")]
    todo = [c for c in CIRCUITS if not wanted or c[0] in wanted]
    if not todo:
        print(f"no circuit matched {wanted}. known: {[c[0] for c in CIRCUITS]}")
        return 1

    print(f"\nValidating one lap model across {len(todo)} circuit(s), "
          f"{YEAR} qualifying\n" + "=" * 74)

    # ---- geometry once, reused by every fit ----
    prepared, failed = {}, []
    for slug, short, label, tests in todo:
        try:
            prepared[slug] = prepare(slug, short, refresh=refresh)
            g = prepared[slug][1]["geom"]
            print(f"      geometry: {g['nativeSpacingM']}m native spacing, "
                  f"{g['window']}-point window, "
                  f"{g['clippedAtFloor']} points clipped at the {R_FLOOR:.0f}m floor "
                  f"({100*g['clippedFraction']:.2f}%)")
        except Exception as e:
            print(f"\n  {label:10s} FAILED: {type(e).__name__}: {e}")
            failed.append((label, f"{type(e).__name__}: {e}"))

    todo = [c for c in todo if c[0] in prepared]
    if not todo:
        print("\nNothing succeeded.")
        return 1

    # ---- choose one grip value shared by every circuit ----
    # Scanned rather than optimised jointly: 1 + 2N parameters is badly
    # conditioned, and a scan shows the shape of the objective instead of
    # returning a single number with no indication of how sharp it is.
    shared_mu = None
    if len(todo) > 1 and "--free-mu" not in argv:
        print(f"\n  Scanning for the grip value that best fits all "
              f"{len(todo)} circuits at once")
        print(f"  {'mu':>6}{'mean RMSE (km/h)':>20}{'worst circuit':>18}")
        print("  " + "-" * 44)
        scan = []
        for mu in np.arange(1.30, 2.01, 0.10):
            rs = []
            for slug, short, label, tests in todo:
                _, g = prepared[slug]
                _, rmse = fit_cl_cd(g["R"], g["speed"], g["drs"], float(mu), ds=DS)
                rs.append((rmse * 3.6, label))
            mean = float(np.mean([r for r, _ in rs]))
            worst = max(rs)[1]
            scan.append((mean, float(mu)))
            print(f"  {mu:>6.2f}{mean:>20.2f}{worst:>18s}")
        shared_mu = min(scan)[1]
        print(f"  " + "-" * 44)
        print(f"  Best shared grip: mu = {shared_mu:.2f} "
              f"(mean speed RMSE {min(scan)[0]:.2f} km/h)")

    rows = []
    label_mode = ("grip shared across circuits at mu = %.2f" % shared_mu
                  if shared_mu is not None else "all three parameters free per circuit")
    print(f"\n  Fitting with {label_mode}\n" + "=" * 74)
    for slug, short, label, tests in todo:
        try:
            rows.append(run_circuit(slug, short, label, tests,
                                    mu=shared_mu, prepared=prepared[slug]))
        except Exception as e:
            print(f"\n  {label:10s} FAILED: {type(e).__name__}: {e}")
            failed.append((label, f"{type(e).__name__}: {e}"))

    if not rows:
        print("\nNothing succeeded.")
        return 1

    # ---- the validation table ----
    print("\n" + "=" * 92)
    print("VALIDATION TABLE")
    print("=" * 92)
    print(f"{'Circuit':<11}{'Tests':<30}{'Actual':>9}{'Model':>9}"
          f"{'Error':>9}{'Err %':>8}{'RMSE':>8}")
    print("-" * 92)
    for r in rows:
        print(f"{r['circuit']:<11}{r['tests']:<30}{r['realLapTime']:>9.3f}"
              f"{r['modelLapTime']:>9.3f}{r['errorSeconds']:>+9.3f}"
              f"{r['errorPercent']:>+7.2f}%{r['speedRmseKph']:>7.1f}k")
    print("-" * 92)
    errs = [abs(r["errorPercent"]) for r in rows]
    print(f"{'':<50}mean |error| {np.mean(errs):5.2f}%   "
          f"worst {max(errs):5.2f}%   best {min(errs):5.2f}%")

    # ---- the part that is actually the headline ----
    print("\n" + "=" * 92)
    print("FITTED AERODYNAMIC PARAMETERS, SOLVED INDEPENDENTLY PER CIRCUIT")
    print("=" * 92)
    print(f"{'Circuit':<11}{'mu':>7}{'clA':>8}{'cdA':>8}{'clA/cdA':>10}"
          f"{'median R':>10}{'min R':>8}{'<250m':>8}")
    print("-" * 92)
    for r in sorted(rows, key=lambda r: r["clA"]):
        print(f"{r['circuit']:<11}{r['mu']:>7.2f}{r['clA']:>8.2f}{r['cdA']:>8.2f}"
              f"{r['clA_over_cdA']:>10.2f}{r['medianRadiusM']:>9.0f}m"
              f"{r['minRadiusM']:>7.0f}m{100*r['fractionUnder250m']:>7.0f}%")
    print("-" * 92)
    if len(rows) > 1:
        cl = [r["clA"] for r in rows]
        cd = [r["cdA"] for r in rows]
        lo = min(rows, key=lambda r: r["clA"])
        hi = max(rows, key=lambda r: r["clA"])
        print(f"\n  clA spread: {min(cl):.2f} ({lo['circuit']}) to {max(cl):.2f} "
              f"({hi['circuit']}), a factor of {max(cl)/min(cl):.2f}")
        print(f"  cdA spread: {min(cd):.2f} to {max(cd):.2f}, "
              f"a factor of {max(cd)/min(cd):.2f}")
        print("\n  Read this against the real world before believing it: the lowest")
        print("  fitted downforce should land on Monza and the highest on Monaco.")
        print("  The model was never told which circuit is which, so if that")
        print("  ordering appears on its own it is evidence of physics rather")
        print("  than of curve-fitting. If it does not appear, say so.")
        got = [r["circuit"] for r in sorted(rows, key=lambda r: r["clA"])]
        print(f"\n  Expected order (low to high downforce): "
              f"Monza, Bahrain, Barcelona, Suzuka, Monaco")
        print(f"  Fitted order   (low to high downforce): {', '.join(got)}")
        if got and got[0] != "Monza":
            print("\n  >> The ordering does NOT appear. The fitted parameters do not")
            print("     recover real aerodynamic configurations, so the headline")
            print("     claim this study was built to test is NOT supported.")
            print("     See the identifiability section below for why.")

    identifiability(rows, prepared, shared_mu)

    if failed:
        print("\n  FAILED: " + "; ".join(f"{n} ({e})" for n, e in failed))

    FIGS.mkdir(parents=True, exist_ok=True)
    validation_figure(rows, FIGS / "f1-validation.png")

    DATA.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated": dt.date.today().isoformat(),
        "year": YEAR,
        "session": SESSION,
        "source": "OpenF1 (openf1.org)",
        "solver": "f1_lap_model.py, imported unchanged",
        "circuits": rows,
        "failed": [{"circuit": n, "error": e} for n, e in failed],
    }
    (DATA / "validation-summary.json").write_text(json.dumps(summary, indent=2),
                                                  encoding="utf-8")
    print(f"  summary -> data/multi/validation-summary.json")
    print(f"  per-circuit lap data -> data/multi/<slug>-lap.json\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
