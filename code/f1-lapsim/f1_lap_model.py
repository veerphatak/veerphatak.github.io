"""
A physics lap time model checked against real Formula 1 telemetry
=================================================================

Veer Phatak, 2026.

WHAT THIS DOES
--------------
Builds a quasi-steady lap time model of a Formula 1 car, runs it around a real
circuit whose geometry is derived from real car position telemetry, and then
compares the prediction against what the car actually did.

The comparison is the point. A model that has never been checked against
measurement is an opinion. This one is scored lap by lap and corner by corner,
and the places where it disagrees are the interesting part, because each one
points at a piece of physics the model does not contain.

DATA
----
Timing and car telemetry come from the official F1 live timing API via FastF1
(https://github.com/theOehrly/Fast-F1). The data is Formula 1's, used here for
non-commercial analysis. Nothing is redistributed: the script downloads what it
needs into a local cache that is not committed.

Session used: 2025 Bahrain Grand Prix, qualifying, pole lap.

METHOD
------
1. Take the pole lap's telemetry: speed, throttle, brake, gear, DRS, and the
   car's X/Y position around the lap.
2. Turn the position trace into circuit geometry. Resample onto an even
   distance grid, smooth it, and compute local curvature analytically:

       kappa = |x' y'' - y' x''| / (x'^2 + y'^2)^(3/2),   R = 1 / kappa

3. Solve the lap with the same three-pass quasi-steady method used in my
   Formula Student model: a cornering limit that accounts for speed-dependent
   downforce, then forward (traction) and backward (braking) passes sharing
   tyre capacity through a friction ellipse.
4. Fit the handful of parameters that are not public (effective grip, downforce
   and drag levels) to the measured speed trace, then report the error honestly.
"""

from dataclasses import dataclass
from pathlib import Path
import warnings, logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import minimize

import fastf1

warnings.filterwarnings("ignore")
logging.getLogger("fastf1").setLevel(logging.ERROR)

HERE = Path(__file__).parent
FIGS = HERE.parent.parent / "media" / "figures"
fastf1.Cache.enable_cache(str(HERE / "cache"))

YEAR, EVENT = 2025, "Bahrain"
DS = 2.0          # metres between solver points


# ----------------------------------------------------------------------
# 1. Real telemetry
# ----------------------------------------------------------------------
def load_pole_lap():
    ses = fastf1.get_session(YEAR, EVENT, "Q")
    ses.load(telemetry=True, weather=False, messages=False)
    lap = ses.laps.pick_fastest()
    tel = lap.get_telemetry().add_distance()
    tel = tel[tel["Distance"] >= 0].reset_index(drop=True)
    return ses, lap, tel


def resample(tel, ds=DS):
    """Put everything on an evenly spaced distance grid."""
    d = tel["Distance"].to_numpy(dtype=float)
    grid = np.arange(d.min(), d.max(), ds)
    out = {"Distance": grid}
    for col in ("Speed", "Throttle", "Brake", "nGear", "DRS", "X", "Y"):
        if col in tel:
            vals = pd.to_numeric(tel[col], errors="coerce").to_numpy(dtype=float)
            out[col] = np.interp(grid, d, vals)
    return pd.DataFrame(out)


def curvature_radius(x, y, ds=DS, window=31):
    """
    Local corner radius from the car's path.

    Position telemetry is noisy, and curvature needs second derivatives, which
    amplify noise badly. Savitzky-Golay filtering fits a low-order polynomial
    over a moving window and differentiates that instead, which is far better
    behaved than differencing raw samples.
    """
    w = min(window if window % 2 else window + 1, (len(x) // 2) * 2 - 1)
    xs = savgol_filter(x, w, 3)
    ys = savgol_filter(y, w, 3)
    dx = np.gradient(xs, ds);   dy = np.gradient(ys, ds)
    ddx = np.gradient(dx, ds);  ddy = np.gradient(dy, ds)
    num = np.abs(dx * ddy - dy * ddx)
    den = (dx ** 2 + dy ** 2) ** 1.5
    kappa = np.where(den > 1e-9, num / den, 0.0)
    kappa = savgol_filter(kappa, w, 3)
    kappa = np.clip(kappa, 0.0, None)
    with np.errstate(divide="ignore"):
        R = np.where(kappa > 1e-6, 1.0 / kappa, np.inf)
    return R


# ----------------------------------------------------------------------
# 2. Car model
# ----------------------------------------------------------------------
@dataclass
class F1Car:
    mass: float = 850.0        # kg, 2025 minimum weight plus qualifying fuel
    mu: float = 1.75           # effective peak friction, fitted
    cl_a: float = 5.0          # downforce, fitted
    cd_a: float = 1.30         # drag, fitted
    power: float = 760_000.0   # W at the wheels
    brake_g: float = 5.0       # peak deceleration in g
    rho: float = 1.20
    v_max: float = 95.0        # m/s, gearing/drag limit
    drs_drag_cut: float = 0.12 # fraction of drag removed with DRS open

    def drag(self, v, drs=0.0):
        cd = self.cd_a * (1.0 - self.drs_drag_cut * drs)
        return 0.5 * self.rho * cd * v ** 2

    def downforce(self, v):
        return 0.5 * self.rho * self.cl_a * v ** 2

    def normal_load(self, v):
        return self.mass * 9.81 + self.downforce(v)


def corner_limit(car, R):
    """Speed where lateral demand exactly equals available grip."""
    k = car.mu * 0.5 * car.rho * car.cl_a / car.mass
    with np.errstate(divide="ignore", invalid="ignore"):
        bracket = 1.0 / R - k
        v2 = np.where(bracket > 0, car.mu * 9.81 / bracket, np.inf)
    return np.minimum(np.sqrt(np.maximum(v2, 0.0)), car.v_max)


def solve_lap(car, R, drs=None, ds=DS):
    """Three-pass quasi-steady solve. Speeds in m/s."""
    n = len(R)
    drs = np.zeros(n) if drs is None else drs
    v_corner = corner_limit(car, R)

    v_f = np.empty(n); v_f[0] = v_corner[0]
    for i in range(n - 1):
        v = min(v_f[i], v_corner[i])
        grip = car.mu * car.normal_load(v)
        lat = car.mass * v ** 2 / R[i] if np.isfinite(R[i]) else 0.0
        remaining = np.sqrt(max(grip ** 2 - lat ** 2, 0.0))
        f = min(car.power / max(v, 1.0), remaining) - car.drag(v, drs[i])
        v_f[i + 1] = min(np.sqrt(max(v ** 2 + 2 * (f / car.mass) * ds, 1.0)), car.v_max)

    v_b = np.empty(n); v_b[-1] = v_corner[-1]
    for i in range(n - 1, 0, -1):
        v = min(v_b[i], v_corner[i])
        grip = car.mu * car.normal_load(v)
        lat = car.mass * v ** 2 / R[i] if np.isfinite(R[i]) else 0.0
        remaining = np.sqrt(max(grip ** 2 - lat ** 2, 0.0))
        f = min(remaining, car.brake_g * car.mass * 9.81) + car.drag(v, drs[i])
        v_b[i - 1] = min(np.sqrt(max(v ** 2 + 2 * (f / car.mass) * ds, 1.0)), car.v_max)

    v = np.minimum.reduce([v_f, v_b, v_corner])
    lap_time = float(np.sum(ds / v))
    return lap_time, v


# ----------------------------------------------------------------------
# 3. Fit the unknown parameters to the measured lap
# ----------------------------------------------------------------------
def fit_car(R, v_real, drs, ds=DS):
    """
    Grip, downforce and drag are not public, so they are fitted to the measured
    speed trace. Everything else (the geometry, the solver, the lap time) then
    follows from physics rather than from tuning.
    """
    def cost(p):
        mu, cl, cd = p
        if not (1.0 < mu < 2.6 and 2.0 < cl < 14.0 and 0.6 < cd < 4.5):
            return 1e6
        car = F1Car(mu=mu, cl_a=cl, cd_a=cd)
        _, v = solve_lap(car, R, drs, ds)
        return float(np.sqrt(np.mean((v - v_real) ** 2)))

    best = minimize(cost, x0=[1.75, 5.0, 1.30], method="Nelder-Mead",
                    options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 300})
    mu, cl, cd = best.x
    return F1Car(mu=mu, cl_a=cl, cd_a=cd), float(best.fun)


def find_corners(v_real, ds=DS, prominence=4.0, min_gap=120.0):
    """
    Corner apexes are local minima in the measured speed trace. Using
    prominence rather than a plain threshold means a genuine slow corner is
    found whether it is a hairpin or a fast kink, and shallow wiggles in the
    trace are ignored.
    """
    from scipy.signal import find_peaks
    idx, _ = find_peaks(-v_real, prominence=prominence, distance=int(min_gap / ds))
    return sorted(int(i) for i in idx)


# ----------------------------------------------------------------------
# 4. Run and report
# ----------------------------------------------------------------------
def main():
    ses, lap, tel_raw = load_pole_lap()
    tel = resample(tel_raw)
    # FastF1 reports position in 1/10 m, so convert to metres before differentiating
    R = curvature_radius(tel["X"].to_numpy() * 0.1, tel["Y"].to_numpy() * 0.1)
    v_real = tel["Speed"].to_numpy() / 3.6
    drs = (tel["DRS"].to_numpy() >= 10).astype(float)
    d = tel["Distance"].to_numpy()

    real_time = lap["LapTime"].total_seconds()
    car, rmse = fit_car(R, v_real, drs)
    model_time, v_model = solve_lap(car, R, drs)

    print(f"Session         : {YEAR} {ses.event['EventName']} qualifying")
    print(f"Pole lap        : {lap['Driver']} ({lap['Team']})  {real_time:.3f} s")
    print(f"Model lap time  : {model_time:.3f} s   error {model_time - real_time:+.3f} s "
          f"({100*(model_time-real_time)/real_time:+.2f}%)")
    print(f"Speed trace RMSE: {rmse*3.6:.1f} km/h")
    print(f"Fitted mu={car.mu:.2f}  CLA={car.cl_a:.2f}  CDA={car.cd_a:.2f}")

    corners = find_corners(v_real)
    finite = R[np.isfinite(R)]
    print(f"Corner radii: min {finite.min():.0f} m, median {np.median(finite):.0f} m, "
          f"{100*np.mean(R < 250):.0f}% of lap under 250 m")
    print(f"Corners detected: {len(corners)}")

    # ---------------- Figure 1: speed trace ----------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]})
    ax = axes[0]
    ax.plot(d, v_real * 3.6, lw=2.0, color="#14314b", label=f"Measured ({lap['Driver']} pole lap)")
    ax.plot(d, v_model * 3.6, lw=1.6, color="#c0392b", ls="--", label="Physics model")
    ax.fill_between(d, 0, 340, where=(R < 250), color="#cfd8dc", alpha=0.45, label="Corner")
    for i in corners:
        ax.annotate(f"{v_real[i]*3.6:.0f}", (d[i], v_real[i]*3.6), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=7, color="#14314b")
    ax.set_ylabel("Speed (km/h)"); ax.set_ylim(0, 340)
    ax.set_title(f"{YEAR} {ses.event['EventName']}: model against measured pole lap "
                 f"({real_time:.3f} s actual, {model_time:.3f} s predicted)")
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(d, tel["Throttle"], color="#2e7d32", lw=1.2, label="Throttle (%)")
    ax2.plot(d, tel["Brake"] * 100, color="#c0392b", lw=1.2, label="Brake (on/off)")
    ax2.fill_between(d, 0, 100, where=(drs > 0), color="#58a6ff", alpha=0.25, label="DRS open")
    ax2.set_xlabel("Distance around lap (m)"); ax2.set_ylabel("Driver inputs")
    ax2.set_xlim(d[0], d[-1]); ax2.legend(loc="lower right", fontsize=8, ncol=3); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGS / "f1-speed-trace.png", dpi=150); plt.close(fig)

    # ---------------- Figure 2: corner speeds + gear ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    xs = np.arange(len(corners))
    ax.bar(xs - 0.2, [v_real[i]*3.6 for i in corners], 0.4, color="#14314b", label="Measured")
    ax.bar(xs + 0.2, [v_model[i]*3.6 for i in corners], 0.4, color="#c0392b", label="Model")
    ax.set_xticks(xs); ax.set_xticklabels([f"T{i+1}" for i in xs], fontsize=8)
    ax.set_ylabel("Apex speed (km/h)"); ax.set_title("Minimum speed, corner by corner")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    sc = ax.scatter(tel["Speed"], tel["nGear"], c=tel["Throttle"], cmap="viridis", s=8)
    ax.set_xlabel("Speed (km/h)"); ax.set_ylabel("Gear")
    ax.set_title("Gear against speed, coloured by throttle")
    ax.set_yticks(range(1, 9)); ax.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax, label="Throttle (%)")
    fig.tight_layout(); fig.savefig(FIGS / "f1-corner-gear.png", dpi=150); plt.close(fig)

    # ---------------- Figure 3: circuit map ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    ax = axes[0]
    pts = ax.scatter(tel["X"], tel["Y"], c=v_real * 3.6, cmap="plasma", s=6)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Circuit from position telemetry, coloured by measured speed")
    plt.colorbar(pts, ax=ax, label="Speed (km/h)", shrink=0.8)

    ax = axes[1]
    err = (v_model - v_real) * 3.6
    pts = ax.scatter(tel["X"], tel["Y"], c=err, cmap="coolwarm", s=6,
                     vmin=-40, vmax=40)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Model error (red = model too fast)")
    plt.colorbar(pts, ax=ax, label="Model minus measured (km/h)", shrink=0.8)
    fig.tight_layout(); fig.savefig(FIGS / "f1-track-map.png", dpi=150); plt.close(fig)

    # ---------------- Export for the interactive browser version ----------------
    # The JavaScript simulator on the website runs the same solver on this data,
    # so visitors can change the car and watch the lap respond.
    import json
    step = 2                              # halve the resolution to keep it light
    data = {
        "circuit": f"{ses.event['EventName']} {YEAR}",
        "session": "Qualifying, pole lap",
        "driver": str(lap["Driver"]),
        "team": str(lap["Team"]),
        "realLapTime": round(real_time, 3),
        "ds": DS * step,
        "fitted": {"mu": round(car.mu, 3), "clA": round(car.cl_a, 3),
                   "cdA": round(car.cd_a, 3), "mass": car.mass,
                   "power": car.power, "brakeG": car.brake_g, "vMax": car.v_max},
        "distance": [round(v, 1) for v in d[::step]],
        "radius": [None if not np.isfinite(r) else round(float(r), 1) for r in R[::step]],
        "speedReal": [round(float(v) * 3.6, 1) for v in v_real[::step]],
        "drs": [int(v) for v in drs[::step]],
        "x": [round(float(v), 1) for v in tel["X"].to_numpy()[::step]],
        "y": [round(float(v), 1) for v in tel["Y"].to_numpy()[::step]],
    }
    out = HERE.parent.parent / "data" / "bahrain-lap.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(data), encoding="utf-8")
    print(f"\nExported {out.name}: {len(data['distance'])} points, "
          f"{out.stat().st_size/1024:.0f} KB")

    # corner error table for the write-up
    print("\nCorner       measured   model    error")
    for n_, i in enumerate(corners, 1):
        print(f"  T{n_:<2d}        {v_real[i]*3.6:6.1f}   {v_model[i]*3.6:6.1f}   {(v_model[i]-v_real[i])*3.6:+6.1f} km/h")


if __name__ == "__main__":
    main()
