"""
Is the lap time a prediction, or just a restatement of the fit?
===============================================================

Veer Phatak, 2026.

THE OBJECTION THIS ANSWERS
--------------------------
`f1_lap_model.py` fits three parameters (grip, downforce, drag) to the measured
speed trace, and then reports that the predicted lap time is within 0.72% of
the real one. Lap time is the integral of distance over speed. If you fit the
speed, a close lap time is close to guaranteed, and the headline number is a
goodness-of-fit statistic wearing the clothes of a prediction.

That objection is correct in principle and has to be tested, not waved away.

THE TEST
--------
Fit the model on part of the lap, then measure its error on the part it never
saw. If the model is memorising this particular lap, held-out error will be
clearly worse than fitted error. If the geometry and the physics are doing the
work, the two will be comparable.

Five splits are used rather than one, because a single convenient split is just
another way of choosing the answer. The alternating-blocks split is the
harshest: it interleaves fitted and unseen track every 100 m, so the model
cannot benefit from a whole section being similar to its neighbours.

THE SECOND TEST
---------------
Separately, grip and downforce both set the cornering limit, so the fit cannot
necessarily tell them apart. Holding grip at a series of fixed values and
refitting the aerodynamics each time shows how much the fit actually cares.
A flat result across a wide range means the pair trades along a ridge, and
neither value should be quoted as a measurement of the real car.

Both tables on the website's lap model page come from this script.

USAGE
-----
    python f1_validate_outofsample.py

Runs entirely from the local FastF1 cache. No network required.
"""

from pathlib import Path
import logging
import warnings

import numpy as np
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
logging.getLogger("fastf1").setLevel(logging.ERROR)

import fastf1

HERE = Path(__file__).parent
fastf1.Cache.enable_cache(str(HERE / "cache"))
# The F1 live timing API began refusing ordinary connections during 2026, so
# this runs cache-only. Bahrain qualifying is already cached.
fastf1.Cache.offline_mode(True)

import f1_lap_model as M


def load():
    ses, lap, tel_raw = M.load_pole_lap()
    tel = M.resample(tel_raw)
    R = M.curvature_radius(tel["X"].to_numpy() * 0.1, tel["Y"].to_numpy() * 0.1)
    v_real = tel["Speed"].to_numpy() / 3.6
    drs = (tel["DRS"].to_numpy() >= 10).astype(float)
    d = tel["Distance"].to_numpy()
    return lap, R, v_real, drs, d


def fit_on(R, v_real, drs, mask):
    """Fit grip, downforce and drag using only the masked part of the lap."""
    def cost(p):
        mu, cl, cd = p
        if not (1.0 < mu < 2.6 and 2.0 < cl < 14.0 and 0.6 < cd < 4.5):
            return 1e6
        _, v = M.solve_lap(M.F1Car(mu=mu, cl_a=cl, cd_a=cd), R, drs)
        return float(np.sqrt(np.mean((v[mask] - v_real[mask]) ** 2)))

    b = minimize(cost, [1.75, 5.0, 1.30], method="Nelder-Mead",
                 options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 400})
    return M.F1Car(mu=b.x[0], cl_a=b.x[1], cd_a=b.x[2])


def fit_aero_at_fixed_grip(R, v_real, drs, mu):
    """Fit downforce and drag only, with grip held at `mu`."""
    def cost(p):
        cl, cd = p
        if not (2.0 < cl < 14.0 and 0.6 < cd < 4.5):
            return 1e6
        _, v = M.solve_lap(M.F1Car(mu=mu, cl_a=cl, cd_a=cd), R, drs)
        return float(np.sqrt(np.mean((v - v_real) ** 2)))

    b = minimize(cost, [5.0, 1.30], method="Nelder-Mead",
                 options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 400})
    return b.x[0], b.x[1], float(b.fun) * 3.6


def main():
    lap, R, v_real, drs, d = load()
    real = lap["LapTime"].total_seconds()
    total = d[-1]

    print(f"\n2025 Bahrain qualifying, {lap['Driver']} pole lap: {real:.3f} s")
    print("=" * 94)
    print("TEST 1: OUT OF SAMPLE. Fit on part of the lap, predict the rest.\n")
    print(f"  {'Fitted on':<26}{'clA':>7}{'Lap':>9}{'Error':>9}"
          f"{'Fitted RMSE':>13}{'UNSEEN RMSE':>13}{'Ratio':>8}")
    print("  " + "-" * 86)

    splits = [
        ("First 50%",                d <= 0.50 * total),
        ("First 66%",                d <= 0.66 * total),
        ("First 75%",                d <= 0.75 * total),
        ("Last 66%",                 d >= 0.34 * total),
        ("Alternating 100 m blocks", np.floor(d / 100).astype(int) % 2 == 0),
    ]
    ratios, errors = [], []
    for name, mask in splits:
        car = fit_on(R, v_real, drs, mask)
        t, v = M.solve_lap(car, R, drs)
        rf = float(np.sqrt(np.mean((v[mask] - v_real[mask]) ** 2))) * 3.6
        rh = float(np.sqrt(np.mean((v[~mask] - v_real[~mask]) ** 2))) * 3.6
        err = 100 * (t - real) / real
        ratios.append(rh / rf)
        errors.append(abs(err))
        print(f"  {name:<26}{car.cl_a:>7.2f}{t:>9.3f}{err:>+8.2f}%"
              f"{rf:>11.1f} km/h{rh:>11.1f} km/h{rh/rf:>8.2f}")

    print("  " + "-" * 86)
    print(f"\n  Worst ratio {max(ratios):.2f}. A ratio at or below about 1.0 means the")
    print( "  model is no worse on track it never saw, so the geometry and the physics")
    print( "  are carrying the prediction rather than the fit.")
    print(f"\n  Lap time error ranges {min(errors):.2f}% to {max(errors):.2f}% across the five")
    print( "  splits. The headline 0.72% is one draw from that range, so 'within about")
    print( "  1%' is the claim this method supports. Quoting two significant figures")
    print( "  would overstate the precision.")

    print("\n" + "=" * 94)
    print("TEST 2: CAN THE FIT SEPARATE GRIP FROM DOWNFORCE?\n")
    print(f"  {'mu (assumed)':>14}{'clA (fitted)':>15}{'cdA (fitted)':>15}{'Speed RMSE':>14}")
    print("  " + "-" * 56)
    rows = []
    for mu in (1.00, 1.15, 1.30, 1.48, 1.60, 1.80, 2.00, 2.30):
        cl, cd, rm = fit_aero_at_fixed_grip(R, v_real, drs, mu)
        rows.append((mu, cl, cd, rm))
        print(f"  {mu:>14.2f}{cl:>15.2f}{cd:>15.2f}{rm:>11.1f} km/h")
    print("  " + "-" * 56)

    flat = [r for r in rows if 1.30 <= r[0] <= 1.80]
    cl_hi, cl_lo = max(r[1] for r in flat), min(r[1] for r in flat)
    rm_sp = max(r[3] for r in flat) - min(r[3] for r in flat)
    best = min(rows, key=lambda r: r[3])
    print(f"\n  Between mu 1.30 and 1.80 the fit quality varies by {rm_sp:.1f} km/h,")
    print(f"  while fitted downforce varies by a factor of {cl_hi/cl_lo:.2f}"
          f" ({cl_lo:.1f} to {cl_hi:.1f}).")
    print(f"  Best fit is at mu = {best[0]:.2f}, so there is a real optimum and the")
    print( "  model is not degenerate. But the precision lives in the PAIR, not in")
    print( "  either number, and neither should be quoted as a property of the car.")
    print( "\n  Separating them needs information the speed trace does not contain:")
    print( "  a second configuration of the same car, a straight-line aerodynamic")
    print( "  measurement, or tyre data. No further fitting will resolve it.\n")


if __name__ == "__main__":
    main()
