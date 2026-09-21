"""
Tyre degradation and race strategy, fitted to real Formula 1 race data
======================================================================

Veer Phatak, 2026.

WHAT THIS DOES
--------------
Takes every green-flag lap of a real Grand Prix, separates the two effects that
make lap times change through a stint, fits a degradation model per tyre
compound, and then uses that model to ask whether the strategy that won the
race was actually the quickest one available.

THE CENTRAL PROBLEM
-------------------
A lap time is not a clean measurement of tyre condition. Two things move it in
opposite directions at once:

  * the tyres wear, which makes the car slower as a stint goes on;
  * the car burns fuel, roughly 1.7 kg a lap, which makes it lighter and
    therefore faster as the race goes on.

Fit degradation without removing the fuel effect and you will underestimate it,
because fuel burn has been quietly cancelling part of it. So both are fitted at
once, in a single regression, rather than one after the other:

    laptime = base(compound) + deg(compound) * tyre_age + fuel_effect * lap

Session used: 2025 Bahrain Grand Prix, a circuit chosen because it is
genuinely degradation limited, which makes the strategy question real.

DATA
----
Official F1 timing data via FastF1. Nothing is redistributed; the script
downloads what it needs into a local cache that is not committed.
"""

from pathlib import Path
from itertools import product
import warnings, logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fastf1

warnings.filterwarnings("ignore")
logging.getLogger("fastf1").setLevel(logging.ERROR)

HERE = Path(__file__).parent
FIGS = HERE.parent.parent / "media" / "figures"
fastf1.Cache.enable_cache(str(HERE / "cache"))

YEAR, EVENT = 2025, "Bahrain"
COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
COLOURS = {"SOFT": "#c0392b", "MEDIUM": "#d4ac0d", "HARD": "#5d6d7e"}


# ----------------------------------------------------------------------
# 1. Clean the race data
# ----------------------------------------------------------------------
def load_clean_laps(ses):
    laps = ses.laps.copy()
    laps["LapTimeS"] = laps["LapTime"].dt.total_seconds()

    before = len(laps)
    # Green flag only. Safety cars and yellows destroy pace data.
    laps = laps[laps["TrackStatus"].astype(str) == "1"]
    # Remove in-laps and out-laps: they contain pit lane time, not race pace.
    laps = laps[laps["PitInTime"].isna() & laps["PitOutTime"].isna()]
    laps = laps[laps["LapTimeS"].notna() & laps["TyreLife"].notna()]
    laps = laps[laps["Compound"].isin(COMPOUNDS)]

    # Drop traffic-affected and damaged laps: anything well off that driver's
    # own best is not a measurement of the tyre.
    keep = []
    for drv, grp in laps.groupby("Driver"):
        cutoff = grp["LapTimeS"].quantile(0.05) * 1.07
        keep.append(grp[grp["LapTimeS"] <= cutoff])
    laps = pd.concat(keep)

    print(f"Laps: {before} total -> {len(laps)} usable green-flag racing laps")
    return laps


def estimate_pit_loss(ses):
    """Time lost in the pit lane, measured from the race rather than assumed."""
    laps = ses.laps.copy()
    laps["LapTimeS"] = laps["LapTime"].dt.total_seconds()
    base = laps["LapTimeS"].quantile(0.20)
    inlaps = laps[laps["PitInTime"].notna()]["LapTimeS"].median()
    outlaps = laps[laps["PitOutTime"].notna()]["LapTimeS"].median()
    return float((inlaps - base) + (outlaps - base))


# ----------------------------------------------------------------------
# 2. Fit degradation and fuel effect together
# ----------------------------------------------------------------------
def fit_model(laps):
    """
    Least squares fit of:
        laptime = base(compound) + deg(compound)*tyre_age + fuel*lap_number

    Solved as one linear system so the fuel term cannot be absorbed into the
    degradation slopes.
    """
    drivers = sorted(laps["Driver"].unique())
    nd, nc = len(drivers), len(COMPOUNDS)

    # Columns: one intercept per driver, then a pace offset for every compound
    # except the reference, then a degradation slope per compound, then fuel.
    #
    # The driver intercepts matter. Without them the fit is comparing a Ferrari
    # on hards against a midfield car on softs and calling the difference tyre
    # behaviour. Giving every driver their own baseline means the compound
    # terms describe the tyre rather than the car.
    rows, y = [], []
    for _, lp in laps.iterrows():
        row = [0.0] * (nd + (nc - 1) + nc + 1)
        row[drivers.index(lp["Driver"])] = 1.0
        c = COMPOUNDS.index(lp["Compound"])
        if c > 0:
            row[nd + c - 1] = 1.0                       # offset vs SOFT
        row[nd + (nc - 1) + c] = float(lp["TyreLife"])  # degradation slope
        row[-1] = float(lp["LapNumber"])                # fuel burn
        rows.append(row); y.append(lp["LapTimeS"])

    A = np.array(rows); y = np.array(y)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)

    ref = float(np.mean(coef[:nd]))                     # average car baseline
    offsets = [0.0] + list(coef[nd:nd + nc - 1])
    model = {
        "base": {c: ref + offsets[i] for i, c in enumerate(COMPOUNDS)},
        "deg":  {c: coef[nd + (nc - 1) + i] for i, c in enumerate(COMPOUNDS)},
        "fuel": coef[-1],
        "drivers": {d: coef[i] for i, d in enumerate(drivers)},
    }
    pred = A @ coef
    model["rmse"] = float(np.sqrt(np.mean((pred - y) ** 2)))
    return model


def stint_time(model, compound, laps_in_stint, start_lap):
    """Predicted total time for a stint, fuel effect included."""
    ages = np.arange(1, laps_in_stint + 1)
    lapnums = start_lap + ages - 1
    return float(np.sum(model["base"][compound]
                        + model["deg"][compound] * ages
                        + model["fuel"] * lapnums))


def best_strategies(model, total_laps, pit_loss, max_stops=2, min_stint=8):
    """
    Enumerate every legal strategy and rank by predicted race time.
    F1 rules require at least two different compounds in a dry race, which is
    applied here as a hard constraint rather than an afterthought.
    """
    out = []
    for stops in (1, max_stops):
        for combo in product(COMPOUNDS, repeat=stops + 1):
            if len(set(combo)) < 2:
                continue                      # must use two compounds
            for splits in _splits(total_laps, stops + 1, min_stint):
                t, lap = 0.0, 1
                for compound, n in zip(combo, splits):
                    t += stint_time(model, compound, n, lap)
                    lap += n
                t += pit_loss * stops
                out.append({"strategy": " > ".join(c[0] for c in combo),
                            "stints": splits, "stops": stops, "time": t})
    return pd.DataFrame(out).sort_values("time").reset_index(drop=True)


def _splits(total, parts, minimum, step=1):
    """All ways to divide the race into stints of at least `minimum` laps."""
    if parts == 1:
        if total >= minimum:
            yield (total,)
        return
    for n in range(minimum, total - minimum * (parts - 1) + 1, step):
        for rest in _splits(total - n, parts - 1, minimum, step):
            yield (n,) + rest


# ----------------------------------------------------------------------
# 3. Run
# ----------------------------------------------------------------------
def main():
    ses = fastf1.get_session(YEAR, EVENT, "R")
    ses.load(telemetry=False, weather=False, messages=False)

    laps = load_clean_laps(ses)
    model = fit_model(laps)
    pit_loss = estimate_pit_loss(ses)
    total_laps = int(ses.laps["LapNumber"].max())

    print(f"\n{YEAR} {ses.event['EventName']}, {total_laps} laps")
    print(f"Measured pit loss      : {pit_loss:.1f} s")
    print(f"Fuel effect            : {model['fuel']:.3f} s per lap "
          f"({abs(model['fuel'])*total_laps:.1f} s across the race)")
    print(f"Model fit RMSE         : {model['rmse']:.3f} s\n")
    print(f"{'Compound':<9}{'base pace':>11}{'deg per lap':>14}{'over 20 laps':>14}")
    for c in COMPOUNDS:
        print(f"{c:<9}{model['base'][c]:>10.2f}s{model['deg'][c]:>13.3f}s"
              f"{model['deg'][c]*20:>13.2f}s")

    # winner's actual strategy
    win = ses.results.iloc[0]["Abbreviation"]
    wl = ses.laps.pick_drivers(win)
    actual = (wl.groupby(["Stint", "Compound"])["LapNumber"]
                .agg(["count", "min", "max"]).reset_index()
                .sort_values("min"))
    print(f"\nWinner {win} actual strategy:")
    for _, s in actual.iterrows():
        print(f"   {s['Compound']:<7} laps {int(s['min']):>2}-{int(s['max']):<2} "
              f"({int(s['count'])} laps)")

    table = best_strategies(model, total_laps, pit_loss)
    print(f"\nStrategies evaluated   : {len(table)}")
    print("Model's fastest five:")
    for _, r in table.head(5).iterrows():
        print(f"   {r['strategy']:<10} {str(r['stints']):<16} "
              f"{r['time']:.1f} s  ({r['stops']} stop)")

    actual_combo = " > ".join(c[0] for c in actual["Compound"])
    actual_splits = tuple(int(n) for n in actual["count"])
    actual_pred = sum(stint_time(model, c, n, int(s))
                      for c, n, s in zip(actual["Compound"], actual_splits, actual["min"])) \
                  + pit_loss * (len(actual_splits) - 1)
    print(f"\nWinner's strategy predicted: {actual_pred:.1f} s")
    print(f"Model optimum              : {table.iloc[0]['time']:.1f} s "
          f"({table.iloc[0]['time'] - actual_pred:+.1f} s)")

    # ---------------- Figure 1: degradation ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    for c in COMPOUNDS:
        sub = laps[laps["Compound"] == c]
        if sub.empty:
            continue
        # strip the fuel effect so the tyre behaviour is visible on its own
        corrected = sub["LapTimeS"] - model["fuel"] * sub["LapNumber"]
        ax.scatter(sub["TyreLife"], corrected, s=9, alpha=0.35, color=COLOURS[c], label=f"{c} (data)")
        ages = np.arange(1, sub["TyreLife"].max() + 1)
        ax.plot(ages, model["base"][c] + model["deg"][c] * ages,
                color=COLOURS[c], lw=2.4)
    ax.set_xlabel("Tyre age (laps)"); ax.set_ylabel("Fuel-corrected lap time (s)")
    ax.set_title("Degradation by compound, fuel effect removed")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ages = np.arange(1, 31)
    for c in COMPOUNDS:
        ax.plot(ages, model["deg"][c] * ages, color=COLOURS[c], lw=2.4, label=c)
    ax.set_xlabel("Laps on the tyre"); ax.set_ylabel("Time lost to degradation (s)")
    ax.set_title("Cumulative cost of staying out")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGS / "f1-degradation.png", dpi=150); plt.close(fig)

    # ---------------- Figure 2: strategy space ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    # The pit window: for each possible first stop, the best race time still
    # achievable afterwards. A flat basin means the stop is not time critical,
    # and steep walls mean it is.
    ax = axes[0]
    ref = table.iloc[0]["time"]
    for stops, colour, label in [(1, "#5d6d7e", "1 stop"), (2, "#2e7d32", "2 stops")]:
        sub = table[table["stops"] == stops].copy()
        sub["first_stop"] = sub["stints"].apply(lambda s: s[0])
        curve = sub.groupby("first_stop")["time"].min()
        ax.plot(curve.index, curve.values - ref, color=colour, lw=2.2, label=label)
    ax.axvline(int(actual_splits[0]), color="#c0392b", ls="--", lw=1.6,
               label=f"{win} stopped on lap {int(actual_splits[0])}")
    ax.set_xlabel("Lap of the first pit stop")
    ax.set_ylabel("Best achievable race time, lost vs optimum (s)")
    ax.set_title("The pit window")
    ax.set_ylim(-1, 40); ax.legend(fontsize=8.5); ax.grid(alpha=0.3)

    ax = axes[1]
    one = table[table["stops"] == 1]["time"].min()
    two = table[table["stops"] == 2]["time"].min()
    bars = ax.bar(["Best 1 stop", "Best 2 stop", f"{win} actual"],
                  [one - two, 0, actual_pred - two],
                  color=["#5d6d7e", "#2e7d32", "#c0392b"])
    ax.bar_label(bars, fmt="%+.1f s", fontsize=9)
    ax.set_ylabel("Predicted time lost vs best 2 stop (s)")
    ax.set_title("Was the winning strategy the right one?")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(FIGS / "f1-strategy.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
