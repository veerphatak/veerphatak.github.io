"""
Quasi-steady lap time simulation with tyre degradation
======================================================

Veer Phatak, 2026.

WHAT THIS IS
------------
An illustrative, from-scratch implementation of the kind of lap time model I
built and used in race engineering with Southampton University Formula Student.
It is written for this portfolio using a synthetic track and generic vehicle
parameters. It is NOT the team's model and contains no team data.

WHY A LAP TIME MODEL IS USEFUL
------------------------------
A student team has very little track time. The value of a model like this is
not the absolute lap time it predicts, which will always be somewhat wrong.
The value is that it ranks setup changes, so you can spend limited running on
the changes that actually move the car, and arrive with a shortlist instead of
a list of guesses.

METHOD
------
Classic three-pass quasi-steady solver over a discretised track:

  1. Cornering limit. At each point the car is limited by lateral grip. Grip
     grows with speed because downforce grows with speed, so the limit is
     solved rather than assumed:

         mu * (m*g + 0.5*rho*Cl*A*v^2) / m  =  v^2 / R
     =>  v^2 * (1/R - mu*0.5*rho*Cl*A/m)   =  mu*g

     If the bracket is zero or negative the corner is aero limited and the car
     is not grip limited there at all.

  2. Forward pass (acceleration). March forwards applying the lesser of engine
     force and remaining traction, with longitudinal and lateral tyre demand
     combined through a friction ellipse.

  3. Backward pass (braking). March backwards applying the braking limit, again
     through the friction ellipse, then take the minimum of the two passes.

Lap time is then the integral of ds / v.

Tyre degradation is modelled as a loss of peak friction with accumulated
distance, which is the simplest form that still reproduces the behaviour that
matters: the car gets slower through a stint, and it gets slower faster if you
run more downforce and therefore higher tyre loads.
"""

from dataclasses import dataclass
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Vehicle
# ----------------------------------------------------------------------
@dataclass
class Car:
    mass: float = 300.0          # kg, car plus driver
    mu: float = 1.55             # peak tyre friction coefficient
    power: float = 60_000.0      # W, at the wheels
    cl_a: float = 2.9            # downforce coefficient times reference area
    cd_a: float = 1.36           # drag coefficient times reference area (see drag_for)
    rho: float = 1.225           # kg/m^3
    brake_g: float = 1.8         # peak braking, multiples of g
    v_max: float = 38.0          # m/s, gearing limit

    def downforce(self, v):
        return 0.5 * self.rho * self.cl_a * v ** 2

    def drag(self, v):
        return 0.5 * self.rho * self.cd_a * v ** 2

    def normal_load(self, v):
        return self.mass * 9.81 + self.downforce(v)


def drag_for(cl_a, cd0=0.60, k=0.09):
    """
    Drag as a function of downforce.

    Downforce is not free. A wing that makes more lift also makes more induced
    drag, and induced drag grows with the square of lift, so the penalty
    accelerates as you add wing. Modelling this as linear, which is the easy
    mistake, makes the optimiser ask for infinite downforce.
    """
    return cd0 + k * cl_a ** 2


# ----------------------------------------------------------------------
# Track: a synthetic circuit built from constant-radius corners and straights
# ----------------------------------------------------------------------
def build_track(ds=0.5):
    """Return (distance, radius) arrays. Radius is inf on a straight."""
    segments = [
        ("straight", 200.0, None),
        ("corner",    45.0, 18.0),
        ("straight",  60.0, None),
        ("corner",    30.0,  9.0),    # tight hairpin
        ("straight",  90.0, None),
        ("corner",    70.0, 32.0),    # long fast sweeper
        ("straight",  40.0, None),
        ("corner",    35.0, 12.0),
        ("straight",  75.0, None),
        ("corner",    50.0, 22.0),
        ("straight",  55.0, None),
    ]
    radius = []
    for _, length, r in segments:
        n = max(1, int(round(length / ds)))
        radius.extend([np.inf if r is None else r] * n)
    radius = np.array(radius, dtype=float)
    distance = np.arange(len(radius)) * ds
    return distance, radius


# ----------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------
def corner_limit_speed(car, radius, mu):
    """Speed at which the car is exactly on the lateral grip limit."""
    k = mu * 0.5 * car.rho * car.cl_a / car.mass       # aero grip term
    with np.errstate(divide="ignore", invalid="ignore"):
        bracket = 1.0 / radius - k
        v2 = np.where(bracket > 0, mu * 9.81 / bracket, np.inf)
    return np.minimum(np.sqrt(v2), car.v_max)


def simulate_lap(car, distance, radius, mu=None, ds=0.5):
    """Three-pass quasi-steady solve. Returns (lap_time, speed trace)."""
    mu = car.mu if mu is None else mu
    n = len(radius)
    v_corner = corner_limit_speed(car, radius, mu)

    # ---- forward pass: acceleration limited ----
    v_fwd = np.empty(n)
    v_fwd[0] = v_corner[0]
    for i in range(n - 1):
        v = min(v_fwd[i], v_corner[i])
        fz = car.normal_load(v)
        grip = mu * fz

        # lateral demand uses up part of the friction circle
        lat = car.mass * v ** 2 / radius[i] if np.isfinite(radius[i]) else 0.0
        remaining = max(grip ** 2 - lat ** 2, 0.0) ** 0.5

        f_engine = car.power / max(v, 1.0)
        f_drive = min(f_engine, remaining) - car.drag(v)
        a = f_drive / car.mass
        v_fwd[i + 1] = min(np.sqrt(max(v ** 2 + 2 * a * ds, 0.1)), car.v_max)

    # ---- backward pass: braking limited ----
    v_bwd = np.empty(n)
    v_bwd[-1] = v_corner[-1]
    for i in range(n - 1, 0, -1):
        v = min(v_bwd[i], v_corner[i])
        fz = car.normal_load(v)
        grip = mu * fz
        lat = car.mass * v ** 2 / radius[i] if np.isfinite(radius[i]) else 0.0
        remaining = max(grip ** 2 - lat ** 2, 0.0) ** 0.5

        f_brake = min(remaining, car.brake_g * car.mass * 9.81) + car.drag(v)
        a = f_brake / car.mass
        v_bwd[i - 1] = min(np.sqrt(max(v ** 2 + 2 * a * ds, 0.1)), car.v_max)

    speed = np.minimum.reduce([v_fwd, v_bwd, v_corner])
    lap_time = float(np.sum(ds / speed))
    return lap_time, speed


def stint(car, distance, radius, laps=12, deg_per_lap=0.012, ds=0.5):
    """
    Tyre friction falls with accumulated running, and falls faster when the
    tyres are worked harder. Downforce buys grip on a single lap but charges
    for it across a stint, which is why a qualifying setup and a race setup
    are not the same car.
    """
    load_factor = (car.cl_a / 2.9) ** 1.5   # more downforce, more tyre work
    times = []
    for lap in range(laps):
        mu_lap = car.mu * (1.0 - deg_per_lap * load_factor * lap)
        t, _ = simulate_lap(car, distance, radius, mu=mu_lap, ds=ds)
        times.append(t)
    return np.array(times)


# ----------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------
def main():
    ds = 0.5
    distance, radius = build_track(ds)
    car = Car()

    base_time, speed = simulate_lap(car, distance, radius, ds=ds)
    print(f"Baseline lap time: {base_time:.2f} s")

    # --- Figure 1: speed trace -------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(distance, speed, lw=2, color="#14314b", label="Simulated speed")
    ax.plot(distance, corner_limit_speed(car, radius, car.mu),
            lw=1, ls="--", color="#c0392b", label="Lateral grip limit")
    corner = np.isfinite(radius)
    ax.fill_between(distance, 0, 40, where=corner, color="#cfd8dc",
                    alpha=0.5, step="mid", label="Corner")
    ax.set_xlabel("Distance around lap (m)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title(f"Simulated speed trace, lap time {base_time:.2f} s")
    ax.set_ylim(0, 40)
    ax.set_xlim(distance[0], distance[-1])
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("../media/figures/lapsim-speed-trace.png", dpi=150)

    # --- Figure 2: tyre degradation over a stint ------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for deg, label, col in [(0.006, "Low degradation", "#2e7d32"),
                            (0.012, "Baseline", "#14314b"),
                            (0.020, "High degradation", "#c0392b")]:
        t = stint(car, distance, radius, laps=12, deg_per_lap=deg, ds=ds)
        ax.plot(np.arange(1, len(t) + 1), t, marker="o", ms=4, color=col, label=label)
    ax.set_xlabel("Lap number")
    ax.set_ylabel("Lap time (s)")
    ax.set_title("Lap time through a stint as tyre grip falls")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("../media/figures/lapsim-degradation.png", dpi=150)

    # --- Figure 3: setup sensitivity ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    cl_values = np.linspace(2.0, 5.0, 13)
    one_lap, stint_avg = [], []
    for cl in cl_values:
        c = Car(cl_a=cl, cd_a=drag_for(cl))        # downforce costs drag, and it costs it quadratically
        one_lap.append(simulate_lap(c, distance, radius, ds=ds)[0])
        stint_avg.append(stint(c, distance, radius, laps=12, ds=ds).mean())
    axes[0].plot(cl_values, one_lap, marker="o", ms=4, color="#14314b", label="Single lap")
    axes[0].plot(cl_values, stint_avg, marker="s", ms=4, color="#c0392b", label="12 lap average")
    axes[0].set_xlabel("Downforce level, $C_L A$")
    axes[0].set_ylabel("Lap time (s)")
    axes[0].set_title("Downforce: quick lap against race pace")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    masses = np.linspace(260, 340, 9)
    m_times = [simulate_lap(Car(mass=m), distance, radius, ds=ds)[0] for m in masses]
    axes[1].plot(masses, m_times, marker="o", ms=4, color="#14314b")
    grad = (m_times[-1] - m_times[0]) / (masses[-1] - masses[0])
    axes[1].set_xlabel("Vehicle mass (kg)")
    axes[1].set_ylabel("Lap time (s)")
    axes[1].set_title(f"Mass sensitivity: {grad*10:.3f} s per 10 kg")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("../media/figures/lapsim-sensitivity.png", dpi=150)

    print(f"Mass sensitivity: {grad*10:.3f} s per 10 kg")
    print(f"Best single-lap downforce: CLA = {cl_values[int(np.argmin(one_lap))]:.2f}")
    print(f"Best 12-lap-average downforce: CLA = {cl_values[int(np.argmin(stint_avg))]:.2f}")


if __name__ == "__main__":
    main()
