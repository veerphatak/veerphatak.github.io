bahrain-lap.json
----------------
Circuit geometry and a single reference lap, used by the interactive lap
simulator on the website.

The corner radius values are derived geometry, computed from car position
telemetry by the script in code/f1-lapsim/. The reference speed trace is from
the 2025 Bahrain Grand Prix qualifying session and belongs to Formula 1. It is
included here only as a comparison curve for the simulator, for non-commercial
purposes, and accessed originally through the open-source FastF1 library.

Regenerate with:  python code/f1-lapsim/f1_lap_model.py


bahrain-race-laps.csv
---------------------
The 979 green-flag racing laps from the 2025 Bahrain Grand Prix that survive
cleaning: safety car laps, in-laps, out-laps and traffic-affected laps are all
removed. Columns are driver, compound, tyre age, lap number and lap time.

This is derived timing data, used as the input to the degradation and strategy
fit. It exists so the MATLAB implementation can be run without a FastF1
install or a network connection. The underlying timing data belongs to Formula
1 and is included here only for non-commercial illustration, accessed
originally through the open-source FastF1 library.

Regenerate with:  python code/f1-lapsim/f1_stint_model.py
