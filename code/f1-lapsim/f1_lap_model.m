function f1_lap_model
% F1_LAP_MODEL  Quasi-steady Formula 1 lap time model, validated against a
%               real qualifying lap.
%
% Veer Phatak, 2026.
%
% WHAT THIS IS
%   The physics half of my Formula 1 lap time model, written in MATLAB. It
%   solves a lap from circuit geometry and a small set of vehicle parameters,
%   then compares the result against the measured pole lap.
%
%   The telemetry download and the circuit geometry extraction stay in Python
%   (f1_lap_model.py), because FastF1, the library that reads the official
%   timing and telemetry feed, is Python only. That script writes
%   data/bahrain-lap.json, which is what this file consumes. Everything from
%   the solver onwards is reproduced here.
%
% METHOD
%   Three-pass quasi-steady solve over the discretised lap.
%
%     1. Corner limit. Lateral grip is solved rather than assumed, because
%        downforce, and therefore grip, grows with speed:
%
%            mu*(m*g + 0.5*rho*ClA*v^2)/m = v^2/R
%        =>  v^2*(1/R - mu*0.5*rho*ClA/m) = mu*g
%
%        Where the bracket is non-positive the corner is aero limited and the
%        car is never grip limited there. On an F1 car at Bahrain that is true
%        through the quick corners, which is the whole point of the aero.
%
%     2. Forward pass. Traction limited acceleration, with lateral and
%        longitudinal tyre demand combined through a friction ellipse, less
%        aerodynamic drag. Drag is reduced where DRS is open.
%
%     3. Backward pass. Braking limited deceleration through the same ellipse.
%
%   Lap time is the integral of ds/v.
%
% VALIDATION
%   Against the 2025 Bahrain Grand Prix pole lap the model predicts the lap
%   time to within about 0.7%. That is close enough to be useful for ranking
%   changes and nowhere near close enough to be treated as truth, which is the
%   correct way to read any lap time model.
%
%   Note on numbers. This file solves the published geometry in
%   data/bahrain-lap.json, which is downsampled to 4 m steps so it stays small
%   enough to ship with the site. The Python script solves the full-resolution
%   trace before downsampling, so it reports 89.20 s where this reports
%   89.31 s. Both are the same model; the difference is step size, and the
%   fact that it is that small is itself a check that the solve has converged.
%
% USAGE
%   >> f1_lap_model

here = fileparts(mfilename('fullpath'));
dataFile = fullfile(here, '..', '..', 'data', 'bahrain-lap.json');
if ~isfile(dataFile)
    error('f1_lap_model:noData', ...
          ['Could not find %s.\nRegenerate it with:  ' ...
           'python code/f1-lapsim/f1_lap_model.py'], dataFile);
end

d = jsondecode(fileread(dataFile));

R      = d.radius(:);
R(~isfinite(R) | R <= 0) = Inf;      % straights come through as null/zero
vReal  = d.speedReal(:) / 3.6;       % km/h in the file, m/s in the solver
drs    = double(d.drs(:) > 0);
ds     = d.ds;

car = struct('mass',    d.fitted.mass, ...
             'mu',      d.fitted.mu, ...
             'power',   d.fitted.power, ...
             'cl_a',    d.fitted.clA, ...
             'cd_a',    d.fitted.cdA, ...
             'brake_g', d.fitted.brakeG, ...
             'v_max',   d.fitted.vMax, ...
             'rho',     1.225, ...
             'drs_drag_cut', 0.12);

[lapTime, v] = solveLap(car, R, drs, ds);

realTime = d.realLapTime;
rmse     = sqrt(mean((v - vReal).^2)) * 3.6;

fprintf('\n%s, %s (%s, %s)\n', d.circuit, d.session, d.driver, d.team);
fprintf('  Measured lap time : %8.3f s\n', realTime);
fprintf('  Model lap time    : %8.3f s\n', lapTime);
fprintf('  Error             : %+8.3f s  (%+.2f%%)\n', ...
        lapTime - realTime, 100*(lapTime-realTime)/realTime);
fprintf('  Speed trace RMSE  : %8.1f km/h\n', rmse);
fprintf('  Fitted: mu=%.3f  ClA=%.2f  CdA=%.3f\n\n', car.mu, car.cl_a, car.cd_a);

plotTrace(d.distance(:), v, vReal, drs, lapTime, realTime);

end

% ---------------------------------------------------------------------------
function F = dragForce(car, v, drs)
cd = car.cd_a * (1.0 - car.drs_drag_cut * drs);
F  = 0.5 * car.rho * cd * v.^2;
end

function N = normalLoad(car, v)
N = car.mass * 9.81 + 0.5 * car.rho * car.cl_a * v.^2;
end

function v = cornerLimit(car, R)
% Speed where lateral demand exactly equals available grip.
k       = car.mu * 0.5 * car.rho * car.cl_a / car.mass;
bracket = 1.0 ./ R - k;
v2      = inf(size(R));
ok      = bracket > 0;
v2(ok)  = car.mu * 9.81 ./ bracket(ok);
v       = min(sqrt(max(v2, 0.0)), car.v_max);
end

function [lapTime, v] = solveLap(car, R, drs, ds)
n       = numel(R);
vCorner = cornerLimit(car, R);

vF = zeros(n,1); vF(1) = vCorner(1);
for i = 1:n-1
    vi   = min(vF(i), vCorner(i));
    grip = car.mu * normalLoad(car, vi);
    if isfinite(R(i)), lat = car.mass * vi^2 / R(i); else, lat = 0.0; end
    remaining = sqrt(max(grip^2 - lat^2, 0.0));
    f = min(car.power / max(vi,1.0), remaining) - dragForce(car, vi, drs(i));
    vF(i+1) = min(sqrt(max(vi^2 + 2*(f/car.mass)*ds, 1.0)), car.v_max);
end

vB = zeros(n,1); vB(n) = vCorner(n);
for i = n:-1:2
    vi   = min(vB(i), vCorner(i));
    grip = car.mu * normalLoad(car, vi);
    if isfinite(R(i)), lat = car.mass * vi^2 / R(i); else, lat = 0.0; end
    remaining = sqrt(max(grip^2 - lat^2, 0.0));
    f = min(remaining, car.brake_g * car.mass * 9.81) + dragForce(car, vi, drs(i));
    vB(i-1) = min(sqrt(max(vi^2 + 2*(f/car.mass)*ds, 1.0)), car.v_max);
end

v       = min([vF, vB, vCorner], [], 2);
lapTime = sum(ds ./ v);
end

% ---------------------------------------------------------------------------
function plotTrace(dist, v, vReal, drs, lapTime, realTime)
f  = figure('Color','w','Position',[100 100 1100 460]);
ax = axes(f); hold(ax,'on'); grid(ax,'on'); box(ax,'on');

yl = [0 max([v; vReal])*3.6*1.1];
edges  = diff([0; drs > 0; 0]);
starts = find(edges == 1); stops = find(edges == -1) - 1;
for i = 1:numel(starts)
    x = [dist(starts(i)) dist(stops(i)) dist(stops(i)) dist(starts(i))];
    patch(ax, x, [yl(1) yl(1) yl(2) yl(2)], [0.85 0.92 1.00], ...
          'EdgeColor','none','HandleVisibility','off');
end

plot(ax, dist, vReal*3.6, 'LineWidth', 1.7, 'Color', [0.11 0.31 0.55], ...
     'DisplayName', sprintf('Measured  %.3f s', realTime));
plot(ax, dist, v*3.6, 'LineWidth', 1.5, 'Color', [0.78 0.20 0.16], ...
     'DisplayName', sprintf('Model     %.3f s', lapTime));
xlim(ax, [0 dist(end)]); ylim(ax, yl);
xlabel(ax, 'Distance around the lap (m)'); ylabel(ax, 'Speed (km/h)');
title(ax, 'Model against measured pole lap (shaded bands are DRS open)');
legend(ax, 'Location', 'southeast');

out = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'media', 'figures');
if isfolder(out)
    exportgraphics(f, fullfile(out, 'f1-speed-trace-matlab.png'), 'Resolution', 150);
end
close(f);
end
