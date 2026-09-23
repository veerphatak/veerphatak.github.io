function lap_time_simulation
% LAP_TIME_SIMULATION  Quasi-steady lap time simulation with tyre degradation.
%
% Veer Phatak, 2026.
%
% WHAT THIS IS
%   An illustrative, from-scratch implementation of the kind of lap time model
%   I built and used in race engineering with Southampton University Formula
%   Student. It is written for this portfolio using a synthetic track and
%   generic vehicle parameters. It is NOT the team's model and contains no
%   team data.
%
% WHY A LAP TIME MODEL IS USEFUL
%   A student team has very little track time. The value of a model like this
%   is not the absolute lap time it predicts, which will always be somewhat
%   wrong. The value is that it ranks setup changes, so you spend limited
%   running on the changes that actually move the car and arrive with a
%   shortlist instead of a list of guesses.
%
% METHOD
%   Classic three-pass quasi-steady solver over a discretised track:
%
%     1. Cornering limit. At each point the car is limited by lateral grip.
%        Grip grows with speed because downforce grows with speed, so the
%        limit is solved rather than assumed:
%
%            mu*(m*g + 0.5*rho*ClA*v^2)/m = v^2/R
%        =>  v^2*(1/R - mu*0.5*rho*ClA/m) = mu*g
%
%        If the bracket is zero or negative the corner is aero limited and the
%        car is not grip limited there at all.
%
%     2. Forward pass. March forwards applying the lesser of engine force and
%        remaining traction, combining longitudinal and lateral tyre demand
%        through a friction ellipse.
%
%     3. Backward pass. March backwards applying the braking limit through the
%        same ellipse, then take the minimum of the two passes.
%
%   Lap time is the integral of ds/v.
%
%   Tyre degradation is modelled as a loss of peak friction with accumulated
%   distance, which is the simplest form that still reproduces the behaviour
%   that matters: the car gets slower through a stint, and slower faster if
%   you run more downforce and therefore higher tyre loads.
%
% USAGE
%   >> lap_time_simulation
%   Prints the baseline lap time and writes three figures.

ds = 0.5;
[distance, radius] = buildTrack(ds);
car = defaultCar();

[baseTime, speed] = simulateLap(car, radius, car.mu, ds);
fprintf('Baseline lap time: %.2f s\n', baseTime);

figSpeedTrace(distance, radius, speed, baseTime);
figSensitivity(car, radius, ds);
figDegradation(car, radius, ds);

end

% ---------------------------------------------------------------------------
% Vehicle
% ---------------------------------------------------------------------------
function car = defaultCar()
car.mass    = 300.0;    % kg, car plus driver
car.mu      = 1.55;     % peak tyre friction coefficient
car.power   = 60e3;     % W, at the wheels
car.cl_a    = 2.9;      % downforce coefficient times reference area
car.rho     = 1.225;    % kg/m^3
car.brake_g = 1.8;      % peak braking, multiples of g
car.v_max   = 38.0;     % m/s, gearing limit
car.cd_a    = dragFor(car.cl_a);
end

function cdA = dragFor(clA, cd0, k)
% Drag as a function of downforce.
%
% Downforce is not free. A wing that makes more lift also makes more induced
% drag, and induced drag grows with the square of lift, so the penalty
% accelerates as you add wing. Modelling this as linear, which is the easy
% mistake, makes the optimiser ask for infinite downforce.
if nargin < 2, cd0 = 0.60; end
if nargin < 3, k   = 0.09; end
cdA = cd0 + k .* clA.^2;
end

function F = downforce(car, v), F = 0.5 * car.rho * car.cl_a * v.^2; end
function F = dragForce(car, v),  F = 0.5 * car.rho * car.cd_a * v.^2; end
function N = normalLoad(car, v), N = car.mass * 9.81 + downforce(car, v);  end

% ---------------------------------------------------------------------------
% Track: a synthetic circuit of constant-radius corners and straights
% ---------------------------------------------------------------------------
function [distance, radius] = buildTrack(ds)
% Columns: segment length (m), corner radius (m), Inf for a straight.
segments = [200.0 Inf
             45.0 18.0
             60.0 Inf
             30.0  9.0     % tight hairpin
             90.0 Inf
             70.0 32.0     % long fast sweeper
             40.0 Inf
             35.0 12.0
             75.0 Inf
             50.0 22.0
             55.0 Inf];

radius = [];
for i = 1:size(segments,1)
    n = max(1, round(segments(i,1)/ds));
    radius = [radius; repmat(segments(i,2), n, 1)]; %#ok<AGROW>
end
distance = (0:numel(radius)-1)' * ds;
end

% ---------------------------------------------------------------------------
% Solver
% ---------------------------------------------------------------------------
function v = cornerLimitSpeed(car, radius, mu)
% Speed at which the car is exactly on the lateral grip limit.
k       = mu * 0.5 * car.rho * car.cl_a / car.mass;   % aero grip term
bracket = 1.0 ./ radius - k;
v2      = inf(size(radius));
ok      = bracket > 0;
v2(ok)  = mu * 9.81 ./ bracket(ok);
v       = min(sqrt(v2), car.v_max);
end

function [lapTime, speed] = simulateLap(car, radius, mu, ds)
% Three-pass quasi-steady solve.
n       = numel(radius);
vCorner = cornerLimitSpeed(car, radius, mu);

% ---- forward pass: acceleration limited ----
vFwd    = zeros(n,1);
vFwd(1) = vCorner(1);
for i = 1:n-1
    v    = min(vFwd(i), vCorner(i));
    grip = mu * normalLoad(car, v);

    % lateral demand uses up part of the friction circle
    if isfinite(radius(i))
        lat = car.mass * v^2 / radius(i);
    else
        lat = 0.0;
    end
    remaining = sqrt(max(grip^2 - lat^2, 0.0));

    fEngine = car.power / max(v, 1.0);
    fDrive  = min(fEngine, remaining) - dragForce(car, v);
    a       = fDrive / car.mass;
    vFwd(i+1) = min(sqrt(max(v^2 + 2*a*ds, 0.1)), car.v_max);
end

% ---- backward pass: braking limited ----
vBwd    = zeros(n,1);
vBwd(n) = vCorner(n);
for i = n:-1:2
    v    = min(vBwd(i), vCorner(i));
    grip = mu * normalLoad(car, v);
    if isfinite(radius(i))
        lat = car.mass * v^2 / radius(i);
    else
        lat = 0.0;
    end
    remaining = sqrt(max(grip^2 - lat^2, 0.0));

    fBrake = min(remaining, car.brake_g * car.mass * 9.81) + dragForce(car, v);
    a      = fBrake / car.mass;
    vBwd(i-1) = min(sqrt(max(v^2 + 2*a*ds, 0.1)), car.v_max);
end

speed   = min([vFwd, vBwd, vCorner], [], 2);
lapTime = sum(ds ./ speed);
end

function times = stint(car, radius, laps, degPerLap, ds)
% Tyre friction falls with accumulated running, and falls faster when the
% tyres are worked harder. Downforce buys grip on a single lap but charges for
% it across a stint, which is why a qualifying setup and a race setup are not
% the same car.
loadFactor = (car.cl_a / 2.9)^1.5;    % more downforce, more tyre work
times = zeros(laps,1);
for lap = 0:laps-1
    muLap      = car.mu * (1.0 - degPerLap * loadFactor * lap);
    times(lap+1) = simulateLap(car, radius, muLap, ds);
end
end

% ---------------------------------------------------------------------------
% Outputs
% ---------------------------------------------------------------------------
function figSpeedTrace(distance, radius, speed, lapTime)
f = figure('Color','w','Position',[100 100 1000 420]);
ax = axes(f); hold(ax,'on'); grid(ax,'on'); box(ax,'on');

% shade the corners so the trace can be read against the track
inCorner = isfinite(radius);
edges = diff([0; inCorner; 0]);
starts = find(edges == 1); stops = find(edges == -1) - 1;
yl = [0 max(speed)*1.12];
for i = 1:numel(starts)
    x = [distance(starts(i)) distance(stops(i)) distance(stops(i)) distance(starts(i))];
    patch(ax, x, [yl(1) yl(1) yl(2) yl(2)], [0.90 0.92 0.96], ...
          'EdgeColor','none','HandleVisibility','off');
end
plot(ax, distance, speed, 'LineWidth', 1.8, 'Color', [0.11 0.31 0.55]);
ylim(ax, yl); xlim(ax, [0 distance(end)]);
xlabel(ax,'Distance around the lap (m)'); ylabel(ax,'Speed (m/s)');
title(ax, sprintf('Speed trace, lap time %.2f s (shaded bands are corners)', lapTime));
exportgraphics(f, fullfile(outDir(),'lapsim-speed-trace.png'), 'Resolution', 150);
close(f);
end

function figSensitivity(car, radius, ds)
clAs  = 2.0:0.25:5.0;
quick = zeros(size(clAs)); race = zeros(size(clAs));
for i = 1:numel(clAs)
    c = car; c.cl_a = clAs(i); c.cd_a = dragFor(clAs(i));
    quick(i) = simulateLap(c, radius, c.mu, ds);
    race(i)  = mean(stint(c, radius, 12, 0.012, ds));
end

masses = 260:10:340;
mTimes = zeros(size(masses));
for i = 1:numel(masses)
    c = car; c.mass = masses(i);
    mTimes(i) = simulateLap(c, radius, c.mu, ds);
end
p = polyfit(masses, mTimes, 1);

f = figure('Color','w','Position',[100 100 1100 420]);
t = tiledlayout(f,1,2,'Padding','compact','TileSpacing','compact');

ax1 = nexttile(t); hold(ax1,'on'); grid(ax1,'on'); box(ax1,'on');
plot(ax1, clAs, quick, '-o','LineWidth',1.6,'MarkerSize',4,'Color',[0.11 0.31 0.55],'DisplayName','Single lap');
plot(ax1, clAs, race,  '-s','LineWidth',1.6,'MarkerSize',4,'Color',[0.78 0.20 0.16],'DisplayName','12 lap average');
[~,iq] = min(quick); [~,ir] = min(race);
plot(ax1, clAs(iq), quick(iq), 'o','MarkerSize',9,'LineWidth',1.6,'Color',[0.11 0.31 0.55],'HandleVisibility','off');
plot(ax1, clAs(ir), race(ir),  's','MarkerSize',9,'LineWidth',1.6,'Color',[0.78 0.20 0.16],'HandleVisibility','off');
xlabel(ax1,'Downforce level, C_LA'); ylabel(ax1,'Lap time (s)');
title(ax1, sprintf('Downforce: quick lap optimum %.2f, race optimum %.2f', clAs(iq), clAs(ir)));
legend(ax1,'Location','north');

ax2 = nexttile(t); hold(ax2,'on'); grid(ax2,'on'); box(ax2,'on');
plot(ax2, masses, mTimes, '-o','LineWidth',1.6,'MarkerSize',4,'Color',[0.20 0.40 0.30]);
xlabel(ax2,'Vehicle mass (kg)'); ylabel(ax2,'Lap time (s)');
title(ax2, sprintf('Mass sensitivity: %.3f s per 10 kg', p(1)*10));

exportgraphics(f, fullfile(outDir(),'lapsim-sensitivity.png'), 'Resolution', 150);
close(f);
end

function figDegradation(car, radius, ds)
levels = [2.5 2.9 3.5 4.2];
f = figure('Color','w','Position',[100 100 900 430]);
ax = axes(f); hold(ax,'on'); grid(ax,'on'); box(ax,'on');
cmap = [0.20 0.45 0.70; 0.11 0.31 0.55; 0.85 0.45 0.10; 0.78 0.20 0.16];
for i = 1:numel(levels)
    c = car; c.cl_a = levels(i); c.cd_a = dragFor(levels(i));
    t = stint(c, radius, 12, 0.012, ds);
    plot(ax, 1:numel(t), t, '-o','LineWidth',1.6,'MarkerSize',4, ...
         'Color', cmap(i,:), 'DisplayName', sprintf('C_LA = %.1f', levels(i)));
end
xlabel(ax,'Lap number'); ylabel(ax,'Lap time (s)');
title(ax,'Tyre degradation through a 12 lap stint');
legend(ax,'Location','northwest');
exportgraphics(f, fullfile(outDir(),'lapsim-degradation.png'), 'Resolution', 150);
close(f);
end

function d = outDir()
d = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'media', 'figures');
if ~exist(d,'dir'), d = pwd; end
end
