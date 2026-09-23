function f1_stint_model
% F1_STINT_MODEL  Tyre degradation, fuel burn and race strategy, fitted to a
%                 real Grand Prix.
%
% Veer Phatak, 2026.
%
% WHAT THIS IS
%   The regression and strategy half of my Formula 1 model, written in MATLAB.
%   The session download stays in Python (f1_stint_model.py) because FastF1 is
%   Python only. That script writes data/bahrain-race-laps.csv, the cleaned
%   green-flag lap set, which is what this file consumes.
%
% THE PROBLEM
%   A lap time in a race is the sum of several effects that all trend the same
%   way over a stint. The car gets lighter as fuel burns, which makes it
%   quicker. The tyre wears, which makes it slower. Fit degradation on its own
%   and the fuel effect is silently absorbed into the tyre slope, which will
%   understate how much the tyre is actually costing.
%
%   So all three are fitted together as one linear system:
%
%       laptime = base(compound) + deg(compound)*tyre_age + fuel*lap_number
%
%   with an intercept per driver. The driver intercepts matter more than they
%   look. Without them the fit compares a front-running car on hards against a
%   midfield car on softs and attributes the difference to the tyre. Giving
%   every driver their own baseline means the compound terms describe the
%   tyre rather than the car.
%
% A STATED LIMITATION
%   The driver fixed effects cut residual error substantially, but this fit
%   still does not cleanly separate the degradation rates of the three
%   compounds. In a one-stop race most drivers run a similar strategy, so
%   compound and stint position are correlated and the data cannot fully tell
%   them apart. That is a property of the race, not something more algebra
%   fixes, and it is worth saying rather than presenting three tidy slopes as
%   if they were measured independently.
%
% USAGE
%   >> f1_stint_model

COMPOUNDS = {'SOFT','MEDIUM','HARD'};
TOTAL_LAPS = 57;        % Bahrain Grand Prix race distance
PIT_LOSS   = 26.083;    % s, measured from the race itself, not assumed

here = fileparts(mfilename('fullpath'));
csvFile = fullfile(here, '..', '..', 'data', 'bahrain-race-laps.csv');
if ~isfile(csvFile)
    error('f1_stint_model:noData', ...
          ['Could not find %s.\nRegenerate it with:  ' ...
           'python code/f1-lapsim/f1_stint_model.py'], csvFile);
end

T = readtable(csvFile, 'TextType','string');
fprintf('\nLoaded %d green-flag racing laps from %d drivers.\n', ...
        height(T), numel(unique(T.Driver)));

model = fitModel(T, COMPOUNDS);

fprintf('\nFitted model\n------------\n');
for i = 1:numel(COMPOUNDS)
    fprintf('  %-7s base %7.3f s   degradation %+6.3f s per lap on the tyre\n', ...
            COMPOUNDS{i}, model.base(i), model.deg(i));
end
fprintf('  Fuel effect      %+6.3f s per lap of race distance\n', model.fuel);
fprintf('  Residual RMSE     %6.3f s\n', model.rmse);

fprintf('\nWithout driver intercepts, RMSE is %.3f s, so the driver terms\n', ...
        fitModel(T, COMPOUNDS, false).rmse);
fprintf('account for a large share of the scatter that would otherwise be\n');
fprintf('mistaken for tyre behaviour.\n');

best = bestStrategies(model, COMPOUNDS, TOTAL_LAPS, PIT_LOSS);
fprintf('\nStrategy search over %d legal one and two stop strategies\n', best.evaluated);
fprintf('(pit loss %.2f s, minimum stint 8 laps)\n', PIT_LOSS);
fprintf('------------------------------------------------------------\n');
for i = 1:min(5, numel(best.time))
    fprintf('  %2d. %-28s %8.2f s  %+6.2f s\n', i, best.label(i), ...
            best.time(i), best.time(i) - best.time(1));
end

plotDegradation(model, COMPOUNDS);

end

% ---------------------------------------------------------------------------
function model = fitModel(T, COMPOUNDS, useDrivers)
% Least squares fit of base pace, degradation and fuel burn in one system.
if nargin < 3, useDrivers = true; end

drivers = unique(T.Driver);
nd = numel(drivers); nc = numel(COMPOUNDS);
n  = height(T);

if useDrivers, nInt = nd; else, nInt = 1; end
A = zeros(n, nInt + (nc-1) + nc + 1);
y = T.LapTimeS;

[~, driverIdx]   = ismember(T.Driver, drivers);
[~, compoundIdx] = ismember(T.Compound, COMPOUNDS);

for i = 1:n
    if useDrivers
        A(i, driverIdx(i)) = 1.0;      % one baseline per driver
    else
        A(i, 1) = 1.0;                 % single shared baseline
    end
    c = compoundIdx(i);
    if c > 1
        A(i, nInt + c - 1) = 1.0;      % pace offset against SOFT
    end
    A(i, nInt + (nc-1) + c) = T.TyreLife(i);   % degradation slope
    A(i, end) = T.LapNumber(i);                % fuel burn
end

coef = A \ y;

ref     = mean(coef(1:nInt));
offsets = [0; coef(nInt+1 : nInt+nc-1)];

model.base    = ref + offsets;
model.deg     = coef(nInt+(nc-1)+1 : nInt+(nc-1)+nc);
model.fuel    = coef(end);
model.rmse    = sqrt(mean((A*coef - y).^2));
if useDrivers
    model.drivers = containers.Map(cellstr(drivers), num2cell(coef(1:nd)));
else
    model.drivers = containers.Map('KeyType','char','ValueType','double');
end
end

% ---------------------------------------------------------------------------
function t = stintTime(model, c, lapsInStint, startLap)
% Total time for a stint of a given compound, starting at a given race lap.
age  = (0:lapsInStint-1)';
lapN = startLap + age;
t    = sum(model.base(c) + model.deg(c)*age + model.fuel*lapN);
end

function best = bestStrategies(model, COMPOUNDS, totalLaps, pitLoss)
% Exhaustive search over every legal one and two stop strategy.
minStint = 8;
nc = numel(COMPOUNDS);
times = []; labels = strings(0);

for stops = 1:2
    parts = stops + 1;
    splits = enumerateSplits(totalLaps, parts, minStint);
    for s = 1:size(splits,1)
        for combo = 0:nc^parts - 1
            idx = zeros(1,parts); rem = combo;
            for k = 1:parts
                idx(k) = mod(rem, nc) + 1; rem = floor(rem / nc);
            end
            if numel(unique(idx)) < 2, continue; end   % must use two compounds
            t = stops * pitLoss; lap = 1;
            for k = 1:parts
                t   = t + stintTime(model, idx(k), splits(s,k), lap);
                lap = lap + splits(s,k);
            end
            times(end+1,1) = t; %#ok<AGROW>
            lbl = sprintf('%s %d', COMPOUNDS{idx(1)}, splits(s,1));
            for k = 2:parts
                lbl = sprintf('%s / %s %d', lbl, COMPOUNDS{idx(k)}, splits(s,k));
            end
            labels(end+1,1) = string(lbl); %#ok<AGROW>
        end
    end
end

[times, order] = sort(times);
best.time      = times;
best.label     = labels(order);
best.evaluated = numel(times);
end

function splits = enumerateSplits(total, parts, minimum)
% All ways of splitting a race into stints of at least `minimum` laps.
if parts == 1
    if total >= minimum, splits = total; else, splits = zeros(0,1); end
    return
end
splits = zeros(0, parts);
for first = minimum : total - minimum*(parts-1)
    rest = enumerateSplits(total - first, parts-1, minimum);
    splits = [splits; repmat(first, size(rest,1), 1), rest]; %#ok<AGROW>
end
end

% ---------------------------------------------------------------------------
function plotDegradation(model, COMPOUNDS)
age = 0:25;
cmap = [0.75 0.23 0.17; 0.83 0.67 0.05; 0.36 0.43 0.49];
f  = figure('Color','w','Position',[100 100 900 430]);
ax = axes(f); hold(ax,'on'); grid(ax,'on'); box(ax,'on');
for i = 1:numel(COMPOUNDS)
    plot(ax, age, model.deg(i)*age, 'LineWidth', 1.8, 'Color', cmap(i,:), ...
         'DisplayName', sprintf('%s  (%+.3f s/lap)', COMPOUNDS{i}, model.deg(i)));
end
xlabel(ax,'Laps on the tyre'); ylabel(ax,'Time lost to degradation (s)');
title(ax,'Fitted tyre degradation, fuel effect removed');
legend(ax,'Location','northwest');
out = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'media', 'figures');
if isfolder(out)
    exportgraphics(f, fullfile(out,'f1-degradation-matlab.png'), 'Resolution', 150);
end
close(f);
end
