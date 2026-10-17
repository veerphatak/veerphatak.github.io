function siteAxes(ax, P)
%SITEAXES  Apply the website's figure styling to one axes.
%
%   Recedes the grid and the axis furniture so the data is the only thing
%   advancing, and puts text on text colours rather than on series colours.
%
%   Pair with a figure created as:
%       f = figure('Color', P.surface, ...);

    if nargin < 2, P = sitePalette(); end

    set(ax, 'Color',      P.surface, ...
            'XColor',     P.muted, ...
            'YColor',     P.muted, ...
            'GridColor',  P.border, ...
            'GridAlpha',  0.9, ...
            'MinorGridColor', P.border, ...
            'LineWidth',  1.0, ...
            'FontSize',   9, ...
            'Layer',      'top', ...
            'Box',        'off', ...
            'TickDir',    'out');

    % Otherwise MATLAB bakes the interactive axes toolbar into the exported
    % PNG, which is a screenshot artefact and not part of the figure.
    if isprop(ax, 'Toolbar') && ~isempty(ax.Toolbar)
        ax.Toolbar.Visible = 'off';
    end

    ax.Title.Color  = P.text;
    ax.XLabel.Color = P.muted;
    ax.YLabel.Color = P.muted;
    ax.Title.FontWeight = 'normal';
    ax.Title.FontSize   = 11.5;

    % Legends sit outside the axes, so they cannot land on the data. Once
    % outside, the surrounding box is doing nothing but adding a line.
    lg = get(ax, 'Legend');
    if ~isempty(lg)
        set(lg, 'Color', 'none', 'TextColor', P.body, ...
                'EdgeColor', 'none', 'Box', 'off');
    end
end
