function P = sitePalette()
%SITEPALETTE  Figure colours for the website, read from site-palette.json.
%
%   P = sitePalette() returns a struct whose colour fields are 1x3 RGB rows
%   in the 0..1 range MATLAB expects, converted from the same hex values the
%   stylesheet and the Python scripts use.
%
%   One palette file, two languages. The alternative is duplicating the
%   colours here and having them drift the first time the site changes, which
%   is exactly how this repository ended up with figures in a palette that
%   appeared nowhere on the pages they were embedded in.
%
%   Fields:
%     P.page P.surface P.sunken P.border      surfaces
%     P.text P.body P.muted                   ink
%     P.accent.<name>                         per-page accents
%     P.series                                Nx3, categorical in fixed order
%     P.reference                             neutral for measured data
%     P.good P.warn P.bad                     reserved status colours
%
%   Styling helpers are in siteFigure.m and siteAxes.m.

    here = fileparts(mfilename('fullpath'));
    raw  = fileread(fullfile(here, 'site-palette.json'));
    J    = jsondecode(raw);

    P.page    = hex2rgb(J.page);
    P.surface = hex2rgb(J.surface);
    P.sunken  = hex2rgb(J.sunken);
    P.border  = hex2rgb(J.border);

    P.text  = hex2rgb(J.text);
    P.body  = hex2rgb(J.body);
    P.muted = hex2rgb(J.muted);

    names = fieldnames(J.accents);
    for k = 1:numel(names)
        P.accent.(names{k}) = hex2rgb(J.accents.(names{k}));
    end

    P.series = cell2mat(cellfun(@hex2rgb, J.series, 'UniformOutput', false));

    P.reference = hex2rgb(J.reference);
    P.good = hex2rgb(J.good);
    P.warn = hex2rgb(J.warn);
    P.bad  = hex2rgb(J.bad);
end

function rgb = hex2rgb(h)
    h = char(h);
    if h(1) == '#', h(1) = []; end
    rgb = double([hex2dec(h(1:2)), hex2dec(h(3:4)), hex2dec(h(5:6))]) / 255;
end
