function C = siteShades(accentRGB, k)
%SITESHADES  k steps of one hue, for an ordered sweep.
%
%   C = siteShades(P.accent.fs, 4) returns a kx3 RGB matrix running from a
%   dim step to the accent.
%
%   Ordered data (four downforce levels, five tiers, a time sequence) gets one
%   hue stepped by lightness. Giving it k unrelated hues implies the levels are
%   separate categories with no order between them, which is false and throws
%   away the one thing the reader most needs to see.
%
%   The low end anchors on the border step rather than on the page surface, so
%   the dimmest line recedes without disappearing into the background.

    if nargin < 2, k = 4; end
    P = sitePalette();
    lo = P.border;
    if k == 1
        C = accentRGB;
        return
    end
    t = linspace(0.30, 1.0, k)';
    C = (1 - t) .* lo + t .* accentRGB;
end
