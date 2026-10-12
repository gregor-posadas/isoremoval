# Colour-vision check for the isoremoval and basin-design plots (adapted from lagmay-visit-hub).
# For every pair of colours that has to be told apart, and every colour against the background it sits on,
# simulates protan, deutan and tritan vision (Machado et al. 2009, full severity, via colorspacious) and reports
# the CAM02-UCS distance (about 10+ reads as clearly different, under about 6 is risky) and, for marks on a
# background, the WCAG contrast ratio (3:1 is the floor for graphics).
# pip install colorspacious
import itertools, sys
from colorspacious import cspace_convert

def rgb(h): h = h.lstrip('#'); return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
def sim(h, kind):
    if kind == 'normal': return rgb(h)
    out = cspace_convert(rgb(h), {'name': 'sRGB1+CVD', 'cvd_type': kind, 'severity': 100}, 'sRGB1')
    return [min(1, max(0, v)) for v in out]
def dist(a, b):
    A = cspace_convert(a, 'sRGB1', 'CAM02-UCS'); B = cspace_convert(b, 'sRGB1', 'CAM02-UCS')
    return sum((x - y) ** 2 for x, y in zip(A, B)) ** 0.5
def lum(c):
    f = lambda v: v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = map(f, c); return 0.2126 * r + 0.7152 * g + 0.0722 * b
def contrast(a, b): la, lb = sorted((lum(a), lum(b)), reverse=True); return (la + 0.05) / (lb + 0.05)
KINDS = ['normal', 'protanomaly', 'deuteranomaly', 'tritanomaly']

import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np, matplotlib.colors as mc
import cmcrameri.cm as cmc
def davos(n):
    """The curve colours the app uses (ui_theme.curve_colors), on the white plot background."""
    xs = np.linspace(0.56, 0.02, n)
    return {f'curve {i+1}': mc.to_hex(cmc.davos(x)) for i, x in enumerate(xs)}

SETS = {
  'curves (7), white plot': ('#ffffff', davos(7)),
  'design plots, white': ('#ffffff', {'connecting line': '#857e6e', 'target star (with ink outline)': '#E7B800', 'ink': '#1b1a17'}),
  'basin side view, white': ('#ffffff', {'water': '#dbe6f3', 'sludge': '#8c6b5a', 'path at overflow rate': '#385F96',
                                         'test path': '#B8481A', 'ink': '#1b1a17'}),
  'paths on water': ('#dbe6f3', {'path at overflow rate': '#385F96', 'test path': '#B8481A'}),
}
# Neighbouring curves differ by little on purpose: they are ordered (Davos) and each carries its own
# marker and a direct label, so only non-adjacent pairs need to separate by colour alone.
SKIP = set()
worst = []
for name, (bg, cols) in SETS.items():
    print('\n' + name)
    for (la, a), (lb, b) in itertools.combinations(cols.items(), 2):
        if (la, lb) in SKIP or (lb, la) in SKIP: continue
        d = {k: dist(sim(a, k), sim(b, k)) for k in KINDS}
        m = min(d.values()); flag = '  <-- low' if m < 8 else ''
        print('  %-22s vs %-22s ' % (la, lb) + ' '.join('%s %5.1f' % (k[:4], v) for k, v in d.items()) + flag)
        worst.append((m, name, la, lb))
    for la, a in cols.items():
        c = contrast(rgb(a), rgb(bg)); print('  %-22s on background  contrast %.2f%s' % (la, c, '  <-- under 3:1' if c < 3 else ''))
worst.sort(); print('\nclosest pairs:'); [print('  %.1f  %s: %s vs %s' % w) for w in worst[:8]]
