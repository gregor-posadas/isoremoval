"""Look and accessibility settings shared by the CE 323 sedimentation tools.

Follows the same system as Gregor Posadas's other teaching sites:
- Atkinson Hyperlegible Next (Braille Institute), 18 px reading text, two weights.
- Warm paper in light mode, black in dark mode; the theme follows the device and
  can be switched in the app menu (top right, Settings).
- Figure colours from Jack Baker's "Choosing colors for figures" palette, and
  Crameri's Davos colour map for ordered series (isoremoval curves). Every
  series also has a marker shape and a direct label, so colour is never the
  only cue; checked under simulated colour blindness with tools/cvd_check.py.
- Visible 3 px focus outlines, 48 px targets, no motion when the device asks
  for reduced motion.
"""
import os
from contextlib import contextmanager

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import streamlit as st

matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
for _f in ("AtkinsonHyperlegibleNext-Regular.ttf", "AtkinsonHyperlegibleNext-Bold.ttf"):
    _p = os.path.join(HERE, "static", "fonts", _f)
    if os.path.exists(_p):
        font_manager.fontManager.addfont(_p)

# Baker palette (light) and its lighter partners for dark backgrounds
PALETTE = {
    "light": {"blue": "#385F96", "orange": "#B8481A", "lightblue": "#9EB8DB", "gold": "#E7B800", "maroon": "#800000",
              "paper": "#f6f2e9", "surface": "#fcfaf5", "ink": "#1b1a17", "ink2": "#4d4a43", "rule": "#d4ccbb",
              "fill": "#9EB8DB", "focus": "#0072b2"},
    "dark": {"blue": "#9EB8DB", "orange": "#ff8a4c", "lightblue": "#385F96", "gold": "#ffe27a", "maroon": "#e8a0a0",
             "paper": "#000000", "surface": "#121212", "ink": "#ffffff", "ink2": "#c9c9c9", "rule": "#3d3d3d",
             "fill": "#2A4F86", "focus": "#56b4e9"},
}
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p"]


def theme_type():
    try:
        t = st.context.theme.type
        return "dark" if t == "dark" else "light"
    except Exception:
        return "light"


def colors():
    return PALETTE[theme_type()]


def curve_colors(n):
    """n colours from Davos, ordered: darker = more removal in light mode, brighter in dark mode."""
    try:
        import cmcrameri.cm as cmc
        cmap = cmc.davos
    except Exception:          # fallback: matplotlib's perceptually even 'cividis'
        cmap = plt.get_cmap("cividis")
    import numpy as np
    if theme_type() == "dark":
        xs = np.linspace(0.34, 0.96, max(n, 2))
    else:
        xs = np.linspace(0.55, 0.0, max(n, 2))
    return [matplotlib.colors.to_hex(cmap(x)) for x in xs[:n]]


@contextmanager
def figure_style():
    c = colors()
    rc = {
        "font.family": ["Atkinson Hyperlegible Next", "DejaVu Sans"],
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "axes.labelweight": "bold",
        "axes.edgecolor": c["ink"],
        "axes.labelcolor": c["ink"],
        "axes.facecolor": c["paper"],
        "figure.facecolor": c["paper"],
        "savefig.facecolor": c["paper"],
        "text.color": c["ink"],
        "xtick.color": c["ink"],
        "ytick.color": c["ink"],
        "grid.color": c["rule"],
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.4,
        "lines.markersize": 7,
    }
    with plt.rc_context(rc):
        yield c


def inject_css():
    c = colors()
    st.markdown(
        f"""
<style>
:root {{ --focus: {c['focus']}; --ink2: {c['ink2']}; --rule: {c['rule']}; }}
/* Reading width and rhythm */
.block-container {{ max-width: 1080px; padding-top: 2.5rem; }}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
[data-testid="stCaptionContainer"] {{ max-width: 70ch; line-height: 1.55; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
/* Visible focus for keyboard users */
*:focus-visible, button:focus-visible, a:focus-visible, input:focus-visible,
[role="radio"]:focus-visible, [role="checkbox"]:focus-visible, [role="tab"]:focus-visible,
[data-baseweb="select"] *:focus-visible {{
  outline: 3px solid var(--focus) !important; outline-offset: 2px !important; box-shadow: none !important; }}
/* Comfortable targets */
.stButton button, .stDownloadButton button, [data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-pills"],
[data-testid="stBaseButton-pillsActive"] {{ min-height: 48px; padding-left: 1rem; padding-right: 1rem; }}
[data-testid="stNumberInput"] input, [data-testid="stTextInput"] input {{ min-height: 46px; }}
[data-baseweb="select"] > div {{ min-height: 48px; }}
[data-testid="stExpander"] summary {{ min-height: 48px; }}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] a {{ display: inline-block; padding: 6px 0; }}
/* Links look like links */
[data-testid="stMarkdownContainer"] a {{ text-decoration: underline; text-underline-offset: 3px;
  text-decoration-thickness: 2px; }}
/* Step labels */
.step-kicker {{ font-size: 0.9rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--ink2); margin-bottom: -0.6rem; }}
.figure-desc {{ max-width: 70ch; color: inherit; }}
.skip-link {{ position: absolute; left: -9999px; }}
.skip-link:focus {{ position: static; display: inline-block; padding: 8px 12px; outline: 3px solid var(--focus); }}
/* Respect reduced motion */
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{ animation-duration: 0.001ms !important; animation-iteration-count: 1 !important;
    transition-duration: 0.001ms !important; scroll-behavior: auto !important; }}
}}
</style>
""",
        unsafe_allow_html=True,
    )


def step_header(number, title, anchor):
    st.markdown(f'<p class="step-kicker">Step {number}</p>', unsafe_allow_html=True)
    st.header(title, anchor=anchor)


def figure_description(text):
    """A sentence that says what a figure shows, read before the figure itself."""
    st.markdown(f'<p class="figure-desc">{text}</p>', unsafe_allow_html=True)
