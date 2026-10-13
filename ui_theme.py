"""Look, behaviour and accessibility shared by the CE 323 sedimentation tools.

- Open Sans (served with the app), 18 px text.
- Light mode on a first visit; the Light / Dark / System switch in Settings still works and is remembered.
- Explainer terms, as on gregorposadas's website: a dotted underline marks a term with a short
  explanation. Hover shows it; a click, tap, Enter or Space pins it; Escape or a click elsewhere closes it.
- Gentle reveal: blocks below the fold fade up as they scroll into view (none for reduced motion).
- Plots are interactive (Plotly) on white cards, with boxed legends outside the plotting area and no
  labels drawn over data. Figure colours: Crameri's Davos for ordered series and Jack Baker's palette,
  checked under simulated colour blindness (tools/cvd_check.py). Marker shapes repeat every cue in colour.
"""
import html
import os
from itertools import count

import matplotlib
from matplotlib import font_manager
import numpy as np
import streamlit as st

matplotlib.use("Agg")
HERE = os.path.dirname(os.path.abspath(__file__))
for _f in ("open-sans-latin-400-normal.ttf", "open-sans-latin-700-normal.ttf", "open-sans-latin-400-italic.ttf"):
    _p = os.path.join(HERE, "static", "fonts", _f)
    if os.path.exists(_p):
        font_manager.fontManager.addfont(_p)

FONT = "'Open Sans', system-ui, -apple-system, 'Segoe UI', sans-serif"
INK, INK2, GRID, AXIS = "#1b1a17", "#4d4a43", "#e6e1d6", "#857e6e"
BLUE, ORANGE, LIGHTBLUE, GOLD, MAROON = "#385F96", "#B8481A", "#9EB8DB", "#E7B800", "#800000"
SYMBOLS = ["circle", "square", "diamond", "triangle-up", "triangle-down", "cross", "x", "star", "hexagon",
           "pentagon", "triangle-left", "triangle-right", "hourglass", "bowtie", "star-diamond", "circle-x",
           "square-x", "diamond-wide", "octagon"]
MPL_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "p", "<", ">", "o", "s", "D", "^", "v", "P", "X"]

PALETTE = {
    "light": {"paper": "#f6f2e9", "surface": "#fcfaf5", "sunk": "#ece6d8", "ink": INK, "ink2": INK2,
              "rule": "#d4ccbb", "rule_strong": INK, "focus": "#0072b2"},
    "dark": {"paper": "#000000", "surface": "#121212", "sunk": "#1c1c1c", "ink": "#ffffff", "ink2": "#c9c9c9",
             "rule": "#3d3d3d", "rule_strong": "#ffffff", "focus": "#56b4e9"},
}


def theme_type():
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def colors():
    return PALETTE[theme_type()]


def curve_colors(n):
    """n ordered colours from Davos on a white plot: pale grey-green = little removal, deep blue = a lot."""
    try:
        import cmcrameri.cm as cmc
        cmap = cmc.davos
    except Exception:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap("cividis")
    xs = np.linspace(0.56, 0.02, max(n, 2))[:n] if n > 1 else [0.3]
    return [matplotlib.colors.to_hex(cmap(x)) for x in xs]


# ---------------------------------------------------------------------------
# Explainer terms
# ---------------------------------------------------------------------------
_ids = count(1)


def term(text, definition, more=None, pic=None):
    """Inline HTML for a term with an explanation (use inside st.markdown(..., unsafe_allow_html=True)).

    more: a (label, url) pair or a list of them, shown as "Learn more" links. pic: a small inline SVG drawn above
    the text (decorative; the words carry the meaning)."""
    i = next(_ids)
    if more and isinstance(more[0], str):
        more = [more]
    links = ""
    if more:
        items = "".join(f'<span class="term__link"><a href="{u}" target="_blank" rel="noopener">{html.escape(l)}'
                        f'<span class="sr"> (opens in a new tab)</span></a></span>' for l, u in more)
        links = f'<span class="term__more">Learn more</span><span class="term__links">{items}</span>'
    figure = f'<span class="term__pic">{pic}</span>' if pic else ""
    cls = "term__tip has-pic" if pic else "term__tip"
    return (f'<span class="term"><button type="button" class="term__btn" aria-expanded="false" '
            f'aria-controls="tt-{i}">{text}</button><span class="{cls}" id="tt-{i}" role="note" hidden>'
            f'{figure}<span class="term__def">{definition}</span>{links}</span></span>')


def md(text):
    st.markdown(text, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page setup: CSS + one script (light default, explainer terms, reveal)
# ---------------------------------------------------------------------------
_JS = r"""
<script>
(function () {
  /* 1. Light mode on a first visit. Streamlit remembers the Light / Dark / System choice per page path; on
     Streamlit Community Cloud the app runs at an inner path (/~/+/), so the keys are built from the real path. */
  try {
    if (!localStorage.getItem("ce323-theme-init-v2")) {
      localStorage.setItem("ce323-theme-init-v2", "1");
      var base = location.pathname.replace(/accessibility\/?$/, "");
      if (base.slice(-1) !== "/") base += "/";
      [base, base + "accessibility"].forEach(function (p) {
        localStorage.setItem("stActiveTheme-" + p + "-v2", JSON.stringify("Light"));
      });
      location.reload();
      return;
    }
  } catch (e) {}

  if (window.__ce323) return;   /* the listeners below only need adding once */
  window.__ce323 = true;
  var doc = document;

  /* 2. Explainer terms (same behaviour as gregorposadas.github.io). */
  var open = null, timer = null;
  var hover = window.matchMedia("(hover: hover) and (pointer: fine)").matches;
  function tipOf(t) { return t.querySelector(".term__tip"); }
  function btnOf(t) { return t.querySelector(".term__btn"); }
  function place(t) {
    var tip = tipOf(t), br = btnOf(t).getBoundingClientRect();
    var vw = doc.documentElement.clientWidth, vh = window.innerHeight;
    tip.style.position = "fixed"; tip.style.top = "0px"; tip.style.left = "0px";
    var w = tip.offsetWidth, h = tip.offsetHeight;
    var left = Math.min(Math.max(12, br.left), vw - 12 - w);
    var top = br.bottom + 8;
    if (top + h > vh - 12 && br.top - 8 - h >= 12) top = br.top - 8 - h;
    tip.style.left = Math.max(12, left) + "px"; tip.style.top = top + "px";
  }
  function show(t, pin) {
    if (open && open !== t) hide(open);
    tipOf(t).hidden = false; btnOf(t).setAttribute("aria-expanded", "true");
    if (pin) t.classList.add("is-pinned");
    open = t; place(t);
  }
  function hide(t) {
    if (!t || !t.isConnected) { open = null; return; }
    tipOf(t).hidden = true; btnOf(t).setAttribute("aria-expanded", "false");
    t.classList.remove("is-pinned"); if (open === t) open = null;
  }
  doc.addEventListener("click", function (e) {
    var b = e.target.closest && e.target.closest(".term__btn");
    if (b) {
      var t = b.closest(".term");
      if (open === t && t.classList.contains("is-pinned")) hide(t); else show(t, true);
      return;
    }
    if (open && !open.contains(e.target)) hide(open);
  });
  if (hover) {
    doc.addEventListener("mouseover", function (e) {
      var t = e.target.closest && e.target.closest(".term");
      if (!t) return;
      clearTimeout(timer);
      timer = setTimeout(function () { if (open !== t) show(t, false); }, 120);
    });
    doc.addEventListener("mouseout", function (e) {
      var t = e.target.closest && e.target.closest(".term");
      if (!t || (e.relatedTarget && t.contains(e.relatedTarget))) return;
      clearTimeout(timer);
      if (!t.classList.contains("is-pinned"))
        timer = setTimeout(function () { if (open === t && !t.classList.contains("is-pinned")) hide(t); }, 300);
    });
  }
  doc.addEventListener("focusout", function (e) {
    if (open && e.relatedTarget && !open.contains(e.relatedTarget)) hide(open);
  });
  doc.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && open) { var b = btnOf(open); hide(open); b.focus(); }
  });
  window.addEventListener("resize", function () { if (open) place(open); });
  doc.addEventListener("scroll", function () { if (open) place(open); }, { passive: true, capture: true });

  /* 3. Gentle reveal: blocks below the fold fade up as they scroll into view.
     Fail-safe by design: visibility is checked directly on every scroll, resize and page change and on a
     timer as a backstop, every block is fully shown again 0.9 s after its fade starts whatever happens, and
     nothing is hidden at all for reduced motion, a hidden tab, or printing. */
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  if (doc.visibilityState === "hidden") return;
  doc.documentElement.classList.add("rv-on");
  var pending = [];
  function done(el) { el.classList.remove("rv", "rv-in"); el.style.removeProperty("--rv-d"); }
  function reveal(el, delay) {
    if (!el.classList.contains("rv") || el.classList.contains("rv-in")) return;
    if (delay) el.style.setProperty("--rv-d", delay + "ms");
    el.classList.add("rv-in");
    setTimeout(function () { done(el); }, 900 + (delay || 0));
  }
  function check() {
    var vh = window.innerHeight || doc.documentElement.clientHeight, n = 0;
    pending = pending.filter(function (el) {
      if (!el.isConnected || !el.classList.contains("rv")) return false;
      var r = el.getBoundingClientRect();
      if (r.top < vh * 0.94 && r.bottom > 0) { reveal(el, Math.min(n++, 4) * 70); return false; }
      return true;
    });
  }
  function scan() {
    var root = doc.querySelector('[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"]');
    if (!root) return;
    var vh = window.innerHeight || doc.documentElement.clientHeight;
    Array.prototype.forEach.call(root.children, function (el) {
      if (el.dataset.rvSeen) return;
      el.dataset.rvSeen = "1";
      if (el.getBoundingClientRect().top < vh * 0.92) return;   /* on screen already: no effect */
      el.classList.add("rv");
      pending.push(el);
    });
    check();
  }
  var ticking = false;
  function soon() { if (ticking) return; ticking = true; setTimeout(function () { ticking = false; check(); }, 16); }
  doc.addEventListener("scroll", soon, { passive: true, capture: true });
  window.addEventListener("resize", soon);
  var mt = null;
  new MutationObserver(function () { clearTimeout(mt); mt = setTimeout(scan, 60); }).observe(doc.body, { childList: true, subtree: true });
  setInterval(check, 400);   /* backstop */
  scan();
  doc.addEventListener("focusin", function (e) {
    var el = e.target.closest && e.target.closest(".rv");
    if (el) reveal(el, 0);
  });
  doc.addEventListener("visibilitychange", function () {
    if (doc.visibilityState === "hidden") Array.prototype.forEach.call(doc.querySelectorAll(".rv"), done);
  });
  window.addEventListener("beforeprint", function () {
    Array.prototype.forEach.call(doc.querySelectorAll(".rv"), done);
  });
})();
</script>
"""


def setup_page():
    c = colors()
    st.markdown(
        f"""
<style>
:root {{ --focus: {c['focus']}; --ink2: {c['ink2']}; --rule: {c['rule']}; --sunk: {c['sunk']};
        --surface: {c['surface']}; --ink: {c['ink']}; --rule-strong: {c['rule_strong']}; }}
.block-container {{ max-width: 1080px; padding-top: 2.5rem; }}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{ max-width: 72ch; line-height: 1.6; }}
[data-testid="stCaptionContainer"] p {{ max-width: 72ch; line-height: 1.5; }}
h1, h2, h3 {{ letter-spacing: -0.005em; }}
/* focus that keyboard users can see */
*:focus-visible, button:focus-visible, a:focus-visible, input:focus-visible, [role="radio"]:focus-visible,
[role="checkbox"]:focus-visible, [role="tab"]:focus-visible {{
  outline: 3px solid var(--focus) !important; outline-offset: 2px !important; box-shadow: none !important; }}
/* comfortable targets */
.stButton button, .stDownloadButton button, .stLinkButton a, [data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-pills"], [data-testid="stBaseButton-pillsActive"],
[data-testid="stBaseButton-segmented_control"], [data-testid="stBaseButton-segmented_controlActive"] {{
  min-height: 48px; padding-left: 1rem; padding-right: 1rem; }}
[data-testid="stNumberInput"] input, [data-testid="stTextInput"] input {{ min-height: 46px; }}
[data-baseweb="select"] > div {{ min-height: 48px; }}
[data-testid="stExpander"] summary {{ min-height: 48px; }}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] a {{ display: inline-block; padding: 6px 0; }}
[data-testid="stMarkdownContainer"] a {{ text-underline-offset: 3px; text-decoration-thickness: 2px; }}
/* step labels */
.step-kicker {{ font-size: 0.9rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--ink2); margin-bottom: -0.6rem; }}
.lede {{ font-size: 1.12rem; }}
/* "the idea" callouts */
.idea {{ border-left: 6px solid var(--rule-strong); background: var(--sunk); padding: 14px 18px; margin: 4px 0 14px;
  max-width: 72ch; }}
.idea p {{ margin: 0 0 8px; }} .idea p:last-child {{ margin-bottom: 0; }}
.idea__k {{ font-size: 0.82rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink2);
  display: block; margin-bottom: 4px; }}
/* plots sit on white cards in both themes */
[data-testid="stPlotlyChart"] {{ background: #ffffff; border: 1px solid var(--rule); border-radius: 8px; overflow: hidden; }}
.figure-desc {{ max-width: 72ch; }}
/* shared legend for side-by-side charts: boxed, centred, wraps on narrow screens */
.lg {{ background: #fff; color: {INK}; border: 1px solid {INK}; width: fit-content; max-width: 100%; margin: 6px auto 18px;
  padding: 8px 14px 10px; text-align: center; font-size: 0.82rem; }}
.lg__t {{ margin-bottom: 4px; }}
.lg__r {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 4px 16px; }}
.lg__i {{ display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }}
.sr {{ position: absolute !important; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }}
.skip-link {{ position: absolute; left: -9999px; }}
.skip-link:focus {{ position: static; display: inline-block; padding: 8px 12px; outline: 3px solid var(--focus); }}
/* explainer terms */
.term {{ position: relative; }}
.term__btn {{ font: inherit; color: inherit; background: none; border: 0; padding: 0; margin: 0; cursor: help;
  text-decoration: underline dotted; text-decoration-thickness: 2px; text-underline-offset: 3px;
  -webkit-tap-highlight-color: transparent; }}
.term__btn:hover, .term__btn[aria-expanded="true"] {{ background: var(--sunk); text-decoration-style: solid; }}
.term__tip {{ position: absolute; left: 0; top: calc(100% + 8px); z-index: 1000; display: block; width: max-content;
  max-width: min(26rem, calc(100vw - 32px)); padding: 12px 14px; background: var(--surface); color: var(--ink);
  border: 2px solid var(--rule-strong); box-shadow: 4px 4px 0 0 var(--rule-strong); font-size: 0.92rem;
  font-weight: 400; line-height: 1.5; white-space: normal; text-align: left; }}
.term__tip[hidden] {{ display: none; }}
.term__def {{ display: block; }}
.term__tip.has-pic {{ width: min(23rem, calc(100vw - 32px)); }}
.term__pic {{ display: block; background: #fff; border: 1px solid var(--rule); border-radius: 6px; padding: 6px 8px;
  margin: 0 0 10px; }}
.term__pic .pic {{ display: block; width: 100%; height: auto; max-height: 130px; }}
.term__pic text {{ font-family: {FONT}; }}
.term__more {{ display: block; margin-top: 8px; font-size: 0.8rem; font-weight: 700; letter-spacing: .04em;
  text-transform: uppercase; color: var(--ink2); }}
.term__links {{ display: block; margin-top: 2px; }}
.term__link {{ display: block; margin: 3px 0; padding-left: 1em; text-indent: -1em; }}
.term__link::before {{ content: "→ "; color: var(--ink2); }}
.term__links a {{ color: var(--ink); font-weight: 600; }}
/* reveal */
.rv-on .rv {{ opacity: 0; transform: translateY(14px); }}
.rv-on .rv.rv-in {{ opacity: 1; transform: none; transition: opacity 600ms cubic-bezier(.2,.6,.2,1),
  transform 600ms cubic-bezier(.2,.6,.2,1); transition-delay: var(--rv-d, 0ms); }}
@media print {{ .rv-on .rv {{ opacity: 1 !important; transform: none !important; }} .term__tip {{ display: none !important; }} }}
@media (prefers-reduced-motion: reduce) {{
  .rv-on .rv {{ opacity: 1; transform: none; transition: none; }}
  *, *::before, *::after {{ animation-duration: 0.001ms !important; transition-duration: 0.001ms !important;
    scroll-behavior: auto !important; }} }}
</style>
""",
        unsafe_allow_html=True,
    )
    st.html(_JS, unsafe_allow_javascript=True)


def step_header(number, title, anchor):
    st.markdown(f'<p class="step-kicker">Step {number}</p>', unsafe_allow_html=True)
    st.header(title, anchor=anchor)


def idea(label, body_html):
    """A highlighted 'the idea' box (HTML body, may contain terms)."""
    md(f'<div class="idea"><span class="idea__k">{label}</span>{body_html}</div>')


def figure_description(text):
    md(f'<p class="figure-desc">{text}</p>')


# ---------------------------------------------------------------------------
# Plotly styling
# ---------------------------------------------------------------------------
PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "scale": 3},
}


def plot_config(filename):
    cfg = dict(PLOT_CONFIG)
    cfg["toImageButtonOptions"] = dict(PLOT_CONFIG["toImageButtonOptions"], filename=filename)
    return cfg


def style_axes(fig, **kw):
    ax = dict(automargin=False, title_standoff=14, showline=True, linecolor=AXIS, linewidth=1.2, ticks="outside", tickcolor=AXIS, ticklen=6,
              gridcolor=GRID, zeroline=False, title_font=dict(size=16, color=INK), tickfont=dict(size=14, color=INK),
              mirror=False)
    ax.update(kw)
    return ax


LEFT_MARGIN = 104  # room for y tick labels, a 14 px standoff, the y title and ~24 px of padding to the card edge
XAXIS_BAND = 88   # px below the plotting area for tick labels, the axis title and a gap before the legend


TITLE_BAND = 44


def base_layout(fig, height, legend_title=None, legend_rows=1, top=24, n_items=None, per_row=4, title=None):
    """White plot, Open Sans, boxed legend centred just below the x-axis title (never on top of data).

    `height` is the plotting area's height. The legend is pinned to the bottom of the plotting area at a fixed
    distance, so the gap to the axis title stays the same however many rows the legend wraps to; if it wraps
    more on a narrow screen, Plotly grows the bottom margin to fit it.
    """
    # Size the bottom margin for a one-row legend (desktop); on a narrow screen the legend wraps and Plotly's
    # autoexpand grows the margin to fit, keeping the same gap below the axis title.
    legend_rows = 1
    if title:
        top += TITLE_BAND
    legend_h = (32 if legend_title else 10) + 26 * legend_rows + 8
    bottom = XAXIS_BAND + legend_h + 18
    fig.update_layout(
        template="none", height=height + top + bottom, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(family=FONT, size=15, color=INK),
        margin=dict(l=LEFT_MARGIN, r=36, t=top, b=bottom, autoexpand=True),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=INK, font=dict(family=FONT, size=15, color=INK)),
        legend=dict(orientation="h", yref="paper", y=-XAXIS_BAND / height, yanchor="top", xanchor="center",
                    x=0.5, bgcolor="#ffffff", bordercolor=INK, borderwidth=1, font=dict(size=14),
                    title=dict(text=legend_title, side="top center") if legend_title else None,
                    itemclick="toggle", itemdoubleclick="toggleothers", tracegroupgap=6),
        clickmode="event+select", dragmode="zoom", hovermode="closest",
    )
    if title:
        fig.update_layout(title=dict(text=f"<b>{title}</b>", x=0.5, xanchor="center", y=1, yref="container",
                                     yanchor="top", pad=dict(t=14), font=dict(size=17, color=INK, family=FONT)))
    return fig
