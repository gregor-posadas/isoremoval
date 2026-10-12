"""Report-quality static figures (matplotlib), bundled for download.

Same content and encodings as the interactive charts: Davos colours with a marker shape per curve, dotted
extensions past the last reading, hollow markers for extrapolated design points, boxed legends below the
plotting area, white background, Open Sans. Each figure is saved as PNG (300 dpi) and SVG.
"""
import io
import zipfile

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import ui_theme as ui

matplotlib.use("Agg")
MPL_SYM = {"circle": "o", "square": "s", "diamond": "D", "triangle-up": "^", "triangle-down": "v", "cross": "P",
           "x": "X", "star": "*", "hexagon": "h", "pentagon": "p", "triangle-left": "<", "triangle-right": ">",
           "hourglass": "d", "bowtie": "H", "star-diamond": "8", "circle-x": "o", "square-x": "s",
           "diamond-wide": "D", "octagon": "8"}


def _style():
    return {
        "font.family": ["Open Sans", "DejaVu Sans"], "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
        "axes.edgecolor": "#857e6e", "axes.labelcolor": ui.INK, "text.color": ui.INK, "xtick.color": ui.INK,
        "ytick.color": ui.INK, "axes.grid": True, "grid.color": "#e6e1d6", "grid.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "legend.frameon": True, "legend.edgecolor": ui.INK, "legend.fancybox": False,
        "legend.framealpha": 1.0,
    }


def _legend_below(fig, handles, title, ncol, axis_space_in=0.62):
    """Boxed legend at the bottom of the figure, with the plots moved up so the axis titles clear it."""
    leg = fig.legend(handles=handles, title=title, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, 0.015),
                     fontsize=9, title_fontsize=9.5, borderpad=0.6, columnspacing=1.4, handlelength=2.6)
    leg.get_frame().set_linewidth(0.8)
    fig.canvas.draw()
    top = leg.get_window_extent().transformed(fig.transFigure.inverted()).y1
    fig.subplots_adjust(bottom=top + axis_space_in / fig.get_figheight())
    return leg


def curves_fig(cv, percents, unit, curve_points):
    cols = ui.curve_colors(len(percents))
    with plt.rc_context(_style()):
        fig, ax = plt.subplots(figsize=(6.5, 5.0))
        xmax = cv.t_last * 1.12
        far = []
        handles = []
        for p, col, sym in zip(percents, cols, ui.SYMBOLS):
            ts, zs, ext, t_b, extrap = curve_points(cv, p)
            if not ts:
                continue
            mk = MPL_SYM.get(sym, "o")
            ax.plot(ts, zs, color=col, lw=1.6, marker=mk, ms=4.5, mec="white", mew=0.5, zorder=3)
            if ext is not None:
                ax.plot(ext[0], ext[1], color=col, lw=1.4, ls=(0, (1, 1.8)), zorder=2)
                ax.plot([ext[0][-1]], [ext[1][-1]], marker=mk, ms=6, mfc="white", mec=col, mew=1.3, zorder=3)
            if t_b is not None:
                if t_b > 3 * cv.t_last:
                    far.append(p)
                else:
                    xmax = max(xmax, t_b * 1.05)
            handles.append(Line2D([0], [0], color=col, lw=1.6, marker=mk, ms=4.5, mec="white", mew=0.5, label=f"{p} %"))
        ax.axvline(cv.t_last, color=ui.INK2, lw=0.9, ls="--", zorder=1)
        ax.annotate(f"last reading ({cv.t_last:g} min)", (cv.t_last, 1), xycoords=("data", "axes fraction"),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=ui.INK2,
                    annotation_clip=False)
        ax.set_xlim(0, xmax)
        ax.set_ylim(cv.H * 1.02, 0)
        ax.set_xlabel("Time since the test started (min)")
        ax.set_ylabel(f"Depth below the water surface ({unit})")
        ax.xaxis.set_label_position("bottom")
        handles.append(Line2D([0], [0], color=ui.INK2, lw=1.4, ls=(0, (1, 1.8)), label="Extended past last reading"))
        n = len(handles)
        ncol = min(n, 5)
        rows = -(-n // ncol)
        fig.subplots_adjust(left=0.11, right=0.97, top=0.93)
        _legend_below(fig, handles, "Isoremoval curve (percent removal)", ncol)
    return fig, far


def design_fig(design, percents, target=None, read=None):
    cols = dict(zip(percents, ui.curve_colors(len(percents))))
    syms = dict(zip(percents, ui.SYMBOLS))
    with plt.rc_context(_style()):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 4.4), sharey=True)
        handles = []
        for ax, x in [(a1, "Overflow rate (m/d)"), (a2, "Detention time (h)")]:
            d = design.sort_values(x)
            ax.plot(d[x], d["Overall removal (%)"], color="#857e6e", lw=1.1, zorder=1)
            for _, r in design.iterrows():
                p = int(r["Curve (%)"]); ex = r["Extrapolated"] == "Yes"; mk = MPL_SYM.get(syms[p], "o")
                ax.plot([r[x]], [r["Overall removal (%)"]], marker=mk, ms=7, color=cols[p], mfc="white" if ex else cols[p],
                        mec=cols[p], mew=1.3, ls="none", zorder=3)
            ax.set_xlabel(x)
            ax.set_xlim(left=0)
        for _, r in design.iterrows():
            p = int(r["Curve (%)"]); ex = r["Extrapolated"] == "Yes"; mk = MPL_SYM.get(syms[p], "o")
            handles.append(Line2D([0], [0], marker=mk, ms=6, color=cols[p], mfc="white" if ex else cols[p], mec=cols[p],
                                  mew=1.2, ls="none", label=f"{p} %" + (" (extrap.)" if ex else "")))
        lo = max(0, design["Overall removal (%)"].min() - 8)
        a1.set_ylim(lo, min(100, design["Overall removal (%)"].max() + 6))
        if target is not None and read is not None:
            vo, th = read
            for ax, xv in [(a1, vo), (a2, th)]:
                ax.plot([0, xv], [target, target], color=ui.INK, lw=0.9, ls="--", zorder=2)
                ax.plot([xv, xv], [lo, target], color=ui.INK, lw=0.9, ls="--", zorder=2)
                ax.plot([xv], [target], marker="*", ms=11, color=ui.GOLD, mec=ui.INK, mew=0.8, zorder=4, ls="none")
            handles.append(Line2D([0], [0], marker="*", ms=10, color=ui.GOLD, mec=ui.INK, mew=0.8, ls="none",
                                  label=f"Target {target:g} %: {vo:.1f} m/d, {th:.2f} h"))
        a1.set_ylabel("Overall removal (%)")
        n = len(handles); ncol = min(n, 4); rows = -(-n // ncol)
        fig.subplots_adjust(left=0.1, right=0.98, top=0.96, wspace=0.08)
        _legend_below(fig, handles, "Isoremoval curve (hollow = extrapolated)", ncol)
    return fig


def profile_fig(z, r, unit, H, bands, p, t_b):
    with plt.rc_context(_style()):
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        fills = ["#c9d8ec", "#a9c1e0"]
        for i, (z0, z1) in enumerate(bands):
            m = (z >= z0 - 1e-9) & (z <= z1 + 1e-9)
            zz = np.concatenate([[z0], z[m], [z1]])
            rr = np.concatenate([[np.interp(z0, z, r)], r[m], [np.interp(z1, z, r)]])
            ax.fill_betweenx(zz, 0, rr, color=fills[i % 2], lw=0)
            ax.axhline(z1, color="white", lw=1)
        ax.plot(r, z, color=ui.BLUE, lw=1.8, marker="o" if len(z) <= 14 else None, ms=4.5, mec="white", mew=0.5)
        area = float(np.trapezoid(r, z))
        ax.set_ylim(H * 1.02, 0); ax.set_xlim(0, 105)
        ax.set_xlabel(f"Percent removal at {t_b:g} min (%)"); ax.set_ylabel(f"Depth ({unit})")
        ax.set_title(f"{p} % curve: area {area:.1f} %·{unit}, overall removal {area / H:.1f} %", fontsize=9.5)
        fig.tight_layout()
    return fig


def bundle(figs):
    """figs: list of (basename, matplotlib Figure). Returns zip bytes with PNG (300 dpi) and SVG of each."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, fig in figs:
            for ext, kw in (("png", {"dpi": 300}), ("svg", {})):
                b = io.BytesIO()
                fig.savefig(b, format=ext, **kw)
                z.writestr(f"{name}.{ext}", b.getvalue())
            plt.close(fig)
        z.writestr("README.txt", "Figures from the CE 323 Isoremoval curves tool.\n"
                                 "PNG files are 300 dpi for Word or Google Docs; SVG files are vector graphics that stay\n"
                                 "sharp at any size. Each figure is 6.5 in wide, the text width of a letter-size page.\n")
    return buf.getvalue()
