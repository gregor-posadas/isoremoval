# Isoremoval curves

A learning tool for CE 323 (Environmental and Fluids Laboratory, Boise State University). It turns
settling-column readings into isoremoval (isoconcentration) curves and the two design plots for a
sedimentation basin: overall removal against overflow rate, and overall removal against detention time.
Those values then go into the [Sedimentation Basin Design tool](https://sedimentationbasindesign.streamlit.app).

**Live app:** https://isoremoval.streamlit.app · For learning; not certified for engineering design.

## Using it

The tool teaches the method as it goes, for students meeting isoremoval curves for the first time. A short
"the idea" section comes first; dotted-underlined terms open a plain-language explanation.

1. **Load your readings.** Upload an Excel or CSV file, or use the lecture's sample data. First column: port
   depths. Column headings: sampling times in minutes. Cells: turbidity (NTU) or suspended solids (mg/L).
2. **Describe your test.** Enter C₀, depth units and the column depth H. The check flags readings above C₀ and
   readings that go up over time, and offers a clean-up.
3. **Percent removal.** The removal table, shaded, with what to notice.
4. **Isoremoval curves.** An interactive chart. Click a point (or pick a curve) to follow one curve and see why
   it bends. Solid lines come from your readings; dotted lines continue past the last reading.
5. **Curves to design values.** Detention time, overflow rate and overall removal for each curve, and the two
   design plots side by side on a shared axis. Click a point to open its worked overall-removal calculation.
6. **Read your design values.** Enter a target removal; the button opens the basin tool with the values filled in.

## Method

`iso_calc.py` holds all the math, with the method written out at the top of the file. In short:

- Removal R = (C₀ − C) ÷ C₀ × 100, starting from 0 % at t = 0 and changing linearly between readings.
- The removal profile at any time runs from 100 % at the surface through the port values, in straight lines.
- A p % curve is where that profile equals p. The time it reaches the bottom is solved exactly; past the last
  reading the curve is extended along its own last segment (the lecture's method), and flagged as extrapolated.
- Overall removal = area under the removal profile from 0 to H, divided by H (trapezoids).
  Detention time = that time; overflow rate = H ÷ time.

### Changes in October 2026

The first version (kept in `tests/legacy_calc.py`) gave the same answers wherever a curve met the bottom
exactly at a reading. Elsewhere it had three problems, now fixed:

| Problem | Effect on the sample data |
|---|---|
| Back-extrapolated removal before the first reading | 14–20 % removal reported at 2–5 min |
| Bottom times read off a 5-minute grid | 40 % curve: 46.0 min instead of 44.4 min (overflow rate 78.3 vs 81.1 m/d) |
| Ports dropped once extrapolated removal passed 100 % | late results based on 2 of 5 ports |
| Curves jumped down at t = 0 above the first port | 50 % curve at 0.25 m after seconds; now leaves the surface steadily |

Extrapolated results (curves that reach the bottom after the last reading) also change, because curves are now
extended along their own last segment. For the sample data the 70 % curve reaches the bottom at 175 min
(82.2 % overall removal) rather than 133 min (84.7 %). The app shows how far past the data each result is.

## Accessibility

Open Sans at 18 px, light mode by default with a dark mode in Settings, 48 px targets, visible focus outlines,
scroll reveals that switch off for reduced motion, interactive Plotly charts on white with boxed legends outside
the plotting area and nothing drawn over the data, colour never used alone (marker shapes, line styles), and a
sentence and a table behind every chart. Plot colours: Crameri's Davos colour map and Jack Baker's figure
palette, checked with `tools/cvd_check.py`. See the Accessibility page in the app for details and known gaps.

## Running locally

```
pip install -r requirements.txt
streamlit run isoremoval.py
python tests/test_calc.py
```

Font: Open Sans, SIL Open Font License (`static/fonts/OFL.txt`).
