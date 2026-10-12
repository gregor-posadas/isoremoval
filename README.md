# Isoremoval curves

A learning tool for CE 323 (Environmental and Fluids Laboratory, Boise State University). It turns
settling-column readings into isoremoval (isoconcentration) curves and the two design plots for a
sedimentation basin: overall removal against overflow rate, and overall removal against detention time.
Those values then go into the [Sedimentation Basin Design tool](https://sedimentationbasindesign.streamlit.app).

**Live app:** https://isoremoval.streamlit.app · For learning; not certified for engineering design.

## Using it

1. **Load your data.** Upload an Excel or CSV file, or use the sample data from the lecture. First column:
   port depths. Column headings: sampling times in minutes. Cells: turbidity (NTU) or suspended solids (mg/L).
2. **Describe your test.** Enter the starting value C₀, depth units and the column depth H. The data check
   flags readings above C₀ and readings that go up over time.
3. **Check the curves.** Pick the percentages to draw. Solid lines come from your readings; dotted lines
   extend a curve past the last reading.
4. **Read your design values.** Enter a target overall removal to read the overflow rate and detention time,
   then download the tables and plots.

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

Extrapolated results (curves that reach the bottom after the last reading) also change, because curves are now
extended along their own last segment. For the sample data the 70 % curve reaches the bottom at 175 min
(82.5 % overall removal) rather than 133 min (84.7 %). The app shows how far past the data each result is.

## Accessibility

Atkinson Hyperlegible Next, 18 px text, light and dark themes that follow the device, 48 px targets, visible
focus outlines, reduced motion, colour never used alone (marker shapes, direct labels, dotted extensions),
a sentence and a table behind every figure. Plot colours: Crameri's Davos colour map and Jack Baker's figure
palette, checked with `tools/cvd_check.py`. See the Accessibility page in the app for details and known gaps.

## Running locally

```
pip install -r requirements.txt
streamlit run isoremoval.py
python tests/test_calc.py
```

Fonts: Atkinson Hyperlegible Next and Mono, © Braille Institute, SIL Open Font License (`static/fonts/OFL.txt`).
