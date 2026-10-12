"""Checks for iso_calc.py. Run with:  python -m pytest tests  (or python tests/test_calc.py)"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import iso_calc as ic          # noqa: E402
import legacy_calc as lc       # noqa: E402

D, T, C = ic.parse_frame(ic.sample_frame())
RD = ic.removal_table(C, ic.SAMPLE_INITIAL, D, T)
PL = [20, 30, 40, 50, 60, 70]


def curves(**kw):
    return ic.Curves(RD, kw.pop("H", D.max()), kw.pop("pl", PL), **kw)


def test_removal_formula():
    # lecture slide: (20 - 14) / 20 = 30 %
    assert RD.loc[0.5, 10] == 30.0


def test_starts_at_zero_removal():
    cv = curves()
    assert np.allclose(cv.port_removals(0), 0)
    # halfway to the first reading, removal is half the first reading
    assert np.allclose(cv.port_removals(5), RD[10].values / 2)


def test_bottom_time_is_exact():
    cv = curves()
    bottom = RD.loc[2.5].values
    for p in [20, 30, 40, 50]:
        t_b, extrap = cv.time_to_bottom(p)
        assert not extrap
        assert abs(t_b - np.interp(p, bottom, T)) < 1e-9


def test_within_data_matches_legacy_where_legacy_was_exact():
    # 20 % and 50 % land exactly on readings, so both methods agree there
    new = curves().design_table("m").set_index("Curve (%)")
    old = lc.Curves(lc.removal_table(C, 20, D, T), 2.5, PL).design_table("m").set_index("Curve (%)")
    for p in [20, 50]:
        assert abs(new.loc[p, "Overall removal (%)"] - old.loc[p, "Overall removal (%)"]) < 0.01
        assert abs(new.loc[p, "Reaches bottom at (min)"] - old.loc[p, "Reaches bottom at (min)"]) < 0.01


def test_profile_methods_agree_at_last_reading():
    cv = curves()
    a = cv.overall_removal(cv.t_last)
    b = cv.overall_removal(cv.t_last + 1e-6)
    assert abs(a - b) < 0.5


def test_extrapolated_curve_follows_its_last_segment():
    cv = curves()
    z70, z85 = cv.depth_at(70, 70), cv.depth_at(70, 85)
    t_b, extrap = cv.time_to_bottom(70)
    assert extrap
    assert abs(t_b - (85 + (2.5 - z85) / ((z85 - z70) / 15))) < 1e-6


def test_removals_clipped_not_dropped():
    df = RD.copy()
    df.iloc[0, -1] = 104.0     # an impossible reading
    cv = ic.Curves(df, 2.5, PL)
    assert cv.profile(85)[1].max() <= 100
    assert len(cv.profile(85)[0]) == 1 + len(D)


def test_monotone_cleanup():
    noisy = C.copy()
    noisy[2, 3] = noisy[2, 2] + 1.0     # a reading that goes up
    fixed = ic.clean_monotone(noisy)
    assert np.all(np.diff(fixed, axis=1) <= 1e-12)
    untouched = [j for j in range(noisy.shape[1]) if j not in (2, 3)]
    assert np.allclose(fixed[2, untouched], noisy[2, untouched])
    assert np.allclose(ic.clean_monotone(C), C)   # clean data is left alone


def test_units_only_change_overflow_rate():
    cv = curves()
    m = cv.design_table("m")
    ft = cv.design_table("ft")
    assert np.allclose(ft["Overflow rate (m/d)"], (m["Overflow rate (m/d)"] * 0.3048).round(2), atol=0.02)
    assert np.allclose(ft["Overall removal (%)"], m["Overall removal (%)"])


def test_deeper_column_than_ports():
    cv = curves(H=3.0)
    z, r = cv.profile(50)
    assert z[-1] == 3.0 and 0 <= r[-1] <= r[-2]


def test_read_design():
    dt = curves().design_table("m")
    vo, th, extrap = ic.read_design(dt, 60.0)
    assert dt["Overflow rate (m/d)"].min() < vo < dt["Overflow rate (m/d)"].max()
    assert ic.read_design(dt, 99.0) is None


def test_bad_headers_are_explained():
    df = ic.sample_frame().rename(columns={10: "ten"})
    try:
        ic.parse_frame(df)
    except ic.DataProblem as e:
        assert "ten" in str(e)
    else:
        raise AssertionError("expected a DataProblem")


def test_headers_with_units_are_read():
    df = ic.sample_frame().rename(columns={10: "10 min", 20: "20 min"})
    _, t, _ = ic.parse_frame(df)
    assert list(t[:2]) == [10.0, 20.0]


if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            n += 1
    print(f"{n} checks passed")
