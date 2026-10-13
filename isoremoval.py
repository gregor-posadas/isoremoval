"""Isoremoval curves for CE 323 settling-column data (Streamlit app entry point)."""
import streamlit as st

st.set_page_config(
    page_title="Isoremoval Curves · CE 323",
    page_icon=":material/water:",
    layout="wide",
    initial_sidebar_state="auto",
)


def _refresh_local_modules(names, stamp_only=False):
    """Streamlit Community Cloud can keep running an old copy of a module after a new commit is pulled.
    Reload any of this app's modules whose file has changed since it was loaded, and everything listed after
    it (they may hold references to the old one)."""
    import importlib
    import os
    import sys
    changed = False
    for name in names:
        m = sys.modules.get(name)
        if m is None:
            continue
        try:
            mt = os.path.getmtime(m.__file__)
        except (OSError, TypeError):
            continue
        if stamp_only:
            if getattr(m, "_loaded_mtime", None) is None:
                m._loaded_mtime = mt
            continue
        if changed or getattr(m, "_loaded_mtime", None) != mt:
            importlib.reload(m)
            m._loaded_mtime = mt
            changed = True


_refresh_local_modules(["explainers", "ui_theme", "iso_calc", "column_anim", "report_figs", "hand_method",
                        "byhand_anim", "pages_iso", "pages_byhand"])
import pages_iso  # noqa: E402  (after set_page_config)
import pages_byhand  # noqa: E402
_refresh_local_modules(["explainers", "ui_theme", "iso_calc", "column_anim", "report_figs", "hand_method",
                        "byhand_anim", "pages_iso", "pages_byhand"], stamp_only=True)

tool = st.Page(pages_iso.tool_page, title="Isoremoval curves", icon=":material/show_chart:", default=True)
byhand = st.Page(pages_byhand.byhand_page, title="The method by hand", icon=":material/edit_note:", url_path="by-hand")
pages_iso.BYHAND_PAGE = byhand
pages_byhand.TOOL_PAGE = tool
nav = st.navigation(
    [
        tool,
        byhand,
        st.Page(pages_iso.accessibility_page, title="Accessibility", icon=":material/accessibility_new:",
                url_path="accessibility"),
    ]
)
nav.run()
