"""Isoremoval curves for CE 323 settling-column data (Streamlit app entry point)."""
import streamlit as st

st.set_page_config(
    page_title="Isoremoval Curves · CE 323",
    page_icon=":material/water:",
    layout="wide",
    initial_sidebar_state="auto",
)

import pages_iso  # noqa: E402  (after set_page_config)

nav = st.navigation(
    [
        st.Page(pages_iso.tool_page, title="Isoremoval curves", icon=":material/show_chart:", default=True),
        st.Page(pages_iso.accessibility_page, title="Accessibility", icon=":material/accessibility_new:",
                url_path="accessibility"),
    ]
)
nav.run()
