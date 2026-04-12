import streamlit as st
st.set_page_config(page_title="Python Assessment — Intermediate", page_icon="📋", layout="centered")

try:
    from questions import INTERMEDIATE_TESTS
    from core import run_app
    run_app(
        level      = "intermediate",
        tests_map  = INTERMEDIATE_TESTS,
        time_limit = 45 * 60,
        title      = "Python Assessment — Intermediate (Banking / Data Analysis)",
    )
except Exception as e:
    import traceback
    st.error(f"Error: {e}")
    st.code(traceback.format_exc())
