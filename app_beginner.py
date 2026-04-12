import streamlit as st
st.set_page_config(page_title="Python Assessment — Beginner", page_icon="📋", layout="centered")

try:
    from questions import BEGINNER_TESTS
    from core import run_app
    run_app(
        level      = "beginner",
        tests_map  = BEGINNER_TESTS,
        time_limit = 30 * 60,
        title      = "Python Assessment — Beginner",
    )
except Exception as e:
    import traceback
    st.error(f"Startup error: {e}")
    st.code(traceback.format_exc())
