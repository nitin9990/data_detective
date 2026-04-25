import streamlit as st
st.set_page_config(page_title="Module 1 Practice — Python Core", page_icon="🐍", layout="centered")
try:
    from questions_m1 import M1_PRACTICE_TESTS
    from core import run_app
    run_app(
        level        = "m1_practice",
        tests_map    = M1_PRACTICE_TESTS,
        time_limit   = 99999,   # no timer for practice
        title        = "Module 1 Practice — Python Core",
        mode         = "practice",
        max_attempts = 3,
    )
except Exception as e:
    import traceback
    st.error(f"Error: {e}")
    st.code(traceback.format_exc())