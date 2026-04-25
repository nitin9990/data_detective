import streamlit as st

try:
    from questions_m1 import M1_FINAL_TEST
    from core import run_app
    run_app(
        level        = "m1_final",
        tests_map    = M1_FINAL_TEST,
        time_limit   = 45 * 60,
        title        = "Module 1 Final Assessment — Python Core",
        mode         = "final",
        max_attempts = 1,
    )
except Exception as e:
    import traceback
    st.error(f"Error: {e}")
    st.code(traceback.format_exc())
