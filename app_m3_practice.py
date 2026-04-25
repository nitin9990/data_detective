import streamlit as st

try:
    from questions_m3 import M3_PRACTICE_TESTS, M3_DATASET
    from core import run_app
    run_app(
        level        = "m3_practice",
        tests_map    = M3_PRACTICE_TESTS,
        time_limit   = 99999,
        title        = "Module 3 Practice — EDA & Statistics",
        mode         = "practice",
        max_attempts = 3,
        dataset      = M3_DATASET,
    )
except Exception as e:
    import traceback
    st.error(f"Error: {e}")
    st.code(traceback.format_exc())
