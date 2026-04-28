import streamlit as st

try:
    from questions_m2 import M2_PRACTICE_TESTS, M2_DATASET
    from core import run_app
    run_app(
        level        = "m2_practice",
        tests_map    = M2_PRACTICE_TESTS,
        time_limit   = 99999,
        title        = "Module 2 Practice — Data Preprocessing",
        mode         = "practice",
        max_attempts = 3,
        dataset      = M2_DATASET,
    )
except Exception as e:
    import traceback
    st.error(f"Error: {e}")
    st.code(traceback.format_exc())
