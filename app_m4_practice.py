import streamlit as st

try:
    from questions_m4 import M4_PRACTICE_TESTS, M4_PRACTICE_DATASET
    from core import run_app
    run_app(
        level        = "m4_practice",
        tests_map    = M4_PRACTICE_TESTS,
        time_limit   = 99999,
        title        = "Module 4 Practice — Clustering & Dimensionality Reduction",
        mode         = "practice",
        max_attempts = 3,
        dataset      = M4_PRACTICE_DATASET,
    )
except Exception as e:
    import traceback
    st.error(f"Error: {e}")
    st.code(traceback.format_exc())
