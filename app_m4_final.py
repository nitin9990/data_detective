import streamlit as st

try:
    from questions_m4 import M4_FINAL_TEST, M4_FINAL_DATASET
    from core import run_app
    run_app(
        level        = "m4_final",
        tests_map    = M4_FINAL_TEST,
        time_limit   = 90 * 60,
        title        = "Module 4 Final Assessment — Clustering & Dimensionality Reduction",
        mode         = "final",
        max_attempts = 1,
        dataset      = M4_FINAL_DATASET,
    )
except Exception as e:
    import traceback
    st.error(f"Error: {e}")
    st.code(traceback.format_exc())
