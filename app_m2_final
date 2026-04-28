import streamlit as st

try:
    from questions_m2 import M2_FINAL_TEST, M2_DATASET
    from core import run_app
    run_app(
        level        = "m2_final",
        tests_map    = M2_FINAL_TEST,
        time_limit   = 45 * 60,
        title        = "Module 2 Final Assessment — Data Preprocessing",
        mode         = "final",
        max_attempts = 1,
        dataset      = M2_DATASET,
    )
except Exception as e:
    import traceback
    st.error(f"Error: {e}")
    st.code(traceback.format_exc())
