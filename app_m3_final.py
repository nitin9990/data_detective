import streamlit as st

try:
    from questions_m3 import M3_FINAL_TEST, M3_DATASET
    from core import run_app
    run_app(
        level        = "m3_final",
        tests_map    = M3_FINAL_TEST,
        time_limit   = 45 * 60,
        title        = "Module 3 Final Assessment — EDA & Statistics",
        mode         = "final",
        max_attempts = 1,
        dataset      = M3_DATASET,
    )
except Exception as e:
    import traceback
    st.error(f"Error: {e}")
    st.code(traceback.format_exc())
