import streamlit as st
import time, hashlib, re
from datetime import datetime, timezone
from supabase import create_client

def _norm(s): return re.sub(r'\s+', '', str(s).strip().lower())
def h(s):     return hashlib.sha256(_norm(s).encode()).hexdigest()
def hc(a, b): return h(a) == h(b)

def _sb(): return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

def run_code(preload, code, timeout=30):
    import io, sys, threading
    result = [None, None]
    def _run():
        buf = io.StringIO(); old = sys.stdout; sys.stdout = buf
        try:    exec(preload + "\n" + code, {})
        except Exception as e: sys.stdout = old; result[1] = str(e); return
        sys.stdout = old; result[0] = buf.getvalue().strip()
    t = threading.Thread(target=_run, daemon=True)
    t.start(); t.join(timeout)
    if t.is_alive(): return "ERROR: Execution timed out (30s limit)"
    return f"ERROR: {result[1]}" if result[1] else (result[0] or "")

def check_attempts(email, level, max_attempts=2):
    res = _sb().table("attempts").select("id").eq("email", email.lower().strip()).eq("level", level).execute()
    count = len(res.data)
    return count + 1, count >= max_attempts

def start_attempt(email, level, employee_id):
    res = _sb().table("attempts").insert({
        "email": email.lower().strip(), "level": level,
        "test_id": 1, "attempt_num": 1, "max_score": 10,
        "employee_id": employee_id.strip(),
    }).execute()
    return res.data[0]["id"]

def finish_attempt(attempt_id, steps_completed, total_steps):
    _sb().table("attempts").update({
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "score": steps_completed, "max_score": total_steps,
        "pct": round(min(steps_completed/total_steps*100, 100), 1),
        "passed": steps_completed == total_steps,
        "valid": True,
    }).eq("id", attempt_id).execute()

def _init(defaults):
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

def main():
    from questions_m5 import M5_1_LAB
    LEVEL = "m5_1_practice"
    TOTAL = len(M5_1_LAB)

    st.set_page_config(page_title="Module 5_1 — Logistic Regression Lab", page_icon="🔬", layout="centered")

    _init({
        "phase": "login", "email": "", "employee_id": "",
        "attempt_id": None, "step_idx": 0,
        "step_results": {}, "run_out": {}, "scratch_out": {},
        "completed": False,
    })

    phase = st.session_state.phase

    # ── LOGIN ─────────────────────────────────────────────────────
    if phase == "login":
        st.title("🔬 Module 5_1 — Credit Default: Logistic Regression Lab")
        st.markdown("""
| | |
|---|---|
| **Type** | Guided lab — no timer, no pass/fail |
| **Steps** | 10 hands-on steps |
| **Dataset** | 10,000 customers, 50 credit behaviour variables |
| **Goal** | Build end-to-end LR model for default prediction |
| **Max Attempts** | 2 |
""")
        st.info("💡 Each step shows expected output after submission. Scratch pad available on every step.")
        st.divider()
        email = st.text_input("Email", placeholder="you@company.com")
        emp_id = st.text_input("Employee ID", placeholder="EMP12345")
        if st.button("▶ Start Lab", type="primary"):
            if not email or "@" not in email:
                st.error("Enter a valid email."); return
            if not emp_id.strip():
                st.error("Enter your Employee ID."); return
            with st.spinner("Checking eligibility..."):
                _, blocked = check_attempts(email, LEVEL)
            if blocked:
                st.error("⛔ You have used both attempts for this lab."); return
            with st.spinner("Setting up lab..."):
                aid = start_attempt(email, LEVEL, emp_id)
            st.session_state.update({
                "phase": "lab", "email": email, "employee_id": emp_id,
                "attempt_id": aid, "step_idx": 0, "step_results": {},
                "run_out": {}, "scratch_out": {}, "completed": False,
            })
            st.rerun()

    # ── LAB ──────────────────────────────────────────────────────
    elif phase == "lab":
        steps = M5_1_LAB
        idx   = st.session_state.step_idx
        done  = len(st.session_state.step_results)

        # Header
        st.progress(done / TOTAL)
        st.caption(f"Step {idx+1} of {TOTAL}  •  {st.session_state.email}  •  Completed: {done}/{TOTAL}")
        st.divider()

        if idx >= TOTAL:
            # All steps done
            finish_attempt(st.session_state.attempt_id, TOTAL, TOTAL)
            st.session_state.phase = "done"
            st.rerun()
            return

        step = steps[idx]
        st.subheader(step["title"])
        st.markdown(step["context"])

        # ── Dataset viewer (always visible) ──────────────────────
        with st.expander("📊 View Dataset (df) — 10,000 rows × 51 columns", expanded=(idx==0)):
            if "cached_df" not in st.session_state:
                import pandas as pd, numpy as np, warnings
                warnings.filterwarnings('ignore')
                exec(compile(
                    __import__('questions_m5').DF_SETUP, "<string>", "exec"
                ), ns := {})
                st.session_state.cached_df = ns["df"]
            st.dataframe(st.session_state.cached_df.head(100), use_container_width=True, height=280)
            st.caption("Showing first 100 rows. All 10,000 rows are loaded in the code environment.")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Rows", "10,000")
            col2.metric("Features", "50")
            col3.metric("Default Rate", f"{round(st.session_state.cached_df['default_next_cycle'].mean()*100,1)}%")

        st.divider()

        # Show if already submitted
        if idx in st.session_state.step_results:
            r = st.session_state.step_results[idx]
            if r["ok"]:
                st.success("✅ Correct!")
            else:
                st.error("❌ Not matching expected output — but you can proceed.")
            st.markdown("**Expected output:**")
            st.code(step["exp"], language="text")
            st.markdown("**💡 Solution:**")
            st.code(step["solution"], language="python")
            st.markdown(f"**💬 Explanation:** {step['explanation']}")
            st.divider()
            if st.button("Next Step →", type="primary", key=f"next_{idx}"):
                st.session_state.step_idx += 1
                st.rerun()
            return

        # Code editor
        run_key     = f"run_{idx}"
        scratch_key = f"scratch_{idx}"
        st.caption("💻 df and all imports are preloaded — write your solution below:")
        code_val = st.text_area("", height=180, key=f"code_{idx}",
                                placeholder="# write your code here")

        c1, c2 = st.columns([1,3])
        with c1:
            if st.button("▶ Run", key=f"runbtn_{idx}"):
                code = st.session_state.get(f"code_{idx}", "")
                lines = code.strip().splitlines()
                if lines and not any("print" in l for l in lines):
                    lines[-1] = f"print({lines[-1]})"
                out = run_code(step["preload"], "\n".join(lines))
                st.session_state.run_out[idx] = out
        with c2:
            if st.button("Submit →", type="primary", key=f"sub_{idx}"):
                code = st.session_state.get(f"code_{idx}", "")
                if not code.strip():
                    st.warning("Write some code first.")
                else:
                    lines = code.strip().splitlines()
                    if lines and not any("print" in l for l in lines):
                        lines[-1] = f"print({lines[-1]})"
                    got = run_code(step["preload"], "\n".join(lines))
                    ok  = hc(got, step["exp"])
                    st.session_state.step_results[idx] = {"ok": ok, "got": got}
                    st.rerun()

        if st.session_state.run_out.get(idx):
            st.code(st.session_state.run_out[idx], language="text")

        # Scratch pad
        st.markdown("---")
        st.caption("🧪 Scratch Pad — explore freely (won't affect submission)")
        scratch = st.text_area("", height=120, key=f"sp_{idx}",
                               placeholder="# try anything here\n# df is preloaded")
        if st.button("▶ Run Code", key=f"sprun_{idx}"):
            code = st.session_state.get(f"sp_{idx}", "")
            lines = code.strip().splitlines()
            if lines and not any("print" in l for l in lines):
                lines[-1] = f"print({lines[-1]})"
            st.session_state.scratch_out[idx] = run_code(step["preload"], "\n".join(lines))
        if st.session_state.scratch_out.get(idx):
            st.code(st.session_state.scratch_out[idx], language="text")

    # ── DONE ─────────────────────────────────────────────────────
    elif phase == "done":
        st.title("🎉 Lab Complete!")
        st.success(f"You completed all {TOTAL} steps of Module 5_1 — Logistic Regression Lab.")
        st.metric("Steps Completed", f"{TOTAL}/{TOTAL}")
        st.divider()
        st.markdown(f"**Email:** {st.session_state.email}")
        st.markdown(f"**Employee ID:** {st.session_state.employee_id}")
        st.caption("Completion recorded. You may close this window.")

try:
    main()
except Exception as e:
    import traceback
    st.error(f"Error: {e}")
    st.code(traceback.format_exc())
