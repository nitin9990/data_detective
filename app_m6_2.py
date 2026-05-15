import streamlit as st
import hashlib, re
from datetime import datetime, timezone
from supabase import create_client

def _norm(s): return re.sub(r'\s+', '', str(s).strip().lower())
def h(s):     return hashlib.sha256(_norm(s).encode()).hexdigest()
def hc(a, b): return h(a) == h(b)

def _sb(): return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

def run_code(preload, code, timeout=60):
    import io, sys, threading
    result = [None, None]
    def _run():
        buf = io.StringIO(); old = sys.stdout; sys.stdout = buf
        try:    exec(preload + "\n" + code, {})
        except Exception as e: sys.stdout = old; result[1] = str(e); return
        sys.stdout = old; result[0] = buf.getvalue().strip()
    t = threading.Thread(target=_run, daemon=True)
    t.start(); t.join(timeout)
    if t.is_alive(): return "ERROR: Execution timed out (60s — ARIMA fitting can be slow)"
    return f"ERROR: {result[1]}" if result[1] else (result[0] or "")

def check_attempts(email, level):
    res = _sb().table("attempts").select("id").eq("email", email.lower().strip()).eq("level", level).execute()
    return len(res.data) >= 2

def start_attempt(email, level, employee_id, total_steps):
    res = _sb().table("attempts").insert({
        "email": email.lower().strip(), "level": level,
        "test_id": 1, "attempt_num": 1, "max_score": total_steps,
        "employee_id": employee_id.strip(),
    }).execute()
    return res.data[0]["id"]

def finish_attempt(attempt_id, total_steps):
    _sb().table("attempts").update({
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "score": total_steps, "max_score": total_steps,
        "pct": 100.0, "passed": True, "valid": True,
    }).eq("id", attempt_id).execute()

def _init(d):
    for k, v in d.items():
        if k not in st.session_state: st.session_state[k] = v

def main():
    from questions_m6 import M6_2_LAB, DF_SETUP
    LEVEL = "m6_2_practice"
    TOTAL = len(M6_2_LAB)

    st.set_page_config(page_title="Module 6_2 — Fraud Forecasting: Box-Jenkins ARIMA/SARIMA",
                       page_icon="🔬", layout="centered")
    _init({"phase":"login","email":"","employee_id":"","attempt_id":None,
           "step_idx":0,"step_results":{},"run_out":{},"scratch_out":{}})

    if st.session_state.phase == "login":
        st.title("🔬 Module 6_2 — Fraud Loss Forecasting: Box-Jenkins Lab")
        st.markdown("""
| | |
|---|---|
| **Type** | Guided lab — no timer, no pass/fail |
| **Steps** | 10 hands-on steps |
| **Dataset** | Same 60-month fraud dataset as M6_1 |
| **Goal** | Full Box-Jenkins framework: ADF → ACF/PACF → ARIMA → SARIMA → Compare |
| **Max Attempts** | 2 |
""")
        st.info("⏱️ Note: ARIMA/SARIMA fitting steps may take 10–30 seconds to run. This is normal.")
        st.warning("📚 Prerequisites: Complete Module 6_1 first for the MA baseline comparison.")
        st.divider()
        email  = st.text_input("Email", placeholder="you@company.com")
        emp_id = st.text_input("Employee ID", placeholder="EMP12345")
        if st.button("▶ Start Lab", type="primary"):
            if not email or "@" not in email: st.error("Enter a valid email."); return
            if not emp_id.strip(): st.error("Enter your Employee ID."); return
            if check_attempts(email, LEVEL): st.error("⛔ You have used both attempts."); return
            aid = start_attempt(email, LEVEL, emp_id, TOTAL)
            st.session_state.update({"phase":"lab","email":email,"employee_id":emp_id,
                "attempt_id":aid,"step_idx":0,"step_results":{},"run_out":{},"scratch_out":{}})
            st.rerun()

    elif st.session_state.phase == "lab":
        idx  = st.session_state.step_idx
        done = len(st.session_state.step_results)
        if idx >= TOTAL:
            finish_attempt(st.session_state.attempt_id, TOTAL)
            st.session_state.phase = "done"; st.rerun(); return

        st.progress(done / TOTAL)
        st.caption(f"Step {idx+1} of {TOTAL}  •  {st.session_state.email}  •  Done: {done}/{TOTAL}")

        # Box-Jenkins progress tracker
        bj_steps = ["1.Stationarity","2.Differencing","3.ACF","4.PACF",
                     "5.ARIMA Fit","6.Diagnostics","7.SARIMA","8.Auto ARIMA",
                     "9.CI Forecast","10.Compare"]
        cols = st.columns(5)
        for i, label in enumerate(bj_steps):
            with cols[i%5]:
                if i < done: st.markdown(f"✅ {label}")
                elif i == idx: st.markdown(f"🔵 **{label}**")
                else: st.markdown(f"⬜ {label}")
        st.divider()

        step = M6_2_LAB[idx]
        st.subheader(step["title"])
        st.markdown(step["context"])

        # Dataset viewer
        with st.expander("📊 View Dataset (df) — 60 months fraud data", expanded=False):
            if "cached_df" not in st.session_state:
                ns = {}; exec(compile(DF_SETUP, "<string>", "exec"), ns)
                st.session_state.cached_df = ns["df"].reset_index()
            st.dataframe(st.session_state.cached_df, use_container_width=True, height=250)
            c1,c2,c3 = st.columns(3)
            c1.metric("Train (S1-S6)", "54 months (Jan 2019–Jun 2023)")
            c2.metric("Test (holdout)", "6 months (Jul–Dec 2023)")
            c3.metric("Seasonality", "Monthly, m=12")
        st.divider()

        # Already submitted
        if idx in st.session_state.step_results:
            r = st.session_state.step_results[idx]
            st.success("✅ Correct!") if r["ok"] else st.error("❌ Not matching — you can still proceed.")
            st.markdown("**Expected output:**"); st.code(step["exp"], language="text")
            st.markdown("**💡 Solution:**"); st.code(step["solution"], language="python")
            st.markdown(f"**💬 Explanation:** {step['explanation']}")
            st.divider()
            if st.button("Next Step →", type="primary", key=f"next_{idx}"):
                st.session_state.step_idx += 1; st.rerun()
            return

        st.caption("💻 df, series, train_s, test_s, adfuller, acf, pacf, ARIMA, SARIMAX are preloaded:")
        if idx >= 4:
            st.info("⏱️ ARIMA/SARIMA fitting may take 10–30 seconds. Click Run and wait.")
        code_val = st.text_area("", height=180, key=f"code_{idx}", placeholder="# write your code here")
        c1, c2 = st.columns([1,3])
        with c1:
            if st.button("▶ Run", key=f"runbtn_{idx}"):
                code = st.session_state.get(f"code_{idx}","")
                lines = code.strip().splitlines()
                if lines and not any("print" in l for l in lines): lines[-1]=f"print({lines[-1]})"
                with st.spinner("Running... (may take up to 30s for ARIMA steps)"):
                    st.session_state.run_out[idx] = run_code(step["preload"], "\n".join(lines))
        with c2:
            if st.button("Submit →", type="primary", key=f"sub_{idx}"):
                code = st.session_state.get(f"code_{idx}","")
                if not code.strip(): st.warning("Write some code first."); return
                lines = code.strip().splitlines()
                if lines and not any("print" in l for l in lines): lines[-1]=f"print({lines[-1]})"
                with st.spinner("Running and evaluating..."):
                    got = run_code(step["preload"], "\n".join(lines))
                st.session_state.step_results[idx] = {"ok": hc(got, step["exp"]), "got": got}
                st.rerun()
        if st.session_state.run_out.get(idx):
            st.code(st.session_state.run_out[idx], language="text")

        st.markdown("---")
        st.caption("🧪 Scratch Pad — explore freely")
        scratch = st.text_area("", height=100, key=f"sp_{idx}",
                               placeholder="# try anything\n# series, train_s, test_s all available")
        if st.button("▶ Run Code", key=f"sprun_{idx}"):
            code = st.session_state.get(f"sp_{idx}","")
            lines = code.strip().splitlines()
            if lines and not any("print" in l for l in lines): lines[-1]=f"print({lines[-1]})"
            with st.spinner("Running..."):
                st.session_state.scratch_out[idx] = run_code(step["preload"], "\n".join(lines))
        if st.session_state.scratch_out.get(idx):
            st.code(st.session_state.scratch_out[idx], language="text")

    elif st.session_state.phase == "done":
        st.title("🎉 Box-Jenkins Lab Complete!")
        st.success(f"You completed all {TOTAL} steps of Module 6_2 — Box-Jenkins ARIMA/SARIMA.")
        st.balloons()
        st.metric("Steps Completed", f"{TOTAL}/{TOTAL}")
        st.divider()
        st.markdown("**What you've mastered:**")
        st.markdown("""
- ✅ ADF stationarity test and interpretation
- ✅ First-order differencing (d parameter)
- ✅ ACF/PACF reading for p and q selection
- ✅ ARIMA model fitting and AIC comparison
- ✅ Residual diagnostics (Ljung-Box test)
- ✅ SARIMA for seasonal data
- ✅ Auto ARIMA vs manual selection
- ✅ Forecast with confidence intervals
- ✅ Model comparison: MA vs ARIMA vs SARIMA
""")
        st.divider()
        st.markdown(f"**Email:** {st.session_state.email}")
        st.markdown(f"**Employee ID:** {st.session_state.employee_id}")
        st.caption("Completion recorded. You may close this window.")

try:
    main()
except Exception as e:
    import traceback
    st.error(f"Error: {e}"); st.code(traceback.format_exc())
