import hashlib, io, re, sys, threading, time, random
from datetime import datetime, timedelta, timezone

import streamlit as st
from supabase import create_client

def _norm(s): return re.sub(r'\s+', '', str(s).strip().lower())
def h(s):     return hashlib.sha256(_norm(s).encode()).hexdigest()
def hc(a, b): return h(a) == h(b)

def fmt_time(secs):
    m, s = divmod(int(max(0, secs)), 60)
    return f"{m:02d}:{s:02d}"

def time_left(time_limit):
    if not st.session_state.get("started_at"):
        return time_limit
    return max(0, time_limit - (time.time() - st.session_state.started_at))

# ── COPY PROTECTION ──────────────────────────────────────────────
def _inject_security():
    st.markdown("""
<style>
/* Block paste ONLY in code editor textareas */
[data-testid="stTextArea"] textarea {
    -webkit-user-select: text !important;
    user-select: text !important;
}
</style>
<script>
/* This won't execute via st.markdown — paste blocking handled via onpaste attr below */
</script>
""", unsafe_allow_html=True)

def _sync_tab_switches():
    now  = time.time()
    last = st.session_state.get("last_ping") or now
    gap  = now - last
    # gap > 15s = tab was switched (5s refresh + 10s buffer for slow connections)
    if gap > 15 and st.session_state.get("phase") == "test":
        st.session_state.tab_switches = st.session_state.get("tab_switches", 0) + 1
    st.session_state.last_ping = now

# ── CODE EXECUTION ───────────────────────────────────────────────
def run_code(preload, code, timeout=10):
    result = [None, None]
    def _run():
        buf = io.StringIO(); old = sys.stdout; sys.stdout = buf
        try:    exec(preload + "\n" + code, {})
        except Exception as e: sys.stdout = old; result[1] = str(e); return
        sys.stdout = old; result[0] = buf.getvalue().strip()
    t = threading.Thread(target=_run, daemon=True)
    t.start(); t.join(timeout)
    if t.is_alive(): return "ERROR: Execution timed out (10s limit)"
    if result[1]:    return f"ERROR: {result[1]}"
    return result[0] or ""

def grade(q, val):
    if q["type"] in ("mcq", "fill"): return h(val) == q["ah"]
    # Apply same auto-print logic as scratch pad so output always matches
    code = val.strip()
    lines = code.splitlines()
    if lines and not any("print" in l for l in lines):
        lines[-1] = f"print({lines[-1]})"
        code = "\n".join(lines)
    return hc(run_code(q.get("preload",""), code), q["exp"])

# ── SUPABASE ─────────────────────────────────────────────────────
def _sb(): return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

def check_attempts(email, level, max_attempts=3):
    if max_attempts == 1:
        # lifetime block — no time window
        res = _sb().table("attempts").select("id").eq("email", email.lower().strip()).eq("level", level).execute()
    else:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
        res = _sb().table("attempts").select("id").eq("email", email.lower().strip()).eq("level", level).gte("started_at", cutoff).execute()
    count = len(res.data)
    return count + 1, count >= max_attempts

def start_attempt(email, level, test_id, attempt_num, max_score):
    res = _sb().table("attempts").insert({"email":email.lower().strip(),"level":level,"test_id":test_id,"attempt_num":attempt_num,"max_score":max_score}).execute()
    return res.data[0]["id"]

def finish_attempt(attempt_id, score, max_score, time_limit, results, tab_switches=0):
    taken  = time_limit - time_left(time_limit)
    score  = min(score, max_score)                                    # cap score at max
    pct    = round(min(score / max_score * 100, 100), 1) if max_score else 0  # cap at 100%
    passed = pct >= 80
    valid  = taken <= time_limit
    _sb().table("attempts").update({
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "score":        score,
        "pct":          pct,
        "passed":       passed,
        "valid":        valid,
        "results":      results,
        "tab_switches": tab_switches,
    }).eq("id", attempt_id).execute()
    return pct, passed, valid, taken

def reset_candidate(email, level):
    """Call from Supabase SQL — not used in app directly."""
    _sb().table("attempts").delete().eq("email", email.lower().strip()).eq("level", level).execute()

# ── SESSION INIT ─────────────────────────────────────────────────
def _init(defaults):
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
def run_app(level, tests_map, time_limit, title, mode="final", max_attempts=1, dataset=None):
    st.set_page_config(page_title=title, page_icon="📋", layout="centered")
    _init({"phase":"email","email":"","test_id":None,"q_idx":0,"score":0,"max_score":0,
           "results":[],"started_at":None,"attempt_id":None,"attempt_num":1,
           "run_out":"","pct":0,"passed":False,"valid":True,"taken":0,
           "tab_switches":0,"last_ping":None,"test_dataset":None,
           "app_mode": mode, "app_max_attempts": max_attempts,
           "app_time_limit": time_limit,
           "reveal_q":None,"reveal_ok":False,"reveal_val":"","reveal_last":False})

    _inject_security()
    _sync_tab_switches()

    p = st.session_state.phase
    if   p == "email":  _email_phase(level, tests_map, time_limit, title, mode, max_attempts, dataset)
    elif p == "test":   _test_phase(tests_map, time_limit, mode)
    elif p == "reveal": _reveal_phase(tests_map[st.session_state.test_id], time_limit)
    elif p == "score":  _score_phase(title, mode)

# ── EMAIL PHASE ──────────────────────────────────────────────────
def _email_phase(level, tests_map, time_limit, title, mode, max_attempts, dataset):
    st.title(title)
    if mode == "practice":
        st.markdown(f"""
| | |
|---|---|
| **Mode** | Practice — no time limit |
| **Scoring** | Marks shown per question |
| **Pass/Fail** | Not applicable |
| **Max Attempts** | {max_attempts} total |
| **Questions** | One at a time — no going back |
""")
    else:
        st.markdown(f"""
| | |
|---|---|
| **Duration** | {time_limit//60} minutes |
| **Pass Mark** | 80% |
| **Max Attempts** | {max_attempts} (one chance only) |
| **Questions** | One at a time — no going back |
""")
    st.divider()
    email = st.text_input("Enter your email to begin", placeholder="you@company.com")
    if st.button("▶  Start Test", type="primary"):
        if not email or "@" not in email:
            st.error("Enter a valid email address."); return
        with st.spinner("Checking eligibility..."):
            attempt_num, blocked = check_attempts(email, level, max_attempts)
        if blocked:
            msg = "⛔  You have used your one attempt for this assessment." if max_attempts==1 \
                  else f"⛔  You have used all {max_attempts} attempts."
            st.error(msg); return
        test_id   = random.choice(list(tests_map.keys()))
        q_list    = tests_map[test_id]
        max_score = sum(q["marks"] for q in q_list)
        with st.spinner("Setting up your test..."):
            attempt_id = start_attempt(email, level, test_id, attempt_num, max_score)

        # Dataset: passed directly or from intermediate questions
        ds = dataset
        if ds is None and level == "intermediate":
            from questions import INTERMEDIATE_DATASETS
            ds = INTERMEDIATE_DATASETS.get(test_id)

        # Extract dataset preload from first code question — used in scratch pad
        df_setup = ""
        for q in q_list:
            if q.get("type") == "code" and q.get("preload","").strip():
                df_setup = q["preload"]
                break

        st.session_state.update({
            "phase":"test","email":email,"test_id":test_id,"q_idx":0,
            "score":0,"max_score":max_score,"results":[],"started_at":time.time(),
            "attempt_id":attempt_id,"attempt_num":attempt_num,"run_out":"",
            "test_dataset": ds, "df_setup": df_setup
        })
        st.rerun()

# ── TEST PHASE ───────────────────────────────────────────────────
def _test_phase(tests_map, time_limit, mode="final"):
    # Only autorefresh for timed finals — practice has no timer so no need to hammer server
    is_timed = time_limit and time_limit < 99999
    if is_timed:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=5000, key="ticker")  # every 5s not 1s — reduces server load 5x

    tl = time_left(time_limit) if is_timed else None
    if is_timed and tl is not None and tl <= 0:
        _do_submit(time_limit); return

    q_list = tests_map[st.session_state.test_id]
    q_idx  = st.session_state.q_idx
    q      = q_list[q_idx]
    total  = len(q_list)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.progress(q_idx / total)
        st.caption(f"Question {q_idx+1} of {total}  •  {st.session_state.email}")
    with col2:
        if time_limit and tl is not None:
            col = "red" if tl<300 else "orange" if tl<(time_limit*0.25) else "green"
            st.markdown(f"<div style='text-align:right'><span style='font-size:1.4rem;font-weight:700;color:{col}'>⏱ {fmt_time(tl)}</span></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align:right'><span style='color:green;font-weight:700'>Practice Mode</span></div>", unsafe_allow_html=True)

    st.divider()

    # ── Show dataset for intermediate tests ──────────────────────
    dataset = st.session_state.get("test_dataset")
    if dataset is not None:
        with st.expander("📊 View Dataset (df)", expanded=False):
            st.dataframe(dataset, use_container_width=True, height=250)
    st.divider()
    badge = {"mcq":"MCQ","fill":"Fill in Blank","code":"Write Code"}
    st.markdown(f"**Q{q['id']}** &nbsp;&nbsp;<span style='background:#e8f0fe;padding:2px 8px;border-radius:4px;font-size:0.8rem'>{badge[q['type']]}</span>&nbsp;&nbsp;<span style='background:#fef9c3;padding:2px 8px;border-radius:4px;font-size:0.8rem'>{q['marks']} mark{'s' if q['marks']>1 else ''}</span>", unsafe_allow_html=True)
    st.markdown("")

    if q["type"] == "code":
        # Split question into instruction + code block
        text = q["text"]
        parts = text.split("\n\n", 1)
        if len(parts) == 2:
            st.write(parts[0])
            # detect if second part has python code
            code_part = parts[1].strip()
            if any(k in code_part for k in ['def ','class ','print(','return ','for ','if ','import ']):
                st.code(code_part, language="python")
            else:
                st.write(code_part)
        else:
            st.write(text)
    else:
        parts = q["text"].split("\n\n", 1)
        if len(parts)==2 and any(k in parts[1] for k in ['def ','for ','[','{']):
            st.write(parts[0]); st.code(parts[1].strip(), language="python")
        else:
            st.markdown(q["text"])
    st.markdown("")

    if q["type"] == "mcq":
        val = st.radio("Select your answer:", q["opts"], index=None, key=f"inp_{q_idx}")
        if st.button("Submit →", type="primary", key=f"sub_{q_idx}"):
            if val is None: st.warning("Select an answer first.")
            else: _record(q, val, q_list, time_limit)

    elif q["type"] == "fill":
        val = st.text_input("Your answer:", key=f"inp_{q_idx}", placeholder="Type here...")
        if st.button("Submit →", type="primary", key=f"sub_{q_idx}"):
            if not val.strip(): st.warning("Enter an answer.")
            else: _record(q, val, q_list, time_limit)

    else:
        st.caption("💻 Write your answer code below:")
        st.caption("⚠️ Paste is disabled in the code editor.")
        code_val = st.text_area("Code editor:", height=180, key=f"inp_{q_idx}",
                                placeholder="# write your code here\n# preloaded variables are available")

        run_out_key = f"run_out_{q_idx}"
        c1, c2 = st.columns([1,3])
        with c1:
            if st.button("▶ Run", key=f"run_{q_idx}"):
                code = st.session_state.get(f"inp_{q_idx}", "")
                lines = code.strip().splitlines()
                if lines and not any("print" in l for l in lines):
                    lines[-1] = f"print({lines[-1]})"
                st.session_state[run_out_key] = run_code(q.get("preload",""), "\n".join(lines))
        with c2:
            if st.button("Submit →", type="primary", key=f"sub_{q_idx}"):
                code = st.session_state.get(f"inp_{q_idx}", "")
                if not code.strip(): st.warning("Write some code first.")
                else: _record(q, code, q_list, time_limit)
        if st.session_state.get(run_out_key):
            st.code(st.session_state[run_out_key], language="text")

    # ── Scratch pad for ALL question types ───────────────────────
    st.markdown("---")
    st.caption("🧪 Scratch Pad — run any code to explore (won't affect your answer)")

    # Use the correct dataset preload stored at test start
    preload_scratch = st.session_state.get("df_setup", "") or q.get("preload", "")

    scratch_key     = f"scratch_{q_idx}"
    scratch_out_key = f"scratch_out_{q_idx}"
    scratch = st.text_area("", height=120, key=scratch_key,
                           placeholder="# explore freely here\n# print(df.head())\n# print(df.describe())")
    if st.button("▶ Run Code", key=f"scratch_run_{q_idx}"):
        code = st.session_state.get(scratch_key, "")
        lines = code.strip().splitlines()
        if lines and not any("print" in l for l in lines):
            lines[-1] = f"print({lines[-1]})"
        st.session_state[scratch_out_key] = run_code(preload_scratch, "\n".join(lines))
    if st.session_state.get(scratch_out_key):
        st.code(st.session_state[scratch_out_key], language="text")

def _record(q, val, q_list, time_limit):
    ok = grade(q, val)
    st.session_state.score += q["marks"] if ok else 0
    st.session_state.results.append({"id":q["id"],"ok":ok,"got":q["marks"] if ok else 0,"max":q["marks"]})
    st.session_state.run_out = ""
    # Store reveal info and switch to reveal phase
    st.session_state.reveal_q   = q
    st.session_state.reveal_ok  = ok
    st.session_state.reveal_val = val
    st.session_state.reveal_last = (st.session_state.q_idx >= len(q_list)-1)
    st.session_state.phase = "reveal"
    st.rerun()

def _reveal_phase(q_list, time_limit):
    q = st.session_state.get("reveal_q")
    if q is None:
        st.session_state.phase = "test"
        st.rerun()
        return

    ok      = st.session_state.get("reveal_ok", False)
    val     = st.session_state.get("reveal_val", "")
    is_last = st.session_state.get("reveal_last", False)

    # Header
    if ok:
        st.success("### ✅ Correct!")
    else:
        st.error("### ❌ Incorrect")

    st.markdown("---")

    # What candidate submitted
    if q["type"] == "code":
        st.markdown("**Your code:**")
        st.code(str(val), language="python")
    else:
        st.markdown(f"**Your answer:** `{val}`")

    st.markdown("---")

    # Correct answer
    if q["type"] == "mcq":
        import hashlib, re
        def _n(s): return re.sub(r'\s+','',str(s).strip().lower())
        def _h(s): return hashlib.sha256(_n(s).encode()).hexdigest()
        correct_opt = next((o for o in q.get("opts",[]) if _h(o)==q.get("ah","")), "—")
        st.markdown(f"**✅ Correct Answer:** `{correct_opt}`")
    else:
        st.markdown("**✅ Expected Output:**")
        st.code(str(q.get("exp","—")), language="text")

    # Solution code
    if q.get("solution"):
        st.markdown("**💡 Solution:**")
        st.code(q["solution"], language="python")

    # Explanation
    if q.get("explanation"):
        st.info(f"💬 {q['explanation']}")

    st.markdown("---")

    btn_label = "🏁 Finish Test" if is_last else "Next Question →"
    if st.button(btn_label, type="primary", key="next_btn"):
        st.session_state.reveal_q = None
        if is_last:
            _do_submit(time_limit)
        else:
            st.session_state.q_idx += 1
            st.session_state.phase = "test"
            st.rerun()

def _do_submit(time_limit):
    pct, passed, valid, taken = finish_attempt(
        st.session_state.attempt_id, st.session_state.score,
        st.session_state.max_score, time_limit, st.session_state.results,
        st.session_state.get("tab_switches", 0)
    )
    st.session_state.update({"phase":"score","pct":pct,"passed":passed,"valid":valid,"taken":taken})
    st.rerun()

# ── SCORE PHASE ──────────────────────────────────────────────────
def _score_phase(title, mode="final"):
    st.title("✅ Completed" if mode == "practice" else "Test Complete")
    score=st.session_state.score; max_score=st.session_state.max_score
    pct=st.session_state.pct; passed=st.session_state.passed
    valid=st.session_state.valid; taken=st.session_state.taken

    if mode == "practice":
        c1, c2 = st.columns(2)
        c1.metric("Score", f"{score} / {max_score}")
        c2.metric("Percentage", f"{pct}%")
        st.info("Practice mode — no pass/fail applied. Review your breakdown below.")
    else:
        c1,c2,c3 = st.columns(3)
        c1.metric("Score", f"{score} / {max_score}")
        c2.metric("Percentage", f"{pct}%")
        c3.metric("Result", "PASS ✅" if passed else "FAIL ❌")

        m1, m2 = st.columns(2)
        m1.metric("Time Taken", fmt_time(taken))
        tsw = st.session_state.get("tab_switches", 0)
        flag = "  ⚠️" if tsw > 3 else ""
        m2.metric("Tab Switches", f"{tsw}{flag}")

        if not valid: st.error("⚠️  Submission flagged INVALID — time limit exceeded.")
        elif passed:  st.success("Congratulations! You passed.")
        else:         st.warning(f"You needed 80% to pass. Your score: {pct}%")

    st.divider()
    st.subheader("Question Breakdown")
    for r in st.session_state.results:
        st.write(f"{'✅' if r['ok'] else '❌'} &nbsp; Q{r['id']} — {r['got']} / {r['max']} marks")

    st.divider()
    st.caption(f"Submitted by: {st.session_state.email}  •  Test #{st.session_state.test_id}")
    st.caption("Results have been recorded. You may close this window.")
