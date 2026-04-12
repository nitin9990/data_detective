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
    # CSS only — scripts in st.markdown don't execute, components.html is sandboxed
    st.markdown("""
<style>
* { -webkit-user-select:none; -moz-user-select:none; user-select:none; }
input, textarea,
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {
    -webkit-user-select:text !important; user-select:text !important;
}
</style>
""", unsafe_allow_html=True)

def _sync_tab_switches():
    """
    Heartbeat approach — no JS needed.
    autorefresh fires every 1s when tab is active.
    Browsers throttle hidden tabs → gap > 5s = tab was switched.
    """
    now = time.time()
    last = st.session_state.get("last_ping", now)
    gap  = now - last

    if gap > 5 and st.session_state.get("phase") == "test":
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
    return hc(run_code(q["preload"], val), q["exp"])

# ── SUPABASE ─────────────────────────────────────────────────────
def _sb(): return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

def check_attempts(email, level):
    cutoff = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    res = _sb().table("attempts").select("id").eq("email", email.lower().strip()).eq("level", level).gte("started_at", cutoff).execute()
    count = len(res.data)
    return count + 1, count >= 3

def start_attempt(email, level, test_id, attempt_num, max_score):
    res = _sb().table("attempts").insert({"email":email.lower().strip(),"level":level,"test_id":test_id,"attempt_num":attempt_num,"max_score":max_score}).execute()
    return res.data[0]["id"]

def finish_attempt(attempt_id, score, max_score, time_limit, results, tab_switches=0):
    taken = time_limit - time_left(time_limit)
    pct   = round(score / max_score * 100, 1) if max_score else 0
    _sb().table("attempts").update({"submitted_at":datetime.now(timezone.utc).isoformat(),"score":score,"pct":pct,"passed":pct>=80,"valid":taken<=time_limit,"results":results,"tab_switches":tab_switches}).eq("id", attempt_id).execute()
    return pct, pct>=80, taken<=time_limit, taken

# ── SESSION INIT ─────────────────────────────────────────────────
def _init(defaults):
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
def run_app(level, tests_map, time_limit, title):
    _init({"phase":"email","email":"","test_id":None,"q_idx":0,"score":0,"max_score":0,
           "results":[],"started_at":None,"attempt_id":None,"attempt_num":1,
           "run_out":"","pct":0,"passed":False,"valid":True,"taken":0,
           "tab_switches":0})

    st.set_page_config(page_title=title, page_icon="📋", layout="centered")
    _inject_security()
    _sync_tab_switches()

    p = st.session_state.phase
    if   p == "email": _email_phase(level, tests_map, time_limit, title)
    elif p == "test":  _test_phase(tests_map, time_limit)
    elif p == "score": _score_phase(title)

# ── EMAIL PHASE ──────────────────────────────────────────────────
def _email_phase(level, tests_map, time_limit, title):
    st.title(title)
    st.markdown(f"""
| | |
|---|---|
| **Duration** | {time_limit//60} minutes |
| **Pass Mark** | 80% |
| **Max Attempts** | 3 *(within any 3-day window)* |
| **Questions** | One at a time — no going back |
""")
    st.divider()
    email = st.text_input("Enter your email to begin", placeholder="you@company.com")
    if st.button("▶  Start Test", type="primary"):
        if not email or "@" not in email:
            st.error("Enter a valid email address."); return
        with st.spinner("Checking eligibility..."):
            attempt_num, blocked = check_attempts(email, level)
        if blocked:
            st.error("⛔  You have used all 3 attempts in the last 3 days."); return
        test_id   = random.choice(list(tests_map.keys()))
        q_list    = tests_map[test_id]
        max_score = sum(q["marks"] for q in q_list)
        with st.spinner("Setting up your test..."):
            attempt_id = start_attempt(email, level, test_id, attempt_num, max_score)
        st.session_state.update({"phase":"test","email":email,"test_id":test_id,"q_idx":0,
            "score":0,"max_score":max_score,"results":[],"started_at":time.time(),
            "attempt_id":attempt_id,"attempt_num":attempt_num,"run_out":""})
        st.rerun()

# ── TEST PHASE ───────────────────────────────────────────────────
def _test_phase(tests_map, time_limit):
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=1000, key="ticker")

    tl = time_left(time_limit)
    if tl <= 0:
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
        col = "red" if tl<300 else "orange" if tl<(time_limit*0.25) else "green"
        st.markdown(f"<div style='text-align:right'><span style='font-size:1.4rem;font-weight:700;color:{col}'>⏱ {fmt_time(tl)}</span></div>", unsafe_allow_html=True)

    st.divider()
    badge = {"mcq":"MCQ","fill":"Fill in Blank","code":"Write Code"}
    st.markdown(f"**Q{q['id']}** &nbsp;&nbsp;<span style='background:#e8f0fe;padding:2px 8px;border-radius:4px;font-size:0.8rem'>{badge[q['type']]}</span>&nbsp;&nbsp;<span style='background:#fef9c3;padding:2px 8px;border-radius:4px;font-size:0.8rem'>{q['marks']} mark{'s' if q['marks']>1 else ''}</span>", unsafe_allow_html=True)
    st.markdown("")

    if q["type"] == "code":
        st.info(q["text"])
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
        st.caption("💻 Variables / DataFrames are preloaded — write your solution below:")
        code_val = st.text_area("Code editor:", height=160, key=f"inp_{q_idx}", placeholder="# write your code here")
        c1, c2 = st.columns([1,3])
        with c1:
            if st.button("▶ Run", key=f"run_{q_idx}"):
                st.session_state.run_out = run_code(q["preload"], code_val or "")
        with c2:
            if st.button("Submit →", type="primary", key=f"sub_{q_idx}"):
                if not (code_val or "").strip(): st.warning("Write some code first.")
                else: _record(q, code_val, q_list, time_limit)
        if st.session_state.run_out:
            st.code(st.session_state.run_out, language="text")

def _record(q, val, q_list, time_limit):
    ok = grade(q, val)
    st.session_state.score += q["marks"] if ok else 0
    st.session_state.results.append({"id":q["id"],"ok":ok,"got":q["marks"] if ok else 0,"max":q["marks"]})
    st.session_state.run_out = ""
    if st.session_state.q_idx < len(q_list)-1:
        st.session_state.q_idx += 1; st.rerun()
    else:
        _do_submit(time_limit)

def _do_submit(time_limit):
    pct, passed, valid, taken = finish_attempt(
        st.session_state.attempt_id, st.session_state.score,
        st.session_state.max_score, time_limit, st.session_state.results,
        st.session_state.get("tab_switches", 0)
    )
    st.session_state.update({"phase":"score","pct":pct,"passed":passed,"valid":valid,"taken":taken})
    st.rerun()

# ── SCORE PHASE ──────────────────────────────────────────────────
def _score_phase(title):
    st.title("Test Complete")
    score=st.session_state.score; max_score=st.session_state.max_score
    pct=st.session_state.pct; passed=st.session_state.passed
    valid=st.session_state.valid; taken=st.session_state.taken

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
