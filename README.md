# Axiontech Assessment Platform — Setup Guide

## Step 1 — Supabase (5 min)

1. Go to https://supabase.com → New Project → name it `axiontest`
2. Once created → **SQL Editor** → paste `supabase_schema.sql` → Run
3. Go to **Settings → API** → copy:
   - `Project URL`
   - `anon/public key`

---

## Step 2 — GitHub (5 min)

1. Create a new repo at https://github.com → name it `axiontest`
2. Upload all files from this folder to the repo root
3. Commit

---

## Step 3 — Streamlit Cloud (10 min — gives you 2 links)

Go to https://share.streamlit.io → **New app**

### App 1 — Beginner
| Field | Value |
|---|---|
| Repository | `your-github-username/axiontest` |
| Branch | `main` |
| Main file path | `app_beginner.py` |

Click **Advanced settings → Secrets** and paste:
```toml
SUPABASE_URL = "https://xxxx.supabase.co"
SUPABASE_KEY = "your-anon-key"
```

Deploy → copy the URL → this is your **Beginner test link**

---

### App 2 — Intermediate
Same steps, but Main file path = `app_intermediate.py`
Same secrets.

Deploy → copy the URL → this is your **Intermediate test link**

---

## Step 4 — View Results (Supabase Dashboard)

Go to Supabase → **Table Editor → attempts**

Columns you'll see:
- `email` — candidate
- `level` — beginner / intermediate
- `test_id` — which of the 5 tests they got
- `score / max_score / pct` — marks
- `passed` — true / false
- `valid` — false if submitted after time limit
- `attempt_num` — 1st, 2nd, or 3rd attempt
- `started_at / submitted_at` — timestamps

Export as CSV anytime via **Table Editor → Download**.

---

## Adding Tests 2–5 (later)

In `questions.py`, add `BEGINNER_T2 = [...]` etc.  
Then update:
```python
BEGINNER_TESTS     = {1: BEGINNER_T1, 2: BEGINNER_T2, ...}
INTERMEDIATE_TESTS = {1: INTERMEDIATE_T1, 2: INTERMEDIATE_T2, ...}
```
Push to GitHub → Streamlit auto-redeploys. Zero downtime.
