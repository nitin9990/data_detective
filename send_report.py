"""
send_report.py — Automated bi-weekly assessment report
- Finals only (m1_final, m2_final, m3_final, m4_final)
- Email dedup: abc1@domain.com → abc@domain.com
- Max pct per canonical email per module
- All-time cumulative
- Module-wise scorecard with Streamlit links
"""

import os, re, smtplib
from collections import defaultdict
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from supabase import create_client

# ── CONFIG ────────────────────────────────────────────────────────
SUPABASE_URL   = os.environ["SUPABASE_URL"]
SUPABASE_KEY   = os.environ["SUPABASE_KEY"]
GMAIL_USER     = os.environ["GMAIL_USER"]
GMAIL_PASSWORD = os.environ["GMAIL_APP_PASSWORD"]
RECIPIENTS     = [e.strip() for e in os.environ["REPORT_RECIPIENTS"].split(",")]

FINALS = {
    "m1_final": {
        "label": "Module 1 — Python Core",
        "link":  "https://datadetective-5tcx2npnjnjff8wspr3vwz.streamlit.app/",
    },
    "m2_final": {
        "label": "Module 2 — Data Preprocessing",
        "link":  "https://datadetective-ndviz8g48ckcnpdeje9pzm.streamlit.app/",
    },
    "m3_final": {
        "label": "Module 3 — EDA & Statistics",
        "link":  "https://datadetective-2tmnswhpkxh7vq6yqegrqs.streamlit.app/",
    },
    "m4_final": {
        "label": "Module 4 — Clustering & Dimensionality Reduction",
        "link":  "https://datadetective-lpkbpzxta27t76oadekld2.streamlit.app/",
    },
}

# ── EMAIL DEDUPLICATION ───────────────────────────────────────────
def canonical(email):
    """abc1@domain.com → abc@domain.com. Strips single trailing digit from prefix."""
    email = email.lower().strip()
    prefix, domain = email.split("@", 1)
    prefix = re.sub(r'\d+$', '', prefix)   # remove trailing digits
    return f"{prefix}@{domain}"

# ── FETCH DATA ────────────────────────────────────────────────────
def fetch_finals():
    sb  = create_client(SUPABASE_URL, SUPABASE_KEY)
    res = sb.table("attempts") \
            .select("email,level,pct,passed,score,max_score,submitted_at") \
            .in_("level", list(FINALS.keys())) \
            .execute()
    return res.data

# ── AGGREGATE ────────────────────────────────────────────────────
def aggregate(data):
    """
    For each (canonical_email, level) keep only the highest pct attempt.
    Returns dict: level → list of {email, pct}
    """
    # best[level][canonical_email] = max pct
    best = defaultdict(dict)
    for r in data:
        lvl   = r["level"]
        email = canonical(r.get("email", ""))
        pct   = r.get("pct") or 0
        if email not in best[lvl] or pct > best[lvl][email]:
            best[lvl][email] = pct
    return best

# ── BUILD HTML ────────────────────────────────────────────────────
def build_html(best):
    now_str = datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC")

    module_blocks = ""
    grand_appeared = grand_passed = 0

    for level, meta in FINALS.items():
        candidates = best.get(level, {})  # {email: pct}
        appeared   = len(candidates)
        passed     = sum(1 for p in candidates.values() if p >= 80)
        band_60_80 = sum(1 for p in candidates.values() if 60 <= p < 80)
        below_60   = sum(1 for p in candidates.values() if p < 60)
        grand_appeared += appeared
        grand_passed   += passed

        pass_rate = f"{round(passed/appeared*100,1)}%" if appeared else "—"

        # Per-candidate table rows
        rows = ""
        for email, pct in sorted(candidates.items(), key=lambda x: -x[1]):
            if pct >= 80:
                badge = "<span style='color:#16a34a;font-weight:bold'>✅ PASS</span>"
            elif pct >= 60:
                badge = "<span style='color:#d97706;font-weight:bold'>⚠️ Near</span>"
            else:
                badge = "<span style='color:#dc2626;font-weight:bold'>❌ FAIL</span>"
            rows += f"""
            <tr>
                <td style='padding:6px 12px;border-bottom:1px solid #e5e7eb'>{email}</td>
                <td style='padding:6px 12px;border-bottom:1px solid #e5e7eb;text-align:center'>{pct}%</td>
                <td style='padding:6px 12px;border-bottom:1px solid #e5e7eb;text-align:center'>{badge}</td>
            </tr>"""

        module_blocks += f"""
        <div style='margin:28px 0;padding:20px;background:#f8fafc;border-radius:8px;border-left:4px solid #1e3a5f'>
            <h2 style='margin:0 0 4px 0;color:#1e3a5f;font-size:1.1rem'>{meta['label']}</h2>
            <a href='{meta['link']}' style='font-size:0.8rem;color:#2563eb'>{meta['link']}</a>

            <div style='display:flex;gap:16px;flex-wrap:wrap;margin:16px 0'>
                <div style='background:white;border-radius:6px;padding:12px 20px;min-width:120px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.08)'>
                    <div style='font-size:1.8rem;font-weight:700;color:#1e3a5f'>{appeared}</div>
                    <div style='font-size:0.75rem;color:#6b7280'>Appeared</div>
                </div>
                <div style='background:white;border-radius:6px;padding:12px 20px;min-width:120px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.08)'>
                    <div style='font-size:1.8rem;font-weight:700;color:#16a34a'>{passed}</div>
                    <div style='font-size:0.75rem;color:#6b7280'>Passed (≥80%)</div>
                </div>
                <div style='background:white;border-radius:6px;padding:12px 20px;min-width:120px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.08)'>
                    <div style='font-size:1.8rem;font-weight:700;color:#d97706'>{band_60_80}</div>
                    <div style='font-size:0.75rem;color:#6b7280'>Score 60–80%</div>
                </div>
                <div style='background:white;border-radius:6px;padding:12px 20px;min-width:120px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.08)'>
                    <div style='font-size:1.8rem;font-weight:700;color:#dc2626'>{below_60}</div>
                    <div style='font-size:0.75rem;color:#6b7280'>Score &lt;60%</div>
                </div>
                <div style='background:white;border-radius:6px;padding:12px 20px;min-width:120px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.08)'>
                    <div style='font-size:1.8rem;font-weight:700;color:#7c3aed'>{pass_rate}</div>
                    <div style='font-size:0.75rem;color:#6b7280'>Pass Rate</div>
                </div>
            </div>

            {'<table style="width:100%;border-collapse:collapse;background:white;border-radius:6px;overflow:hidden"><thead><tr style="background:#1e3a5f;color:white"><th style="padding:8px 12px;text-align:left">Email</th><th style="padding:8px 12px;text-align:center">Best Score</th><th style="padding:8px 12px;text-align:center">Status</th></tr></thead><tbody>' + rows + '</tbody></table>' if rows else '<p style="color:#9ca3af;font-style:italic">No attempts yet.</p>'}
        </div>"""

    overall_pass_rate = f"{round(grand_passed/grand_appeared*100,1)}%" if grand_appeared else "—"

    return f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'></head>
<body style='font-family:Arial,sans-serif;color:#1f2937;margin:0;padding:20px;max-width:800px'>

    <div style='background:#1e3a5f;padding:20px 28px;border-radius:8px;margin-bottom:24px'>
        <h1 style='color:white;margin:0;font-size:1.4rem'>📊 Axiontech Assessment Report</h1>
        <p style='color:#93c5fd;margin:4px 0 0 0;font-size:0.85rem'>
            All-time cumulative &nbsp;|&nbsp; Finals only &nbsp;|&nbsp; Generated: {now_str}
        </p>
    </div>

    <div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:16px 20px;margin-bottom:24px;display:flex;gap:32px'>
        <div><span style='font-size:1.5rem;font-weight:700;color:#15803d'>{grand_appeared}</span>
             <span style='color:#6b7280;margin-left:6px'>Total Appeared</span></div>
        <div><span style='font-size:1.5rem;font-weight:700;color:#15803d'>{grand_passed}</span>
             <span style='color:#6b7280;margin-left:6px'>Total Passed</span></div>
        <div><span style='font-size:1.5rem;font-weight:700;color:#15803d'>{overall_pass_rate}</span>
             <span style='color:#6b7280;margin-left:6px'>Overall Pass Rate</span></div>
    </div>

    {module_blocks}

    <p style='margin-top:32px;color:#9ca3af;font-size:0.75rem;border-top:1px solid #e5e7eb;padding-top:12px'>
        Sent automatically every Tuesday &amp; Friday at 9AM IST.<br>
        Duplicate emails (abc vs abc1) are merged — best score retained.
    </p>
</body></html>"""

# ── SEND EMAIL ────────────────────────────────────────────────────
def send_email(html):
    print(f"Preparing email...")
    print(f"From: {GMAIL_USER}")
    print(f"To: {RECIPIENTS}")
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Axiontech Assessment Report — {datetime.now().strftime('%d %b %Y')}"
    msg["From"]    = GMAIL_USER
    msg["To"]      = ", ".join(RECIPIENTS)
    msg.attach(MIMEText(html, "html"))
    print("Connecting to Gmail SMTP...")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        print("Logging in...")
        s.login(GMAIL_USER, GMAIL_PASSWORD)
        print("Sending...")
        s.sendmail(GMAIL_USER, RECIPIENTS, msg.as_string())
    print(f"✅ Report sent successfully to {RECIPIENTS}")

# ── MAIN ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Fetching final assessment data...")
    data = fetch_finals()
    print(f"Found {len(data)} records across finals")
    best = aggregate(data)
    for lvl, candidates in best.items():
        print(f"  {lvl}: {len(candidates)} unique candidates")
    html = build_html(best)
    send_email(html)
