"""
questions.py — Test 1 for Beginner + Intermediate
All intermediate answers precomputed from fixed-seed dataset at import time.
"""

import hashlib, re, io, sys
import numpy as np
import pandas as pd

# ── UTILS ────────────────────────────────────────────────────────
def _norm(s): return re.sub(r'\s+', '', str(s).strip().lower())
def h(s):     return hashlib.sha256(_norm(s).encode()).hexdigest()

def _exec(code, globs=None):
    buf = io.StringIO(); old = sys.stdout; sys.stdout = buf
    exec(code, globs or {})
    sys.stdout = old
    return buf.getvalue().strip()

# ════════════════════════════════════════════════════════════════
# BEGINNER — TEST 1  (Data Structures & Loops)
# ════════════════════════════════════════════════════════════════
BEGINNER_T1 = [
    {
        "id":1, "type":"mcq", "marks":1,
        "text": "What is the output of:\n\nlen({'a': 1, 'b': 2, 'c': 3})",
        "opts": ["2", "3", "4", "6"],
        "ah":   h("3"),
    },
    {
        "id":2, "type":"mcq", "marks":1,
        "text": "Which of the following data types is mutable?",
        "opts": ["tuple", "list", "string", "int"],
        "ah":   h("list"),
    },
    {
        "id":3, "type":"fill", "marks":2,
        "text": "What is the output of:\n\n[1, 2, 3, 4, 5][1:4]\n\nType exactly as Python would print it.",
        "ah":   h("[2, 3, 4]"),
    },
    {
        "id":4, "type":"fill", "marks":2,
        "text": "What is the output of:\n\nlist(range(2, 10, 3))\n\nType exactly as Python would print it.",
        "ah":   h("[2, 5, 8]"),
    },
    {
        "id":5, "type":"fill", "marks":3,
        "text": "What is the output of:\n\n[i**2 for i in range(5) if i % 2 != 0]",
        "ah":   h("[1, 9]"),
    },
    {
        "id":6, "type":"fill", "marks":3,
        "text": (
            "What does this print?\n\n"
            "d = {}\n"
            "for i in range(5):\n"
            "    d[i] = i % 3\n"
            "print(d)"
        ),
        "ah":   h("{0: 0, 1: 1, 2: 2, 3: 0, 4: 1}"),
    },
    {
        "id":7, "type":"code", "marks":4,
        "text": (
            "Preloaded:  nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\n\n"
            "Write a one-line list comprehension to extract all even numbers.\n"
            "Store in variable 'result' and print it."
        ),
        "preload": "nums = [1,2,3,4,5,6,7,8,9,10]\n",
        "exp":     "[2, 4, 6, 8, 10]",
    },
    {
        "id":8, "type":"code", "marks":4,
        "text": (
            "Preloaded:  words = ['python', 'data', 'science', 'machine', 'learning']\n\n"
            "Build a dictionary  {word: len(word)}  and print it."
        ),
        "preload": "words = ['python','data','science','machine','learning']\n",
        "exp":     "{'python': 6, 'data': 4, 'science': 7, 'machine': 7, 'learning': 8}",
    },
    {
        "id":9, "type":"code", "marks":5,
        "text": (
            "Preloaded:  nums = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]\n\n"
            "Find the most frequent element. If tie, print the smallest.\n"
            "Print only the number."
        ),
        "preload": "nums = [3,1,4,1,5,9,2,6,5,3,5]\n",
        "exp":     "5",
    },
]

# ════════════════════════════════════════════════════════════════
# INTERMEDIATE — TEST 1  (Credit Line Assignment)
# ════════════════════════════════════════════════════════════════

# Fixed-seed dataset
np.random.seed(42)
_n = 200
_df = pd.DataFrame({
    'customer_id':          range(1001, 1201),
    'age':                  np.random.randint(22, 65, _n),
    'income':               np.random.randint(300000, 2000000, _n),
    'bureau_score':         np.random.randint(500, 850, _n),
    'existing_loans':       np.random.randint(0, 6, _n),
    'employment_type':      np.random.choice(
                                ['Salaried','Self-Employed','Business','Freelancer'],
                                _n, p=[0.5, 0.2, 0.2, 0.1]),
    'city':                 np.random.choice(
                                ['Mumbai','Delhi','Bangalore','Chennai','Hyderabad'], _n),
    'credit_line_assigned': np.random.randint(50000, 1000000, _n),
})

# Precompute MCQ / fill answers
_a1  = str(_df['employment_type'].nunique())
_a2  = _df.groupby('city')['credit_line_assigned'].mean().idxmax()
_a3  = str(_df['bureau_score'].median())
_a4  = str(_df[_df['existing_loans'] == 0].shape[0])
_a5  = _df.groupby('employment_type')['credit_line_assigned'].mean().idxmax()
_df2 = _df.copy()
_df2['income_band'] = pd.cut(_df2['income'], 3, labels=['L','M','H'])
_a6  = str(_df2['income_band'].value_counts().idxmax())

# Precompute code question expected outputs
_G = {"df": _df.copy(), "pd": pd, "np": np}

_exp7 = _exec("""
df['score_band'] = pd.cut(df['bureau_score'], 5, labels=['1','2','3','4','5'])
print(df.groupby('score_band')['credit_line_assigned'].mean().idxmax())
""", {**_G, "df": _df.copy()})

_exp8 = _exec("""
top3 = df.groupby('city')['income'].median().nlargest(3).index.tolist()
print(top3)
""", {**_G, "df": _df.copy()})

_exp9 = _exec("""
print(round(df['income'].corr(df['credit_line_assigned']), 2))
""", {**_G, "df": _df.copy()})

_exp10 = _exec("""
print(len(df[(df['bureau_score'] < 600) & (df['existing_loans'] > 2)]))
""", {**_G, "df": _df.copy()})

# Preload string injected into candidate code sandbox
DF_SETUP = """\
import pandas as pd, numpy as np
np.random.seed(42)
_n = 200
df = pd.DataFrame({
    'customer_id':          range(1001, 1201),
    'age':                  np.random.randint(22, 65, _n),
    'income':               np.random.randint(300000, 2000000, _n),
    'bureau_score':         np.random.randint(500, 850, _n),
    'existing_loans':       np.random.randint(0, 6, _n),
    'employment_type':      np.random.choice(['Salaried','Self-Employed','Business','Freelancer'], _n, p=[0.5,0.2,0.2,0.1]),
    'city':                 np.random.choice(['Mumbai','Delhi','Bangalore','Chennai','Hyderabad'], _n),
    'credit_line_assigned': np.random.randint(50000, 1000000, _n),
})
"""

INTERMEDIATE_T1 = [
    {
        "id":1, "type":"mcq", "marks":1,
        "text": "df['employment_type'].nunique()\n\nWhat does this return?",
        "opts": ["2", "3", "4", "5"],
        "ah":   h(_a1),
    },
    {
        "id":2, "type":"mcq", "marks":2,
        "text": "Which city has the highest average credit_line_assigned?\n(Use the preloaded df)",
        "opts": ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad"],
        "ah":   h(_a2),
    },
    {
        "id":3, "type":"fill", "marks":2,
        "text": (
            "What is the output of:\n\n"
            "df['bureau_score'].median()\n\n"
            "Type the exact value (e.g. 675.0)"
        ),
        "ah":   h(_a3),
    },
    {
        "id":4, "type":"fill", "marks":3,
        "text": (
            "How many customers have exactly 0 existing loans?\n\n"
            "df[df['existing_loans'] == 0].shape[0]\n\n"
            "Type the integer."
        ),
        "ah":   h(_a4),
    },
    {
        "id":5, "type":"fill", "marks":3,
        "text": (
            "What does this return?\n\n"
            "df.groupby('employment_type')['credit_line_assigned'].mean().idxmax()\n\n"
            "Type the exact string."
        ),
        "ah":   h(_a5),
    },
    {
        "id":6, "type":"mcq", "marks":4,
        "text": (
            "After running:\n\n"
            "df['income_band'] = pd.cut(df['income'], 3, labels=['L','M','H'])\n\n"
            "Which income band has the most customers?"
        ),
        "opts": ["L", "M", "H"],
        "ah":   h(_a6),
    },
    {
        "id":7, "type":"code", "marks":4,
        "text": (
            "df is preloaded (200 rows).\n\n"
            "Cut bureau_score into 5 equal-width bands, label them '1' to '5'.\n"
            "Find the band with the highest average credit_line_assigned.\n"
            "Print only the label."
        ),
        "preload": DF_SETUP,
        "exp":     _exp7,
    },
    {
        "id":8, "type":"code", "marks":4,
        "text": (
            "df is preloaded (200 rows).\n\n"
            "Find the top 3 cities ranked by median income.\n"
            "Print the result as a Python list."
        ),
        "preload": DF_SETUP,
        "exp":     _exp8,
    },
    {
        "id":9, "type":"code", "marks":4,
        "text": (
            "df is preloaded (200 rows).\n\n"
            "Compute the Pearson correlation between 'income' and 'credit_line_assigned'.\n"
            "Print rounded to 2 decimal places."
        ),
        "preload": DF_SETUP,
        "exp":     _exp9,
    },
    {
        "id":10, "type":"code", "marks":5,
        "text": (
            "df is preloaded (200 rows).\n\n"
            "Flag customers as High Risk if:\n"
            "    bureau_score < 600  AND  existing_loans > 2\n\n"
            "Print the count of High Risk customers."
        ),
        "preload": DF_SETUP,
        "exp":     _exp10,
    },
]

# ── QUESTION MAP (add Tests 2-5 here later) ──────────────────────
BEGINNER_TESTS     = {1: BEGINNER_T1}
INTERMEDIATE_TESTS = {1: INTERMEDIATE_T1}